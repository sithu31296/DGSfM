import torch
import numpy as np
from tqdm import trange
from torch import nn, Tensor
import pypose as pp
from pypose.optim import LM
from pypose.optim.solver import PCG
from pypose.optim.kernel import Huber
from dgsfm.database.structures import ImageBatch, CameraBatch, TrackBatch

FLOAT_DTYPE = np.float32
INT_DTYPE = np.int32


class BAEOptimizer:
    def __init__(self, max_num_iters, min_num_views_per_track, device=torch.device("cuda")):
        self.device = device
        self.window_size = 4
        self.func_tol = 1.e-6
        self.num_iters = max_num_iters
        self.min_num_views_per_track = min_num_views_per_track
        self.reject_counts = 30
        self.strategy = pp.optim.strategy.TrustRegion(radius=1e2, max=1e10, down=0.5**4)
        self.sparse_solver = PCG(tol=1e-5)
        self.huber_kernel = Huber(1.0)

    def to_tensor(self, x: list | np.ndarray | Tensor, dtype=torch.float32) -> Tensor:
        if isinstance(x, Tensor):
            return x.to(self.device, dtype=dtype)
        return torch.tensor(np.array(x), dtype=dtype, device=self.device)

    def to_numpy(self, x: Tensor, dtype=np.float64) -> np.ndarray:
        return x.detach().cpu().numpy().astype(dtype)
    
    def filter_tracks_and_images(self, images: ImageBatch, tracks: TrackBatch) -> None:
        # filter out tracks with too few observations
        valid_tracks_mask = np.array([((len(track.observations) >= self.min_num_views_per_track) and track.is_initialized) for track in tracks], dtype=bool)
        # filter tracks in-place to maintain reference semantics
        tracks.filter_by_mask(valid_tracks_mask)

        # filter out images that have no tracks and cameras that have no images
        image_used = np.zeros(len(images), dtype=bool)
        for track in tracks:
            image_used[np.unique(track.observations[:, 0])] = True
            if all(image_used): break
        images.is_registered &= image_used

    def get_registered_image_indices(self, images: ImageBatch) -> tuple[dict, dict]:
        image_id2idx, image_idx2id = {}, {}
        for image in images:
            if image.is_registered:
                image_id2idx[image.id] = len(image_id2idx)
                image_idx2id[len(image_idx2id)] = image.id
        self.image_idx2id = image_idx2id
        return image_id2idx

    def get_registered_camera_indices(self, cameras: CameraBatch, images: ImageBatch) -> tuple[dict, dict]:
        camera_used = np.zeros(len(cameras), dtype=bool)
        registered_cam_ids = images.cam_ids[images.is_registered]
        camera_used[registered_cam_ids] = True

        camera_id2idx, camera_idx2id = {}, {}
        for camera in cameras:
            if camera_used[camera.id]:
                camera_id2idx[camera.id] = len(camera_id2idx)
                camera_idx2id[len(camera_idx2id)] = camera.id
        self.camera_idx2id = camera_idx2id
        return camera_id2idx, camera_used


    def get_optimizer(self, model: nn.Module):
        return LM(model, strategy=self.strategy, solver=self.sparse_solver, kernel=self.huber_kernel, reject=self.reject_counts, sparse=True)

    def run_iters(self, inputs: dict[str, Tensor], model: nn.Module, pbar_text="Optimization") -> None:
        if self.num_iters <= 0:
            raise ValueError("Optimizer iteration count must be positive")
        optimizer = self.get_optimizer(model)
        loss_history = []
        progress_bar = trange(self.num_iters, desc=pbar_text)
        for _ in progress_bar:
            loss = optimizer.step(inputs)
            loss_history.append(loss.item())
            if len(loss_history) >= 2*self.window_size:
                avg_recent = np.mean(loss_history[-self.window_size:])
                avg_previous = np.mean(loss_history[-2*self.window_size:-self.window_size])
                denominator = max(abs(avg_previous), np.finfo(float).eps)
                improvement = (avg_previous - avg_recent) / denominator
                if abs(improvement) < self.func_tol:
                    break
            progress_bar.set_postfix({"loss": loss.item()})
        progress_bar.close()
        return loss_history[-1]

    @torch.no_grad()
    def update_points_and_cams(self, images: ImageBatch, tracks: TrackBatch, points=None, centers=None):
        if points is not None:
            tracks.xyzs = self.to_numpy(points, FLOAT_DTYPE)
        if centers is not None:
            centers = self.to_numpy(centers, FLOAT_DTYPE)
            image_indices = np.array(list(self.image_idx2id.values()))
            images.world2cams[image_indices, :3, 3] = -np.einsum('nij,nj->ni', images.world2cams[image_indices, :3, :3], centers)

    @torch.no_grad()
    def update_intrinsics_and_extrinsics(self, cameras: CameraBatch, images: ImageBatch, focals=None, poses=None):
        image_indices = np.array(list(self.image_idx2id.values()))
        images.world2cams[image_indices] = pp.SE3(poses).matrix().cpu().numpy().astype(FLOAT_DTYPE)
        if focals is not None:
            camera_indices = np.array(list(self.camera_idx2id.values()))
            cameras.focals[camera_indices] = self.to_numpy(focals.squeeze(), FLOAT_DTYPE)

    @torch.no_grad()
    def update_scales_and_depths(self, images: ImageBatch, scales: Tensor) -> None:
        image_indices = np.array(list(self.image_idx2id.values()))
        new_scales = self.to_numpy(scales, FLOAT_DTYPE).reshape(-1)
        if np.any(~np.isfinite(new_scales)) or np.any(new_scales <= 0):
            raise ValueError("Depth scales must be finite and positive")
        applied_scales = images.depth_scales_applied[image_indices]
        if np.any(~np.isfinite(applied_scales)) or np.any(applied_scales <= 0):
            raise ValueError("Applied depth scales must be finite and positive")
        scale_updates = new_scales / applied_scales
        for img_idx, scale_update in zip(image_indices, scale_updates):
            images.depths[img_idx] *= scale_update
        images.scales[image_indices] = new_scales
        images.depth_scales_applied[image_indices] = new_scales
