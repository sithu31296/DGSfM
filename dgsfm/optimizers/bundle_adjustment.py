import torch
from torch import nn, Tensor
import numpy as np
import pypose as pp
from pypose.autograd.function import psjac
from dgsfm.optimizers.base import BAEOptimizer
from dgsfm.database.structures import ImageBatch, CameraBatch, TrackBatch
from dgsfm.database.config import DGSfMConfig


class BAModel(nn.Module):
    def __init__(self, F, C, P, optimize_focals=True, optimize_poses=True, optimize_points=True, fixed_cam_idx=None):
        super().__init__()
        self.F = pp.Parameter(F, sjac=True) if optimize_focals else F
        self.P = pp.Parameter(P, sjac=True) if optimize_points else P
        indices = torch.arange(len(C), device=C.device)
        free = torch.ones(len(C), dtype=torch.bool, device=C.device)
        if fixed_cam_idx is not None:
            free[fixed_cam_idx] = False
            indices[fixed_cam_idx + 1:] -= 1
            indices[fixed_cam_idx] = len(C) - 1
        self.register_buffer("pose_indices", indices)
        self.register_buffer("fixed_pose", C[~free].detach().clone())
        if optimize_poses and free.any():
            self.free_poses = pp.Parameter(C[free], sjac=True)
        else:
            self.register_buffer("free_poses", C[free].detach().clone())

    @property
    def C(self):
        # Keep the root constant and restore registered-image order for indexing.
        return torch.cat((self.free_poses, self.fixed_pose))[self.pose_indices]

    def forward(self, observe: Tensor, point_indices: Tensor, image_indices: Tensor, camera_indices: Tensor):
        return BAModel.project(self.F[camera_indices], self.C[image_indices], self.P[point_indices]) - observe

    @psjac
    def project(F, C, P):
        cp = C.Act(P)
        cpn = cp[..., :2] / (cp[..., -1:] + 1e-12)
        return cpn * F


class BundleAdjusterBAE(BAEOptimizer):
    def __init__(self, cfg: DGSfMConfig):
        super().__init__(cfg.mapping.ba_num_iters, cfg.mapping.min_num_views_per_track, cfg.device)
        self.optimize_focals = True
        self.optimize_poses = True
        self.optimize_points = True

    def solve(self, cameras: CameraBatch, images: ImageBatch, tracks: TrackBatch, *, fixed_cam_id: int):
        if not len(tracks): return
        counts = np.fromiter(map(len, tracks.observations), dtype=np.intp, count=len(tracks))
        image_ids, feature_ids = np.concatenate(tracks.observations).T
        registered = images.is_registered[image_ids]
        if not np.any(registered): return
        point_indices = np.repeat(tracks.ids, counts)[registered]
        image_ids, feature_ids = image_ids[registered], feature_ids[registered]

        # Filtering can leave registered images (and their intrinsics) without any
        # observations. Exclude unused variables to avoid empty columns in J.
        used_ids, image_indices = np.unique(image_ids, return_inverse=True)
        cam_ids = images.cam_ids[image_ids]
        used_cam_ids, camera_indices = np.unique(cam_ids, return_inverse=True)
        used_point_ids, point_indices = np.unique(point_indices, return_inverse=True)
        self.image_idx2id = dict(enumerate(used_ids))
        self.camera_idx2id = dict(enumerate(used_cam_ids))
        image_id2idx = {image_id: index for index, image_id in self.image_idx2id.items()}

        # Pack only observed images, preserving observation order within each track.
        feature_counts = np.fromiter((len(images.features[i]) for i in used_ids), dtype=np.intp)
        feature_offsets = np.cumsum(feature_counts) - feature_counts
        feature_indices = feature_offsets[image_indices] + feature_ids
        features = np.concatenate([images.features[i] for i in used_ids])[feature_indices].astype(np.float32)
        points_2d = features - cameras.principal_points[cam_ids]

        fixed_pose = images[fixed_cam_id].world2cam.copy()
        if not images[fixed_cam_id].is_registered:
            raise ValueError("Bundle adjustment requires a registered root image")
        camera_focals = self.to_tensor(cameras.focals[used_cam_ids], torch.float64)[:, None]
        image_poses = self.to_tensor(pp.mat2SE3(images.world2cams[used_ids]), torch.float64)
        points_3d = self.to_tensor(tracks.xyzs[used_point_ids], torch.float64)

        bae_inputs = {
            "observe": self.to_tensor(points_2d, torch.float64),
            "point_indices": self.to_tensor(point_indices, torch.int32),
            "image_indices": self.to_tensor(image_indices, torch.int32),
            "camera_indices": self.to_tensor(camera_indices, torch.int32)
        }
        model = BAModel(camera_focals, image_poses, points_3d, self.optimize_focals,
                        self.optimize_poses, self.optimize_points, fixed_cam_idx=image_id2idx.get(fixed_cam_id)).to(self.device)

        self.run_iters(bae_inputs, model, "Bundle Adjustment")
        tracks.xyzs[used_point_ids] = self.to_numpy(model.P)
        self.update_intrinsics_and_extrinsics(cameras, images, model.F, model.C)
        # Avoid numerical changes to the anchor during SE(3)-to-matrix conversion.
        images.world2cams[fixed_cam_id] = fixed_pose

