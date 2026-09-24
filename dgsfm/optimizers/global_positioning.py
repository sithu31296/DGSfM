import numpy as np
import torch
import networkx as nx
import pypose as pp
from torch import nn
from pypose.autograd.function import psjac
from dgsfm.optimizers.base import BAEOptimizer
from dgsfm.database.viewgraph import ViewGraph
from dgsfm.database.structures import CameraBatch, ImageBatch, TrackBatch
from dgsfm.database.config import DGSfMConfig


class GPModel(nn.Module):
    def __init__(self, cam_centers, points_3d, scales, optimize_points=True, optimize_cams=True, optimize_scales=True, fixed_cam_idx=None):
        super().__init__()
        self.scales = pp.Parameter(scales, sjac=True) if optimize_scales else scales
        self.points_3d = pp.Parameter(points_3d, sjac=True) if optimize_points else points_3d
        indices = torch.arange(len(cam_centers), device=cam_centers.device)
        free = torch.ones(len(cam_centers), dtype=torch.bool, device=cam_centers.device)
        if fixed_cam_idx is not None:
            free[fixed_cam_idx] = False
            indices[fixed_cam_idx + 1:] -= 1
            indices[fixed_cam_idx] = len(cam_centers) - 1
        self.register_buffer("center_indices", indices)
        self.register_buffer("fixed_cam_center", cam_centers[~free].detach().clone())
        if optimize_cams and free.any():
            self.free_cam_centers = pp.Parameter(cam_centers[free], sjac=True)
        else:
            self.register_buffer("free_cam_centers", cam_centers[free].detach().clone())

    @property
    def cam_centers(self):
        # Only free centers are parameters; restore registered-image order for indexing.
        return torch.cat((self.free_cam_centers, self.fixed_cam_center))[self.center_indices]

    def forward(self, rays, cidx, pidx, weights):
        return weights * (rays - GPModel.compute(self.points_3d[pidx], self.cam_centers[cidx], self.scales))

    @psjac
    def compute(points, centers, scales):
        return scales * (points - centers)


class GlobalPositioning(BAEOptimizer):
    def __init__(self, cfg: DGSfMConfig):
        super().__init__(cfg.mapping.gp_num_iters, cfg.mapping.min_num_views_per_track, cfg.device)
        self.optimize_points = True
        self.optimize_cams = True
        self.optimize_scales = True
        self.init_strategy = cfg.mapping.gp_init
        self.fixed_cam_id = None

    def initialize(self, viewgraph: ViewGraph, tracks: TrackBatch, cameras: CameraBatch, images: ImageBatch):
        if viewgraph.fixed_cam_id not in viewgraph.graph:
            viewgraph.set_fixed_cam_params(images)
        self.fixed_cam_id = viewgraph.fixed_cam_id
        if self.init_strategy == 'random':
            self.initialize_randomly(images, tracks)
            images.world2cams[self.fixed_cam_id, :3, 3] = viewgraph.fixed_cam_trans
        else:
            self.initialize_from_maximum_spanning_tree(viewgraph, cameras, images, tracks)

    def initialize_randomly(self, images: ImageBatch, tracks: TrackBatch):
        images.world2cams[:, :3, 3] = 100 * np.random.uniform(-1, 1, (len(images), 3))
        tracks.xyzs = 100 * np.random.uniform(-1, 1, (len(tracks), 3))
        tracks.is_initialized[:] = True

    def initialize_from_maximum_spanning_tree(self, view_graph: ViewGraph, cameras: CameraBatch, images: ImageBatch, tracks: TrackBatch):
        ids = np.flatnonzero(images.is_registered)
        if view_graph.fixed_cam_id not in view_graph.graph:
            view_graph.set_fixed_cam_params(images)
        self.fixed_cam_id = view_graph.fixed_cam_id
        image_ts = {view_graph.fixed_cam_id: view_graph.fixed_cam_trans[:, None]}
        tree = nx.maximum_spanning_tree(view_graph.graph, weight="weight")
        for parent, child in nx.bfs_edges(tree, view_graph.fixed_cam_id):
            pair = view_graph._pair_for(parent, child)
            # Translation uses the stored image1's depth scale in either direction.
            pair_t = pair.translation[:, None] * images[pair.image_id1].scale
            pair_rot = images[child].rotation @ images[parent].rotation.T
            if pair.image_id1 == parent:
                relative_t = pair_t
            else:
                relative_t = -pair_rot @ pair_t
            image_ts[child] = pair_rot @ image_ts[parent] + relative_t
        images.world2cams[ids, :3, -1:] = np.array([image_ts[i] for i in ids])

        if not len(tracks):
            return
        counts = np.fromiter(map(len, tracks.observations), dtype=np.intp, count=len(tracks))
        nonempty = counts > 0
        tracks.xyzs[:] = np.nan
        if np.any(nonempty):
            image_ids, feature_ids = np.concatenate(tracks.observations).T
            # Flatten ragged feature arrays once, then gather all track observations.
            feature_counts = np.fromiter(map(len, images.features), dtype=np.intp, count=len(images))
            feature_offsets = np.cumsum(feature_counts) - feature_counts
            feature_indices = feature_offsets[image_ids] + feature_ids
            features = np.concatenate(images.features)[feature_indices]
            depths = np.concatenate(images.depths)[feature_indices]
            cam_ids = images.cam_ids[image_ids]
            xy = (features - cameras.principal_points[cam_ids]) / cameras.focals[cam_ids, None]
            points_cam = np.concatenate((xy, np.ones((len(xy), 1), dtype=xy.dtype)), axis=1) * depths[:, None]
            points_world = np.einsum(
                "nji,nj->ni", images.world2cams[image_ids, :3, :3],
                points_cam - images.world2cams[image_ids, :3, 3],
            )
            # Observations remain contiguous by track, so reduce each segment to its mean.
            starts = np.cumsum(counts) - counts
            sums = np.add.reduceat(points_world, starts[nonempty], axis=0)
            tracks.xyzs[nonempty] = sums / counts[nonempty, None]
        tracks.is_initialized[:] = True

    def optimize(self, cameras: CameraBatch, images: ImageBatch, tracks: TrackBatch):
        if not len(tracks):
            return
        counts = np.fromiter(map(len, tracks.observations), dtype=np.intp, count=len(tracks))
        image_ids, feature_ids = np.concatenate(tracks.observations).T
        registered = images.is_registered[image_ids]
        if not np.any(registered):
            return
        point_indices = np.repeat(tracks.ids, counts)[registered]
        image_ids, feature_ids = image_ids[registered], feature_ids[registered]

        # Unobserved variables create empty columns in the sparse Jacobian.
        used_ids, image_indices = np.unique(image_ids, return_inverse=True)
        used_point_ids, point_indices = np.unique(point_indices, return_inverse=True)
        self.image_idx2id = dict(enumerate(used_ids))
        image_id2idx = {image_id: index for index, image_id in self.image_idx2id.items()}

        # Pack only observed images, so discarded images need no feature/confidence data.
        feature_counts = np.fromiter((len(images.features[i]) for i in used_ids), dtype=np.intp)
        feature_offsets = np.cumsum(feature_counts) - feature_counts
        feature_indices = feature_offsets[image_indices] + feature_ids
        features = np.concatenate([images.features[i] for i in used_ids])[feature_indices]
        feature_confs = np.concatenate([images.features_confs[i] for i in used_ids])[feature_indices]
        depth_confs = np.concatenate([images.depths_confs[i] for i in used_ids])[feature_indices]
        weights = feature_confs * depth_confs
        cam_ids = images.cam_ids[image_ids]
        xy = (features - cameras.principal_points[cam_ids]) / cameras.focals[cam_ids, None]
        cam_rays = np.concatenate((xy, np.ones((len(xy), 1), dtype=xy.dtype)), axis=1)
        cam_rays /= np.linalg.norm(cam_rays, axis=1, keepdims=True)
        global_rays = np.einsum("nji,nj->ni", images.world2cams[image_ids, :3, :3], cam_rays)
        if self.fixed_cam_id is None or not images[self.fixed_cam_id].is_registered:
            raise ValueError("Initialize global positioning with a registered root image before optimizing")
        fixed_cam_idx = image_id2idx.get(self.fixed_cam_id)
        fixed_translation = images[self.fixed_cam_id].translation.copy()

        cam2worlds = np.linalg.inv(images.world2cams[used_ids])
        cam_centers = self.to_tensor(cam2worlds[:, :3, 3], torch.float64)
        points_3d = self.to_tensor(tracks.xyzs[used_point_ids], torch.float64)
        scales = torch.ones(len(global_rays), 1, dtype=torch.float64, device=self.device)

        model = GPModel(cam_centers, points_3d, scales, self.optimize_points, self.optimize_cams,
                        self.optimize_scales, fixed_cam_idx=fixed_cam_idx).to(self.device)
        bae_inputs = {
            "rays": self.to_tensor(global_rays, torch.float64),
            "cidx": self.to_tensor(image_indices, torch.int32),
            "pidx": self.to_tensor(point_indices, torch.int32),
            "weights": self.to_tensor(weights, torch.float64)[:, None]
        }
        self.run_iters(bae_inputs, model, "Global Positioning")
        tracks.xyzs[used_point_ids] = self.to_numpy(model.points_3d)
        self.update_points_and_cams(images, tracks, centers=model.cam_centers)
        # Preserve the anchor exactly through the center-to-translation conversion.
        images.world2cams[self.fixed_cam_id, :3, 3] = fixed_translation
