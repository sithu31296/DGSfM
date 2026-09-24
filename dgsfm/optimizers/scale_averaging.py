import math
import networkx as nx
import numpy as np
from tqdm.auto import tqdm
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix
from dgsfm.database.structures import ImageBatch
from dgsfm.database.viewgraph import ViewGraph
from dgsfm.database.config import DGSfMConfig


class ScaleAveraging:
    def __init__(self, cfg: DGSfMConfig):
        self.num_iters = cfg.mapping.sa_num_iters

    def get_registered_image_indices(self, images: ImageBatch) -> dict[int, int]:
        image_ids = np.flatnonzero(images.is_registered)
        self.image_idx2id = dict(enumerate(image_ids))
        return {image_id: index for index, image_id in self.image_idx2id.items()}

    def update_scales_and_depths(self, images: ImageBatch, scales: np.ndarray) -> None:
        image_indices = np.asarray(list(self.image_idx2id.values()), dtype=np.intp)
        new_scales = np.asarray(scales, dtype=images.scales.dtype).reshape(-1)
        applied_scales = images.depth_scales_applied[image_indices]
        for image_id, scale_update in zip(image_indices, new_scales / applied_scales):
            images.depths[image_id] *= scale_update
        images.scales[image_indices] = new_scales
        images.depth_scales_applied[image_indices] = new_scales

    def initialize_from_maximum_spanning_tree(self, view_graph: ViewGraph, images: ImageBatch):
        ids = np.flatnonzero(images.is_registered)
        if view_graph.fixed_cam_id not in view_graph.graph: 
            view_graph.set_fixed_cam_params(images)
        image_scales = {view_graph.fixed_cam_id: view_graph.fixed_cam_scale}
        tree = nx.maximum_spanning_tree(view_graph.graph, weight="weight")
        for parent, child in nx.bfs_edges(tree, view_graph.fixed_cam_id):
            pair = view_graph._pair_for(parent, child)
            pair_scale = pair.scale
            if pair.image_id1 == child:
                pair_scale = 1 / pair_scale
            image_scales[child] = pair_scale * image_scales[parent]
        images.scales[ids] = np.array([image_scales[i] for i in ids])
    
    def optimize(self, view_graph: ViewGraph, images: ImageBatch):
        image_id2idx = self.get_registered_image_indices(images)
        fixed_idx = image_id2idx[view_graph.fixed_cam_id]
        fixed_cam_scale = view_graph.fixed_cam_scale
        relative_scales, indices1, indices2, weights = [], [], [], []
        for pair in view_graph.image_pairs.values():
            if not pair.is_valid or pair.image_id1 not in image_id2idx or pair.image_id2 not in image_id2idx: continue
            relative_scales.append(pair.scale)
            indices1.append(image_id2idx[pair.image_id1])
            indices2.append(image_id2idx[pair.image_id2])
            weights.append(math.sqrt(pair.weight))

        weights = np.asarray(weights, dtype=np.float64)
        fixed_log_scale = math.log(fixed_cam_scale)
        log_scales = np.log(images.scales[images.is_registered].astype(np.float64))
        log_scales += fixed_log_scale - log_scales[fixed_idx]
        log_scales[fixed_idx] = fixed_log_scale
        free_mask = np.arange(len(log_scales)) != fixed_idx

        rows = np.arange(len(relative_scales))
        # Each edge has only two derivatives: -sqrt(weight), +sqrt(weight).
        incidence = csr_matrix(
            (np.concatenate((-weights, weights)),
                (np.concatenate((rows, rows)), np.concatenate((indices1, indices2)))),
            shape=(len(rows), len(log_scales)),
        )
        jacobian = incidence[:, free_mask]
        # Use the dense solver for the single-variable case.
        if jacobian.shape[1] == 1:
            jacobian = jacobian.toarray()

        targets = weights * np.log(relative_scales)
        targets -= incidence[:, fixed_idx].toarray().ravel() * fixed_log_scale

        with tqdm(desc="Scale Averaging", total=self.num_iters) as pbar:
            def callback(intermediate_result):
                pbar.update(1)
                pbar.set_postfix(cost=f"{intermediate_result.cost:.6g}")

            result = least_squares(
                lambda free_scales: jacobian @ free_scales - targets,
                log_scales[free_mask],
                jac=lambda _: jacobian.copy(),
                method="trf",
                loss="huber",
                f_scale=1.0,
                ftol=1e-6,
                xtol=1e-6,
                gtol=1e-6,
                max_nfev=self.num_iters,
                callback=callback,
            )

        log_scales[free_mask] = result.x

        scales = np.exp(log_scales)
        scales[fixed_idx] = fixed_cam_scale
        self.update_scales_and_depths(images, scales)


