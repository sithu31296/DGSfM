import networkx as nx
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import bmat, csr_matrix, eye
from scipy.sparse.linalg import lsmr
from scipy.spatial.transform import Rotation
from tqdm.auto import tqdm
from dgsfm.database.structures import ImageBatch
from dgsfm.database.viewgraph import ViewGraph


class RotationAveraging:
    """MST initialization followed by L1, Geman-McClure IRLS, or both.

    Relative rotations map camera 1 to camera 2: Rij = Rj @ Ri.T.
    L1 minimizes componentwise absolute tangent residuals using HiGHS;
    IRLS weights each edge by its full angular error. Both use the usual
    small-angle incidence approximation and backtracking on the actual loss.
    The fixed camera is excluded from the unknowns. Pair weights are used
    only to choose the spanning tree.
    """

    def __init__(self):
        self.max_num_l1_iters = 10
        self.max_num_irls_iters = 100
        self.l1_step_convergence_thres = 1e-5
        self.irls_step_convergence_thres = 1e-5
        self.irls_loss_param_sigma = 5.0  # Degrees.
        self.cost_history = {}

    def _initialize(self, view_graph, images):
        ids = np.flatnonzero(images.is_registered)
        pairs = [view_graph._pair_for(i, j) for i, j in view_graph.graph.edges]
        if view_graph.fixed_cam_id not in view_graph.graph: 
            view_graph.set_fixed_cam_params(images)
        rotations = {view_graph.fixed_cam_id: Rotation.from_matrix(view_graph.fixed_cam_rot)}
        tree = nx.maximum_spanning_tree(view_graph.graph, weight="weight")
        for parent, child in nx.bfs_edges(tree, view_graph.fixed_cam_id):
            pair = view_graph._pair_for(parent, child)
            relative = Rotation.from_matrix(pair.rotation)
            if pair.image_id1 == child:
                relative = relative.inv()
            rotations[child] = relative * rotations[parent]
        return ids, pairs, Rotation.concatenate([rotations[i] for i in ids])

    def _refine(self, rotations, relative, i, j, incidence, free, mode, iterations, tolerance):
        def residual(estimate):
            return (estimate[j].inv() * relative * estimate[i]).as_rotvec()

        sigma2 = np.deg2rad(self.irls_loss_param_sigma) ** 2

        def cost(error):
            if mode == "l1":
                return np.abs(error).sum()
            squared = np.sum(error**2, axis=1)
            return np.sum(squared / (squared + sigma2))

        m, n = incidence.shape
        if mode == "l1":
            # min sum(t), subject to -t <= A @ step - error <= t.
            constraints = bmat([[incidence, -eye(m)], [-incidence, -eye(m)]], format="csr")
            objective = np.r_[np.zeros(n), np.ones(m)]
            bounds = [(None, None)] * n + [(0, None)] * m

        error = residual(rotations)
        history = self.cost_history[mode] = [float(cost(error))]
        with tqdm(total=iterations, desc=f"Rotation Averaging ({mode.upper()})",
                  postfix={"loss": f"{history[-1]:.6g}"}) as progress:
            for _ in range(iterations):
                step = np.zeros((len(free), 3))
                if mode == "irls":
                    # sqrt(Geman–McClure weight); a common scale is immaterial.
                    sqrt_weight = sigma2 / (np.sum(error**2, axis=1) + sigma2)
                    weighted = incidence.multiply(sqrt_weight[:, None]).tocsr()
                for axis in range(3):
                    if mode == "l1":
                        result = linprog(objective, A_ub=constraints,
                                         b_ub=np.r_[error[:, axis], -error[:, axis]],
                                         bounds=bounds, method="highs")
                        if not result.success:
                            return None
                        step[free, axis] = result.x[:n]
                    else:
                        result = lsmr(weighted, sqrt_weight * error[:, axis],
                                      atol=1e-8, btol=1e-8, maxiter=max(100, 3*n))
                        if result[1] not in {0, 1, 2, 4, 5}:
                            return None
                        step[free, axis] = result[0]
                if not np.all(np.isfinite(step)):
                    return None
                progress.update(1)
                step_size = np.linalg.norm(step[free], axis=1).mean()
                if step_size < tolerance:
                    break

                # The incidence model is local: accept only a decrease on SO(3).
                for backtrack in range(16):
                    fraction = 0.5**backtrack
                    candidate = rotations * Rotation.from_rotvec(fraction * step)
                    candidate_error = residual(candidate)
                    candidate_cost = float(cost(candidate_error))
                    if candidate_cost < history[-1]:
                        rotations, error = candidate, candidate_error
                        history.append(candidate_cost)
                        progress.set_postfix(loss=f"{candidate_cost:.6g}")
                        break
                else:
                    break  # No improving step; keep the last accepted estimate.
                if fraction * step_size < tolerance:
                    break
        return rotations

    def optimize(self, view_graph: ViewGraph, images: ImageBatch) -> bool:
        """Write a finite estimate, returning False on a linear-solver failure.

        Invalid/disconnected inputs raise ValueError. Iteration limits or a
        stalled line search retain the best accepted estimate; True does not
        certify convergence or a global optimum. Set a stage's iteration count
        to zero to skip it. No image poses are changed on solver failure.
        """
        self.cost_history = {}
        ids, pairs, rotations = self._initialize(view_graph, images)
        index = {image_id: k for k, image_id in enumerate(ids)}
        free = ids != view_graph.fixed_cam_id
        if pairs and np.any(free):
            i = np.array([index[p.image_id1] for p in pairs])
            j = np.array([index[p.image_id2] for p in pairs])
            rows = np.arange(len(pairs))
            incidence = csr_matrix(
                (np.r_[-np.ones(len(pairs)), np.ones(len(pairs))],
                 (np.r_[rows, rows], np.r_[i, j])), shape=(len(pairs), len(ids)),
            )[:, free]
            relative = Rotation.from_matrix(np.stack([p.rotation for p in pairs]))
            stages = [
                ("l1", self.max_num_l1_iters, self.l1_step_convergence_thres),
                ("irls", self.max_num_irls_iters, self.irls_step_convergence_thres),
            ]
            for mode, iterations, tolerance in stages:
                rotations = self._refine(rotations, relative, i, j, incidence, free, mode, iterations, tolerance)

        images.world2cams[ids, :3, :3] = rotations.as_matrix()
        images.world2cams[view_graph.fixed_cam_id, :3, :3] = view_graph.fixed_cam_rot
