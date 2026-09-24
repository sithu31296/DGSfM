import torch
import numpy as np
from torch import Tensor




class NNMatcher:
    def __init__(self) -> None:
        pass

    def to_numpy(self, x: Tensor) -> np.ndarray:
        return x.detach().cpu().numpy()

    def to_tensor(self, x: np.ndarray) -> Tensor:
        return torch.from_numpy(x)

    def compute_similarity(self, des1: Tensor, des2: Tensor) -> Tensor:
        """
        Compute similarity matrix between two sets of L2 normalized descriptors.
        """
        return des1 @ des2.T

    def __call__(self, des1: Tensor, des2: Tensor, matcher_type: str = "mutual_nn_ratio", ratio=0.8) -> np.ndarray:
        sim = self.compute_similarity(des1, des2)
        if matcher_type == "mutual_nn_ratio":
            ids1, ids2, mask = self.mutual_nn_ratio_matcher(sim, ratio)
        elif matcher_type == "mutual_nn":
            ids1, ids2, mask = self.mutual_nn_matcher(sim)
        elif matcher_type == "ratio":
            ids1, ids2, mask = self.ratio_matcher(sim, ratio)
        else:
            raise ValueError(f"Unknown matcher type: {matcher_type}")
        
        matches = torch.stack([ids1[mask], ids2[mask]], dim=-1)
        return self.to_numpy(matches)

    def compute_nn_ratios(self, sim: Tensor) -> np.ndarray:
        """
        Compute the ratio of the nearest neighbor distance to the second nearest neighbor distance.
        """
        # Retrieve top 2 nearest neighbors 1->2.
        nns_sim, nns = torch.topk(sim, 2, dim=1)
        nns_dist = torch.sqrt(2 - 2 * nns_sim)
        # Compute Lowe's ratio.
        ratios12 = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
        # Save first NN.
        nn12 = nns[:, 0]

        # Retrieve top 2 nearest neighbors 1->2.
        nns_sim, nns = torch.topk(sim.t(), 2, dim=1)
        nns_dist = torch.sqrt(2 - 2 * nns_sim)
        # Compute Lowe's ratio.
        ratios21 = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
        # Save first NN.
        nn21 = nns[:, 0]
        return nn12, ratios12, nn21, ratios21
        
    def ratio_matcher(self, sim: Tensor, ratio=0.8) -> np.ndarray:
        """
        Symmetric Lowe's ratio test matcher for L2 normalized descriptors.
        """
        nn12, ratios12, nn21, ratios21 = self.compute_nn_ratios(sim)
        # Symmetric ratio test.
        ids1 = torch.arange(0, sim.shape[0], device=sim.device)
        mask = torch.min(ratios12 <= ratio, ratios21[nn12] <= ratio)
        return ids1, nn12, mask
    
    def mutual_nn_ratio_matcher(self, sim: Tensor, ratio=0.8):
        """
        Mutual NN + symmetric Lowe's ratio test matcher for L2 normalized descriptors.
        """
        nn12, ratios12, nn21, ratios21 = self.compute_nn_ratios(sim)
        # Mutual NN + symmetric ratio test.
        ids1 = torch.arange(0, sim.shape[0], device=sim.device)
        mask = torch.min(ids1 == nn21[nn12], torch.min(ratios12 <= ratio, ratios21[nn12] <= ratio))
        return ids1, nn12, mask

    def mutual_nn_matcher(self, sim: Tensor):
        """
        Mutual Nearest Neighbors Matcher for L2 normalized descriptors
        """
        nn12 = torch.max(sim, dim=1)[1]
        nn21 = torch.max(sim, dim=0)[1]
        ids1 = torch.arange(0, sim.shape[0], device=sim.device)
        mask = ids1 == nn21[nn12]
        # matches = torch.stack([ids1[mask], nn12[mask]]).t()
        return ids1, nn12, mask

    