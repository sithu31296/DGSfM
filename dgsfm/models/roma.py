import torch
from torch import nn
import numpy as np
from PIL import Image
from loguru import logger
from pathlib import Path
from torch.nn import functional as F

# romav1
from romatch import roma_indoor, roma_outdoor, tiny_roma_v1_outdoor
from romatch.utils.kde import kde
# romav2
from romav2.device import device
from romav2 import RoMaV2 as RoMaV2Model


def get_normalized_grid(
    B: int,
    H: int,
    W: int,
    overload_device: torch.device | None = None,
) -> torch.Tensor:
    x1_n = torch.meshgrid(
        *[
            torch.linspace(-1 + 1 / n, 1 - 1 / n, n, device=overload_device)
            for n in (B, H, W)
        ],
        indexing="ij",
    )
    x1_n = torch.stack((x1_n[2], x1_n[1]), dim=-1).reshape(B, H, W, 2)
    return x1_n

def sample_feats(matches, certainty, num_feats=10000, sample_thresh=0.05, expansion_factor=4):
    good_certainty = certainty.clone()
    good_certainty[certainty > sample_thresh] = 1
    matches, certainty, good_certainty = matches.reshape(-1, 4), certainty.reshape(-1), good_certainty.reshape(-1)
    good_samples = torch.multinomial(good_certainty, num_samples=min(expansion_factor * num_feats, len(certainty)), replacement=False)
    good_matches, good_certainty, certainty = matches[good_samples], good_certainty[good_samples], certainty[good_samples]

    density = kde(good_matches, std=0.1)
    p = 1 / (density + 1)
    p[density < 10] = 1e-7  # Basically should have at least 10 perfect neighbours, or around 100 ok ones

    balanced_samples = torch.multinomial(p, num_samples=min(num_feats, len(certainty)), replacement=False)
    return good_matches[balanced_samples], certainty[balanced_samples].reshape(-1, 1)


def sample_feats_v2(matches, confidence, num_feats=10000, sample_thresh=0.05):
    expansion_factor = 4
    H_A = 1280
    confidence = confidence * matches.abs().amax(dim=-1).le(1 - 1 / H_A).float()
    corresp_inds = torch.multinomial(confidence, expansion_factor * num_feats, replacement=False)
    sampled_matches, sampled_confidence = matches[corresp_inds], confidence[corresp_inds]
    density = kde(sampled_matches)
    p = 1 / (density + 1)
    p[density < 10] = 1e-7  # Basically should have at least 10 perfect neighbours, or around 100 ok ones
    balanced_samples = torch.multinomial(p, num_samples=min(num_feats, len(sampled_confidence)), replacement=False)
    sampled_matches, sampled_confidence = sampled_matches[balanced_samples], sampled_confidence[balanced_samples]
    mask = sampled_confidence >= sample_thresh
    return sampled_matches[mask], sampled_confidence[mask].reshape(-1, 1)


def to_pixel_coordinates(coords, H_A, W_A, H_B, W_B):
    kpts_A, kpts_B = coords[..., :2], coords[..., 2:]
    return _to_pixel_coordinates(kpts_A, H_A, W_A), _to_pixel_coordinates(kpts_B, H_B, W_B)

def _to_pixel_coordinates(coords, H, W):
    return torch.stack((W / 2 * (coords[..., 0] + 1), H / 2 * (coords[..., 1] + 1)), dim=-1)

def to_normalized_coordinates(coords, H_A, W_A, H_B, W_B):
    kpts_A, kpts_B = coords[..., :2], coords[..., 2:]
    return _to_normalized_coordinates(kpts_A, H_A, W_A), _to_normalized_coordinates(kpts_B, H_B, W_B)

def _to_normalized_coordinates(coords, H, W):
    return torch.stack((2 / W * coords[:, 0] - 1, 2 / H * coords[..., 1] - 1), dim=-1)



def match_keypoints(x_A, x_B, warp, certainty, max_dist=0.005, cert_th=0):
    H, W2, _ = warp.shape
    W = W2//2
    x_A = _to_normalized_coordinates(x_A, H, W)
    x_B = _to_normalized_coordinates(x_B, H, W)
    x_A_to_B = F.grid_sample(
        warp[..., -2:].permute(2, 0, 1)[None],
        x_A[None, None],
        align_corners=False,
        mode="bilinear",
    )[0, :, 0].mT
    cert_A_to_B = F.grid_sample(
        certainty[None, None, ...],
        x_A[None, None],
        align_corners=False,
        mode="bilinear",
    )[0, 0, 0]
    D = torch.cdist(x_A_to_B, x_B)
    inds_A, inds_B = torch.nonzero(
        (D == D.min(dim=-1, keepdim=True).values)
        * (D == D.min(dim=-2, keepdim=True).values)
        * (cert_A_to_B[:, None] > cert_th)
        * (D < max_dist),
        as_tuple=True,
    )
    return inds_A, inds_B


class RoMav1(nn.Module):
    def __init__(self, indoor=False, tiny=False, device=torch.device("cuda")):
        super().__init__()
        # disable RoMa logging outputs
        logger.disable("romatch")
        self.device = device
        if indoor:
            self.model = roma_indoor(device=device)
        else:
            if tiny:
                self.model = tiny_roma_v1_outdoor(device=device)
            else:
                self.model = roma_outdoor(device=device)


    def postprocess(self, warp: torch.Tensor, certainty: torch.Tensor, im1_size, im2_size):
        warp, certainty = warp.squeeze(), certainty.squeeze()
        matches, good_certainty = self.model.sample(warp, certainty, self.num_kpts)

        # convert to pixel coordinates (RoMa uses normalized coordinates [-1, 1]x[-1, 1])
        kpts1, kpts2 = self.model.to_pixel_coordinates(matches, *im1_size, *im2_size)
        kpts1, kpts2 = self.to_numpy(kpts1, self.int_dtype), self.to_numpy(kpts2, self.int_dtype)
        warp, certainty = self.to_numpy(warp, self.float_dtype), self.to_numpy(certainty, self.float_dtype)
        confs = self.to_numpy(good_certainty, self.float_dtype)

        matches = np.concatenate([kpts1, kpts2, confs[:, None]], axis=-1)
        matches = self.filter_bounds(matches, im1_size, im2_size)
        dense_matches = np.concatenate([warp, certainty[..., None]], axis=-1)
        return matches, dense_matches

    def forward(self, image1: np.ndarray, image2: np.ndarray, save_path: Path | str | None = None):
        image1, image2 = Image.fromarray(image1), Image.fromarray(image2)
        warp, certainty = self.model.match(image1, image2, device=self.device)
        warp, certainty = warp.squeeze(), certainty.squeeze()
        if save_path is not None:
            self.model.visualize_warp(warp, certainty, image1, image2, save_path=save_path)
        dense_match, certainty = self.to_numpy(warp), self.to_numpy(certainty)
        return dense_match, certainty

    def to_numpy(self, x: torch.Tensor, dtype=np.float32) -> np.ndarray:
        return x.detach().cpu().numpy().astype(dtype)

    def filter_bounds(self, matches, size1, size2):
        h1, w1 = size1
        h2, w2 = size2
        mask = (matches[:, 0] > 0) & (matches[:, 0] < w1-1) & (matches[:, 1] > 0) & (matches[:, 1] < h1-1) & \
                (matches[:, 2] > 0) & (matches[:, 2] < w2-1) & (matches[:, 3] > 0) & (matches[:, 3] < h2-1)
        return matches[mask]



class RoMav2(nn.Module):
    def __init__(self, device=torch.device("cuda")):
        super().__init__()
        # disable RoMa logging outputs
        logger.disable("romav2")
        self.device = device
        self.model = RoMaV2Model()
        self.model.apply_setting("precise")
        self.H, self.W = (self.model.H_lr, self.model.W_lr) if (self.model.H_hr is None or self.model.W_hr is None) \
                            else (self.model.H_hr, self.model.W_hr)
        self.grid = get_normalized_grid(1, self.H, self.W, self.device)[0]
        self.model.to(self.device)

    def forward(self, image1: np.ndarray, image2: np.ndarray, save_path: Path | str | None = None):
        image1 = torch.from_numpy(image1).permute(2, 0, 1).to(self.device)
        image2 = torch.from_numpy(image2).permute(2, 0, 1).to(self.device)
        preds = self.model.match(image1[None], image2[None])
        warp_AB, overlap_AB = preds['warp_AB'][0], preds['overlap_AB'][0]
        warp_BA, overlap_BA = preds['warp_BA'][0], preds['overlap_BA'][0]
        # precision_AB, precision_BA = preds['precision_AB'][0], preds['precision_BA'][0]
        warp_AB = torch.cat([self.grid, warp_AB], dim=-1).reshape(-1, 4)
        warp_BA = torch.cat([warp_BA, self.grid], dim=-1).reshape(-1, 4)
        warp = torch.cat([warp_AB, warp_BA], dim=0)
        certainty = torch.cat([overlap_AB.reshape(-1), overlap_BA.reshape(-1)], dim=0)
        if save_path is not None:
            self.model.visualize_warp(warp, certainty, image1, image2, save_path=save_path)
        dense_match, certainty = self.to_numpy(warp), self.to_numpy(certainty)
        return dense_match, certainty      

    def to_numpy(self, x: torch.Tensor, dtype=np.float32) -> np.ndarray:
        return x.detach().cpu().numpy().astype(dtype)

