import numpy as np



def points_registration(src_pts: np.ndarray, tgt_pts: np.ndarray, weights: np.ndarray = None, compute_scaling: bool = False):
    if weights is None:
        weights = np.ones(src_pts.shape[0])
    
    src_pts_mean = np.mean(weights[..., None] * src_pts, axis=0)
    tgt_pts_mean = np.mean(weights[..., None] * tgt_pts, axis=0)

    src_pts_centered = src_pts - src_pts_mean
    tgt_pts_centered = tgt_pts - tgt_pts_mean
    weights /= (weights.sum() + 1e-12)

    cov = (weights[:, None] * src_pts_centered).T @ tgt_pts_centered
    U, S, Vh = np.linalg.svd(cov)
    R = Vh.T @ U.T
    if np.linalg.det(R) < 0:
        Vh[2, :] *= -1
        R = Vh.T @ U.T

    if compute_scaling:
        scale = np.sum(S) / np.trace((weights[:, None] * src_pts_centered).T @ src_pts_centered)
        t = tgt_pts_mean - scale * (src_pts_mean @ R.T)
        return R, t, scale
    else:
        t = tgt_pts_mean - (src_pts_mean @ R.T)
        return R, t, 1
    


