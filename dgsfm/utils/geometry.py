import torch
import math
import numpy as np
from torch import nn, Tensor
from typing import List, Tuple, Dict, Union
from scipy.spatial.transform import Rotation as R


EPS = 1e-15

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def focal2intrinsic(focal, cx, cy):
    return np.array([
        [focal, 0, cx],
        [0, focal, cy],
        [0, 0, 1]
    ], dtype=np.float32)

def vector_to_skew(x: np.ndarray):
    x = x.ravel()
    return np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ])

def constrain_depths(depths1, depths2, lower=5, upper=90):
    inlier_mask = (depths1 > np.percentile(depths1, lower)) & (depths1 <= np.percentile(depths1, upper))
    inlier_mask &= (depths2 > np.percentile(depths2, lower)) & (depths2 <= np.percentile(depths2, upper))
    return inlier_mask

def is_consistent_with_focal_mde(f1, f2, f_mde1, f_mde2, valid_focal_ratio=0.2):
    image1_const = (1-valid_focal_ratio) < (f1 / f_mde1) < (1+valid_focal_ratio)
    image2_const = (1-valid_focal_ratio) < (f2 / f_mde2) < (1+valid_focal_ratio)
    return image1_const and image2_const


def normalize(x: np.ndarray):
    return x / (norm(x) + EPS)


def norm(x: Union[np.ndarray, torch.Tensor]):
    if isinstance(x, np.ndarray):
        return np.linalg.norm(x)
    else:
        return torch.linalg.norm(x)


def angle_between_two_vectors(x: np.ndarray, y: np.ndarray):
    x, y = x.ravel(), y.ravel()
    x_norm, y_norm = norm(x), norm(y)

    dot_product = np.dot(x, y)
    magnitude = x_norm * y_norm
    angle = np.arccos(np.clip(dot_product / magnitude, -1, 1))
    return np.rad2deg(angle)


def rotmat2rotvec(rot_mat: np.ndarray) -> np.ndarray:
    return R.from_matrix(rot_mat).as_rotvec()

def project(points, intrinsic):
    return perspective_devision(points @ intrinsic.T)

def perspective_devision(points):
    return points[:, :2] / (points[:, -1:] + 1e-12)


def rotmat2quat(rot_mat: np.ndarray, scalar_first=False) -> np.ndarray:
    """
    Docstring for rotmat2quat

    :param rot_mat: Rotation matrix
    :type rot_mat: np.ndarray
    :param scalar_first: order of scalar part [w]
        scalar-first order -> (w, x, y, z)
        scalar-last order (default) -> (x, y, z, w)
    :return: Description
    :rtype: ndarray
    """
    r = R.from_matrix(rot_mat)
    return r.as_quat(scalar_first=scalar_first)

def rotvec2rotmat(rotvec):
    return R.from_rotvec(rotvec).as_matrix()


def quat2rotmat(qvec):
    return R.from_quat(qvec).as_matrix()


def quat2mat(qvec, tvec):
    rot = R.from_quat(qvec).as_matrix()
    mat = np.eye(4)
    mat[:3, :3] = rot
    mat[:3, -1] = tvec
    return mat

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )



def rotation_angle_from_rotation_matrix(R: np.ndarray):
    trace = np.trace(R)
    angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
    return np.rad2deg(angle)

def get_relative_pose(R1: np.ndarray, R2: np.ndarray, t1: np.ndarray, t2: np.ndarray):
    R = R2 @ R1.T
    t = -R @ t1 + t2
    return R, t


def fundamental_from_pose(
    R1: np.ndarray, R2: np.ndarray,
    t1: np.ndarray, t2: np.ndarray,
    K1: np.ndarray, K2: np.ndarray
):
    """
    E = K2.T x F x K1
    F = K2.T-1 x E x K1-1
    """
    E = essential_from_pose(R1, R2, t1, t2)
    F = np.linalg.inv(K2.T) @ E @ np.linalg.inv(K1)
    return F

def essential_from_pose(
    R1: np.ndarray, R2: np.ndarray,
    t1: np.ndarray, t2: np.ndarray
):
    """
    E = [t]xR
    """
    R, t = get_relative_pose(R1, R2, t1, t2)
    E = vector_to_skew(t) @ R
    return E


def make_homogeneous(x):
    # x should be in shape [N, d]
    if x.shape[0] > x.shape[-1]:    # N > d
        ones = np.ones_like(x)[..., -1:]
        return np.concatenate([x, ones], axis=-1)
    else:
        ones = np.ones_like(x)[-1:, ...]
        return np.concatenate([x, ones], axis=0)


def get_liners_keypoints(x1, x2, F, threshold=2.0):
    x1, x2 = make_homogeneous(x1), make_homogeneous(x2)
    line1_in_2 = x1 @ F.T
    line2_in_1 = x2 @ F

    numerator = np.sum(x2 * line1_in_2, axis=1) ** 2
    denominator = line1_in_2[:, 0]**2 + line1_in_2[:, 1]** 2 + line2_in_1[:, 0]**2 + line2_in_1[:, 1]**2
    out = numerator / denominator
    return out < threshold ** 2

def add_rand_pts(x, width, height, multiplier=1):
    x_new = np.random.rand(int(multiplier * len(x)), 2)
    x_new[:, 0] *= width
    x_new[:, 1] *= height
    return np.concatenate([x, x_new], axis=0)

def force_inliers(x1, x2, R, t, K1, K2, ratio, threshold):
    F = np.linalg.inv(K2.T) @ vector_to_skew(t) @ R @ np.linalg.inv(K1)
    inliers = get_liners_keypoints(x1, x2, F, threshold)
    x1, x2 = x1[inliers], x2[inliers]

    multiplier = (1 - ratio) / ratio

    x1_new = add_rand_pts(x1, K1[0, 2]/2, K1[1, 2]/2, multiplier)
    x2_new = add_rand_pts(x2, K2[0, 2]/2, K2[1, 2]/2, multiplier)

    return x1_new, x2_new

def make_grid(height, width):
    """Create mesh grid

    Returns:
        grid: (H*W, 2)
    """
    uu, vv = np.meshgrid(np.arange(width), np.arange(height))
    grid = np.vstack([uu.flatten(), vv.flatten()])
    return grid


def create_grid(height, width):
    """Create mesh grid
    Returns:
        grid: (H, W, 2)
    """
    # # uu, vv = np.meshgrid(np.arange(width), np.arange(height))
    # uu, vv = np.meshgrid(np.linspace(0, width-1, width), np.linspace(0, height-1, height))
    # return np.stack([uu, vv], axis=-1).astype(np.uint32)

    uu = np.linspace(0, width-1, width).reshape(1, width).repeat(height, axis=0)
    vv = np.linspace(0, height-1, height).reshape(height, 1).repeat(width, axis=1)
    return np.stack([uu, vv], axis=-1).astype(np.uint32)

def make_pointcloud(depth, K, image=None, flatten=True):
    height, width = depth.shape
    grid = make_grid(height, width)
    grid_homo = make_homogeneous(grid)
    K_inv = np.linalg.inv(K)
    z = depth.reshape(-1, 1)
    points = z * (K_inv @ grid_homo).T
    if not flatten:
        return points.reshape(height, width, 3), image
    return points, image.reshape(-1, 3)

def intersectnd(A: np.ndarray, B: np.ndarray, assume_unique=False):
    Av = A.view([('', A.dtype)] * A.shape[1]).ravel()
    Bv = B.view([('', B.dtype)] * B.shape[1]).ravel()

    C, x_ind, y_ind = np.intersect1d(Av, Bv, return_indices=True, assume_unique=assume_unique)
    C = C.view(A.dtype).reshape(-1, A.shape[1])
    return C, x_ind, y_ind


def points_img2cam(pixels, depths, focal, principal_point):
    cams = (pixels - principal_point) / focal
    return depths[:, None] * make_homogeneous(cams)

def geometric_transform(points, trf_mat, norm=False):
    """Apply a geometric transformation to a list of 3D points.
    Arguments:
        points: list of arrays of tensors, shape must be in (..., 2) or (..., 3)
        trf_mat: array or tensor in shape (3, 3) or (4, 4)
    Returns:
        Transformed points
    """
    assert trf_mat.ndim >= 2

    if isinstance(trf_mat, torch.Tensor):
        trf_mat = trf_mat.cpu().numpy()

    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()

    # adapt shape if necessary
    output_shape = points.shape

    if trf_mat.ndim >= 3:   # (B, 3, 3) or (B, 4, 4) batched transformations
        assert trf_mat.shape[0] == points.shape[0], "batch size does not match"

        trf_mat = trf_mat.reshape(-1, trf_mat.shape[-2], trf_mat.shape[-1])

        if points.ndim > trf_mat.ndim:  # points (B, H, W, 3) and trf_mat (B, 3/4, 3/4)
            points = points.reshape(points.shape[0], -1, points.shape[-1])
        elif points.ndim == 2:
            points = points[:, None, :]
    else:
        if points.ndim > trf_mat.ndim:
            points = points.reshape(-1, points.shape[-1])

    trf_mat = trf_mat.swapaxes(-1, -2)
    if points.shape[-1] + 1 == trf_mat.shape[-1]:
        points = points @ trf_mat[..., :-1, :] + trf_mat[..., -1:, :]
    else:
        points = points @ trf_mat

    return points[..., :output_shape[-1]].reshape(*output_shape)



def sample_depth_at_pixel_numpy(depth_map, pixel_coords, image_w, image_h, method='nearest'):
    H, W = depth_map.shape
    x, y = pixel_coords[:, 0], pixel_coords[:, 1]
    x_norm = (x / image_w).clip(0.0, 1.0)
    y_norm = (y / image_h).clip(0.0, 1.0)

    # convert to size of depth map
    x_converted, y_converted = x_norm * (W - 1), y_norm * (H - 1)

    if method == 'nearest':
        x_int, y_int = x_converted.astype(int), y_converted.astype(int)
        depth = depth_map[y_int, x_int]
    else:   # bilinear
        x0 = np.floor(x_converted).astype(int)
        x1 = (x0 + 1).clip(0, W-1)
        y0 = np.floor(y_converted).astype(int)
        y1 = (y0 + 1).clip(0, H-1)

        wx = x_converted - x0
        wy = y_converted - y0

        d00 = depth_map[y0, x0]
        d01 = depth_map[y1, x0]
        d10 = depth_map[y0, x1]
        d11 = depth_map[y1, x1]

        depth = (d00 * (1-wx) * (1-wy) +
                 d10 * wx * (1-wy) +
                 d01 * (1-wx) * wy +
                 d11 * wx * wy)
    return depth


def normalize_coords(coords, x, y):
    xs = 2.0 * (coords[..., :1] / (x - 1)) - 1.0
    ys = 2.0 * (coords[..., 1:] / (y - 1)) - 1.0
    return torch.cat([xs, ys], dim=1)

def sample_depth_at_pixel(depth_map: Tensor, pixel_coords: Tensor, image_w: int, image_h: int, method='nearest'):
    H, W = depth_map.shape
    N, _ = pixel_coords.shape
    depth_map = depth_map.view(1, 1, H, W)
    pixel_coords = normalize_coords(pixel_coords, image_w, image_h).view(1, 1, N, 2)
    sampled_depth = nn.functional.grid_sample(depth_map, pixel_coords, mode=method, align_corners=False)
    return sampled_depth.squeeze()



def E_from_pose(qvec, tvec):
    # get the essential matrix from relative pose
    rot = R.from_quat(qvec).as_matrix()
    E = np.array([
        [0, -tvec[2], tvec[1]],
        [tvec[2], 0, -tvec[0]],
        [-tvec[1], tvec[0], 0]
    ])
    E = E @ rot
    return E


def F_from_pose_and_cam(qvec, tvec, K1, K2):
    # get the fundamental matrix from relative pose
    E = E_from_pose(qvec, tvec)
    F = np.linalg.inv(K2.T) @ E @ np.linalg.inv(K1)
    return F

def is_physically_plausible(focal, image_width, lower=0.5, upper=5.0):
    # most cameras have a FoV between 10 deg and 100 deg
    # in pixels, focal is usually between 0.5x and 5x the image width
    lower_bound = lower * image_width
    upper_bound = upper * image_width
    return lower_bound < focal < upper_bound

def isRotationMatrix(R):
    # square matrix test
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        return False
    should_be_identity = np.allclose(R.dot(R.T), np.identity(R.shape[0], np.float32))
    should_be_one = np.allclose(np.linalg.det(R), 1)
    return should_be_identity and should_be_one

def homography_error(H, x1, x2):
    # homography error for the homography matrix
    Hx1 = H @ np.hstack([x1, 1])
    Hx1_norm = Hx1[:2] / (Hx1[2] + EPS)
    return np.sum((Hx1_norm - x2)**2)



def sampson_error(E, x1, x2):
    # sampson error for the essential matrix
    # input the normalized image coordinates
    # x can be undistorted as well as distorted
    x1 = np.hstack([x1, 1]) if len(x1) == 2 else x1
    x2 = np.hstack([x2, 1]) if len(x2) == 2 else x2
    Ex1 = np.dot(E, x1) / (x1[2] + EPS)
    Etx2 = np.dot(E.T, x2) / (x2[2] + EPS)

    C = np.dot(x2, Ex1)
    Cx = Ex1[0] * Ex1[0] + Ex1[1] * Ex1[1]
    Cy = Etx2[0] * Etx2[0] + Etx2[1] * Etx2[1]
    return C**2 / (Cx + Cy)

def check_cheirality(qvec, tvec, x1, x2, min_depth=0., max_dept=100.):
    # cheirality check for essential matrix
    rot = R.from_quat(qvec).as_matrix()
    Rx1 = rot @ x1

    a = -Rx1 @ x2
    b1 = -Rx1 @ tvec
    b2 = x2 @ tvec  # tvec @ x2

    lambda1 = b1 - a @ b2
    lambda2 = -a @ b1 + b2
    min_depth = min_depth * (1 - a**2)
    max_depth = max_depth * (1 - a**2)
    return lambda1 > min_depth and lambda2 > min_depth and lambda1 < max_depth and lambda2 < max_depth

def get_orientation_signum(F, epipole, pt1, pt2):
    # get the orientation signum for fundamental matrix
    # for chierality check of fundamental matrix
    signum1 = F[0, 0] * pt2[0] + F[1, 0] * pt2[1] + F[2, 0]
    signum2 = epipole[1] - epipole[2] * pt1[1]
    return signum1 * signum2


def normalize_vector(x: np.ndarray, eps: float = 1e-10):
    return x / max(eps, float(np.linalg.norm(x)))


def calculate_trans_between_poses(pose1: np.ndarray, pose2: np.ndarray):
    # calculate the center difference between two poses
    assert pose1.shape[0] == 4
    assert pose2.shape[0] == 4
    cam1, cam2 = np.linalg.inv(pose1), np.linalg.inv(pose2)
    return np.linalg.norm(cam1[:3, -1] - cam2[:3, -1])


def calculate_angle_between_poses(pose1: np.ndarray, pose2: np.ndarray):
    cos_r = normalize_vector(pose1) @ normalize_vector(pose2)
    cos_r = np.clip(cos_r, -1., 1.)
    return np.rad2deg(np.arccos(cos_r))


def calculate_trans_angle_between_poses(pose1: np.ndarray, pose2: np.ndarray):
    # calculate the translation direction difference between two poses
    cos_r = (pose1[:3, -1] @ pose2[:3, -1]) / \
        (np.linalg.norm(pose1[:3, -1]) * np.linalg.norm(pose2[:3, -1]))
    cos_r = np.clip(cos_r, -1., 1.)
    return np.rad2deg(np.arccos(cos_r))


def calculate_angle_between_translations(t1: np.ndarray, t2: np.ndarray):
    # calculate the translation direction difference between two poses
    cos_r = (t1 @ t2) / (np.linalg.norm(t1) * np.linalg.norm(t2))
    cos_r = np.clip(cos_r, -1., 1.)
    return np.rad2deg(np.arccos(cos_r))

def calculate_angle_between_rotations(r1: np.ndarray, r2: np.ndarray):
    # calculate the rotation angle difference between two rotations
    r12 = r1.T @ r2
    return get_rotation_angle(r12)


def get_rotation_angle(rot_mat):
    cos_r = (np.trace(rot_mat) - 1) / 2
    cos_r = np.clip(cos_r, -1., 1.)
    return np.rad2deg(np.arccos(cos_r))

def is_valid_focal(f1, f2, width1, width2, lower=0.4, upper=4.0):
    valid_focal = is_physically_plausible(f1, width1, lower, upper) and is_physically_plausible(f2, width2, lower, upper)
    return valid_focal

def lift_and_transform(feats, depths, intrinsic, rotation, translation, depth_scale=1.0):
    points = depth_scale * depths[:, None] * (np.hstack([feats, np.ones((feats.shape[0], 1))]) @ np.linalg.inv(intrinsic).T)
    return (points @ rotation.T) + translation

