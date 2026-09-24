import numpy as np
from enum import IntEnum
from dataclasses import dataclass
from dgsfm.utils.geometry import rotmat2rotvec


def _validate_index(index: int, size: int, name: str) -> int:
    if not isinstance(index, (int, np.integer)):
        raise TypeError(f"{name} index must be an integer")
    index = int(index)
    if index < 0 or index >= size:
        raise IndexError(f"{name} index {index} out of range [0, {size})")
    return index


class FocalSource(IntEnum):
    """Column indices in ``CameraBatch.focal_estimates``."""

    KNOWN = 0
    MDE = 1
    SOLVER = 2
    GRAPH = 3
    FINAL = 4


@dataclass(slots=True)
class CameraView:
    """A lightweight view of one camera stored in a ``CameraBatch``."""

    _container: "CameraBatch"
    _index: int

    @property
    def id(self):
        return int(self._container.ids[self._index])

    @property
    def focal(self):
        return self._container.focals[self._index]

    @focal.setter
    def focal(self, value):
        self._container.focals[self._index] = value

    @property
    def focal_known(self):
        return self._container.focal_estimates[self._index, FocalSource.KNOWN]

    @focal_known.setter
    def focal_known(self, value):
        self._container.focal_estimates[self._index, FocalSource.KNOWN] = value

    @property
    def focal_mde(self):
        return self._container.focal_estimates[self._index, FocalSource.MDE]

    @focal_mde.setter
    def focal_mde(self, value):
        self._container.focal_estimates[self._index, FocalSource.MDE] = value

    @property
    def focal_solver(self):
        return self._container.focal_estimates[self._index, FocalSource.SOLVER]

    @focal_solver.setter
    def focal_solver(self, value):
        self._container.focal_estimates[self._index, FocalSource.SOLVER] = value

    @property
    def focal_graph(self):
        return self._container.focal_estimates[self._index, FocalSource.GRAPH]

    @focal_graph.setter
    def focal_graph(self, value):
        self._container.focal_estimates[self._index, FocalSource.GRAPH] = value

    @property
    def focal_final(self):
        return self._container.focal_estimates[self._index, FocalSource.FINAL]

    @focal_final.setter
    def focal_final(self, value):
        self._container.focal_estimates[self._index, FocalSource.FINAL] = value

    @property
    def principal_point(self):
        return self._container.principal_points[self._index]

    @principal_point.setter
    def principal_point(self, value):
        self._container.principal_points[self._index] = value

    @property
    def intrinsic(self):
        K = np.eye(3)
        K[:2, :2] *= self.focal
        K[:2, -1] = self.principal_point
        return K

    def img2cam(self, uv, depth=1.0):
        return (uv - self.principal_point) / self.focal * depth

    def cam2img(self, xyz):
        xy = xyz[..., :2] / (xyz[..., -1:] + 1e-12)
        return xy * self.focal + self.principal_point

    def img2ray(self, uv):
        uv = self.img2cam(uv)
        uvw = np.concatenate([uv, np.ones_like(uv)[..., -1:]], axis=-1)
        uvw /= np.linalg.norm(uvw, axis=-1, keepdims=True)
        return uvw


class CameraBatch:
    """Vectorized storage for cameras using the ``[f, cx, cy]`` model."""

    def __init__(self, num_cameras: int = 0) -> None:
        if num_cameras < 0:
            raise ValueError("num_cameras must be non-negative")
        self.ids = np.arange(num_cameras, dtype=np.uint32)
        self.focals = np.zeros(num_cameras, dtype=np.float32)
        self.principal_points = np.zeros((num_cameras, 2), dtype=np.float32)
        self.focal_estimates = np.full((num_cameras, len(FocalSource)), -1.0, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def num_cameras(self) -> int:
        """Compatibility alias; the array shape is the source of truth."""
        return len(self)

    def __getitem__(self, index: int) -> CameraView:
        index = _validate_index(index, len(self), "Camera")
        return CameraView(self, index)


@dataclass(slots=True)
class ImageView:
    """A lightweight view of one image stored in an ``ImageBatch``."""

    _container: "ImageBatch"
    _index: int

    @property
    def id(self):
        return int(self._container.ids[self._index])

    @property
    def cam_id(self):
        return int(self._container.cam_ids[self._index])

    @property
    def filename(self):
        return self._container.filenames[self._index]

    @property
    def image_size(self):
        return self._container.image_sizes[self._index].tolist()

    @property
    def is_registered(self):
        return bool(self._container.is_registered[self._index])

    @is_registered.setter
    def is_registered(self, value: bool):
        self._container.is_registered[self._index] = value

    @property
    def cluster_id(self):
        return int(self._container.cluster_ids[self._index])

    @cluster_id.setter
    def cluster_id(self, value):
        self._container.cluster_ids[self._index] = value

    @property
    def world2cam(self):
        return self._container.world2cams[self._index]

    @world2cam.setter
    def world2cam(self, value: np.ndarray):
        self._container.world2cams[self._index] = value

    @property
    def rotation(self):
        return self.world2cam[:3, :3]
    
    @property
    def translation(self):
        return self.world2cam[:3, -1]
    
    @property
    def cam_center(self):
        return -self.rotation.T @ self.translation

    @property
    def axis_angle(self):
        return rotmat2rotvec(self.rotation)

    @property
    def scale(self):
        return self._container.scales[self._index]

    @scale.setter
    def scale(self, value: float):
        self._container.scales[self._index] = value

    @property
    def features(self):
        return self._container.features[self._index]

    @property
    def depths(self):
        return self._container.depths[self._index]

    @property
    def features_confs(self):
        return self._container.features_confs[self._index]

    @property
    def depths_confs(self):
        return self._container.depths_confs[self._index]

    

class ImageBatch:
    """Vectorized image metadata with per-image variable-sized observations."""

    def __init__(self, num_images: int = 0) -> None:
        if num_images < 0:
            raise ValueError("num_images must be non-negative")
        self.ids = np.arange(num_images, dtype=np.uint32)
        self.filenames = [""] * num_images
        self.cam_ids = np.zeros(num_images, dtype=np.uint32)
        self.image_sizes = np.zeros((num_images, 2), dtype=np.uint32)
        self.is_registered = np.ones(num_images, dtype=bool)

        self.features = [np.empty((0, 2), dtype=np.float32) for _ in range(num_images)]
        self.features_confs = [np.empty(0, dtype=np.float32) for _ in range(num_images)]
        self.depths = [np.empty(0, dtype=np.float32) for _ in range(num_images)]
        self.depths_confs = [np.empty(0, dtype=np.float32) for _ in range(num_images)]

        self.cluster_ids = np.zeros(num_images, dtype=np.uint32)
        self.world2cams = np.tile(np.eye(4, dtype=np.float32), (num_images, 1, 1))
        self.scales = np.ones(num_images, dtype=np.float32)
        # Absolute depth scale currently baked into each depths array. Keep
        # this separate from scales, which may hold a newly initialized or
        # optimized estimate that has not been applied yet.
        self.depth_scales_applied = np.ones(num_images, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.filenames)

    @property
    def num_images(self) -> int:
        """Compatibility alias; the array shape is the source of truth."""
        return len(self)

    def __getitem__(self, index: int) -> ImageView:
        index = _validate_index(index, len(self), "Image")
        return ImageView(self, index)

    @property
    def registered_indices(self):
        return np.flatnonzero(self.is_registered)


@dataclass(slots=True)
class TrackView:
    """A lightweight view of one track stored in a ``TrackBatch``."""

    _container: "TrackBatch"
    _index: int

    @property
    def id(self):
        return int(self._container.ids[self._index])

    @property
    def xyz(self):
        return self._container.xyzs[self._index]

    @xyz.setter
    def xyz(self, value):
        self._container.xyzs[self._index] = value

    @property
    def rgb(self):
        return self._container.rgbs[self._index]

    @rgb.setter
    def rgb(self, value):
        self._container.rgbs[self._index] = value

    @property
    def is_initialized(self):
        return bool(self._container.is_initialized[self._index])

    @is_initialized.setter
    def is_initialized(self, value):
        self._container.is_initialized[self._index] = value

    @property
    def observations(self):
        return self._container.observations[self._index]

    @observations.setter
    def observations(self, value):
        self._container.observations[self._index] = value


class TrackBatch:
    """Vectorized 3D track data with variable-length observation arrays."""

    def __init__(self, num_tracks: int = 0) -> None:
        if num_tracks < 0:
            raise ValueError("num_tracks must be non-negative")
        self.ids = np.arange(num_tracks, dtype=np.uint32)
        self.observations = [np.zeros((0, 2), dtype=np.uint32) for _ in range(num_tracks)]

        self.is_initialized = np.zeros(num_tracks, dtype=bool)
        self.xyzs = np.zeros((num_tracks, 3), dtype=np.float32)
        self.rgbs = np.zeros((num_tracks, 3), dtype=np.uint8)

    def __len__(self) -> int:
        return self.ids.shape[0]

    @property
    def num_tracks(self) -> int:
        """Compatibility alias; the array shape is the source of truth."""
        return len(self)

    def __getitem__(self, index: int) -> TrackView:
        index = _validate_index(index, len(self), "Track")
        return TrackView(self, index)

    @property
    def initialized_indices(self):
        return np.flatnonzero(self.is_initialized)

    def filter_by_mask(self, mask: np.ndarray | list[int]):
        if isinstance(mask, list):
            values = np.asarray(mask)
            if values.dtype == bool:
                mask = values
            else:
                indices = values.astype(np.int64, copy=False)
                mask = np.zeros(len(self), dtype=bool)
                mask[indices] = True
        else:
            mask = np.asarray(mask)
        if mask.dtype != bool or mask.ndim != 1 or mask.shape[0] != len(self):
            raise ValueError("mask must be a one-dimensional boolean array of track count")

        valid_indices = np.flatnonzero(mask)
        self.ids = np.arange(len(valid_indices), dtype=np.uint32)
        self.xyzs = self.xyzs[mask]
        self.rgbs = self.rgbs[mask]
        self.is_initialized = self.is_initialized[mask]
        self.observations = [self.observations[idx] for idx in valid_indices]

    def filter_tracks_by_length(self, min_num_view_per_track=2):
        valid_indices = []
        for track_id, track_obs in enumerate(self.observations):
            if len(track_obs) >= min_num_view_per_track:
                valid_indices.append(track_id)
        print(f"\tFiltered {len(self.observations)-len(valid_indices)}/{len(self.observations)} tracks with track length less than {min_num_view_per_track}")
        self.filter_by_mask(valid_indices)

    def filter_tracks_by_triangulation_angle(self, images: ImageBatch, min_angle: float = 1.0):
        """Keep tracks with at least one camera pair meeting the minimum angle."""
        thres = np.cos(np.deg2rad(min_angle))
        image_centers = -np.einsum("nij,nj->ni", images.world2cams[:, :3, :3].transpose(0, 2, 1), images.world2cams[:, :3, -1])
        valid_indices = []
        for track_id in self.ids:
            pts_calc = self.xyzs[track_id].astype(np.float64) - image_centers[np.unique(self.observations[track_id][:, 0])]
            lengths = np.linalg.norm(pts_calc, axis=1)
            valid_rays = np.isfinite(lengths) & (lengths > 0)
            if np.count_nonzero(valid_rays) < 2:
                continue
            pts_calc = pts_calc[valid_rays] / lengths[valid_rays, None]
            # abs(cos(theta)) tests min(theta, pi - theta) without arccos.
            result = np.abs(pts_calc @ pts_calc.T)
            np.fill_diagonal(result, np.inf)
            if np.any(result <= thres):
                valid_indices.append(track_id)

        print(f"\tFiltered {len(self.observations)-len(valid_indices)}/{len(self.observations)} tracks by too small triangulation angle {min_angle}.")
        self.filter_by_mask(valid_indices)

    def _observation_geometry_by_image(self, images: ImageBatch, cameras: CameraBatch):
        """Yield track IDs, observed rays and camera points in batches per image."""
        lengths = np.fromiter(map(len, self.observations), dtype=np.intp, count=len(self))
        if not np.any(lengths):
            return
        observations = np.concatenate(self.observations)
        track_ids = np.repeat(self.ids, lengths)
        order = np.argsort(observations[:, 0])
        image_ids = observations[order, 0]
        boundaries = np.flatnonzero(image_ids[1:] != image_ids[:-1]) + 1

        # Features are ragged; gather only referenced features, one image at a time.
        for indices in np.split(order, boundaries):
            image_id = observations[indices[0], 0]
            feature_ids = observations[indices, 1]
            batch_track_ids = track_ids[indices]
            camera = cameras[images.cam_ids[image_id]]
            rays = camera.img2ray(images.features[image_id][feature_ids])
            pose = images.world2cams[image_id]
            points = self.xyzs[batch_track_ids] @ pose[:3, :3].T + pose[:3, 3]
            yield batch_track_ids, rays, points

    def filter_observations_by_angle(self, images: ImageBatch, cameras: CameraBatch, max_angle: float = 1.0, min_num_view_per_track: int = 2):
        """Keep tracks whose observations all pass the angle test."""
        thres = np.cos(np.deg2rad(max_angle*2))
        valid_tracks = np.ones(len(self), dtype=bool)
        for track_ids, rays, points in self._observation_geometry_by_image(images, cameras):
            in_front = points[:, 2] >= 1e-6
            projected_rays = points[in_front]
            projected_rays /= np.linalg.norm(projected_rays, axis=1, keepdims=True)
            valid = np.zeros(len(track_ids), dtype=bool)
            valid[in_front] = np.einsum("ij,ij->i", projected_rays, rays[in_front]) > thres
            valid_tracks[track_ids[~valid]] = False

        print(f"\tFiltered {len(self)-np.count_nonzero(valid_tracks)}/{len(self)} tracks by angle error {max_angle}.")
        self.filter_by_mask(valid_tracks)

    def filter_observations_by_reproj_error(self, images: ImageBatch, cameras: CameraBatch, max_reproj_error: float = 1e-2):
        """Keep tracks whose observations all pass the normalized reprojection test."""
        valid_tracks = np.ones(len(self), dtype=bool)
        for track_ids, rays, points in self._observation_geometry_by_image(images, cameras):
            in_front = points[:, 2] >= 1e-6
            points = points[in_front]
            rays = rays[in_front]
            observed = rays[:, :2] / (rays[:, 2:] + 1e-10)
            projected = points[:, :2] / (points[:, 2:] + 1e-10)
            valid = np.zeros(len(track_ids), dtype=bool)
            valid[in_front] = np.linalg.norm(projected - observed, axis=1) < max_reproj_error
            valid_tracks[track_ids[~valid]] = False

        print(f"\tFiltered {len(self)-np.count_nonzero(valid_tracks)}/{len(self)} tracks by reprojection error above {max_reproj_error}.")
        self.filter_by_mask(valid_tracks)
