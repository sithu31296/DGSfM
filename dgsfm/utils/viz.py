import torch
import numpy as np
# import open3d as o3d
import trimesh
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from torch.nn import functional as F
from typing import List, Any, Optional, Tuple, Union
from scipy.spatial.transform import Rotation
from dgsfm.utils.geometry import geometric_transform



Cam_to_Trimesh = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
])

OPENGL = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
])




def todevice(batch, device, callback=None, non_blocking=False):
    if callback:
        batch = callback(batch)

    if isinstance(batch, dict):
        return {k: todevice(v, device) for k, v in batch.items()}

    if isinstance(batch, (tuple, list)):
        return type(batch)(todevice(x, device) for x in batch)

    x = batch
    if device == 'numpy':
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    elif x is not None:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if torch.is_tensor(x):
            x = x.to(device, non_blocking=non_blocking)
    return x


def to_numpy(x): return todevice(x, 'numpy')
def to_cpu(x): return todevice(x, 'cpu')
def to_cuda(x, gpu_id=0): return todevice(x, f'cuda:{gpu_id}')

def to_image(x: Union[torch.Tensor, Image.Image, np.ndarray]) -> Image.Image:
    if isinstance(x, Image.Image):
        return x

    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    if x.max() <= 1.01:
        x *= 255

    if x.shape[0] == 3:
        x = x.transpose((1, 2, 0))
    return Image.fromarray(x.astype(np.uint8))


def to_image_tensor(x: Union[torch.Tensor, Image.Image, np.ndarray], normalize: bool = False) -> torch.Tensor:
    if isinstance(x, Image.Image):
        x = torch.from_numpy(np.asarray(x).copy()).to(torch.float32)
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)

    if normalize and (x.max() >= 1.01):
        x /= 255.0

    if x.shape[2] == 3:
        x = x.permute(2, 0, 1)
    return x



def stack_tensors(vecs):
    """
    Arguments:
        vecs: list of tensors in shape (H, W, 3)
    Returns:
        stacked tensors in shape (NxHxW, 3)
    """
    if isinstance(vecs, (np.ndarray, torch.Tensor)):
        vecs = [vecs]
    return np.concatenate([p.reshape(-1, 3) for p in to_numpy(vecs)])


class PlotlyViewer:
    def __init__(self):
        pass


    def init_figure(height: int = 800) -> go.Figure:
        """Initialize a 3D figure."""
        fig = go.Figure()
        axes = dict(
            visible=False,
            showbackground=False,
            showgrid=False,
            showline=False,
            showticklabels=True,
            autorange=True,
        )
        fig.update_layout(
            template="plotly_dark",
            height=height,
            scene_camera=dict(
                eye=dict(x=0., y=-.1, z=-2),
                up=dict(x=0, y=-1., z=0),
                projection=dict(type="orthographic")),
            scene=dict(
                xaxis=axes,
                yaxis=axes,
                zaxis=axes,
                aspectmode='data',
                dragmode='orbit',
            ),
            margin=dict(l=0, r=0, b=0, t=0, pad=0),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.1
            ),
        )
        return fig

    def plot_points(
            fig: go.Figure,
            pts: np.ndarray,
            color: str = 'rgba(255, 0, 0, 1)',
            ps: int = 2,
            colorscale: Optional[str] = None,
            name: Optional[str] = None):
        """Plot a set of 3D points."""
        x, y, z = pts.T
        tr = go.Scatter3d(
            x=x, y=y, z=z, mode='markers', name=name, legendgroup=name,
            marker=dict(
                size=ps, color=color, line_width=0.0, colorscale=colorscale))
        fig.add_trace(tr)

    def plot_interactive_pointcloud(xyz, rgb, subsampling_rate=0.05):
        # plots a plotly pointcloud
        # params:
        # xyz - n x 3 - array with 3D coordinates of points
        # rgb - n x 3 - array with RGB triplets in 0-255
        n = xyz.shape[0]

        l = np.random.rand(n) <= subsampling_rate
        fig = init_figure()
        plot_points(fig, xyz[l], color=rgb[l])
        fig.show()


class Open3DViewer:
    def __init__(self):
        pass


    def array2pcd(self, points, colors):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors) # Colors should be in [0,1]

    def plot_interactive_pointcloud(xyz, rgb):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)

        # Visualize using Plotly
        o3d.visualization.draw_plotly([pcd], point_sample_factor=0.05, window_name="Point Cloud Viewer", width=800, height=600)


class TrimeshViewer:
    def __init__(self):
        self.point_size = 2
        self.cam_size = 1
        self.image_size = None  # in (width, height)
        self.subsample_rate = 1.0

        self.scene = trimesh.Scene()

        self.cam_colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 0, 255), (255, 204, 0), (0, 204, 204),
              (128, 255, 255), (255, 128, 255), (255, 255, 128), (0, 0, 0), (128, 128, 128)]

    def set_params(self, params: dict):
        for k, v in params.items():
            setattr(self, k, v)

    def preprocess(self, points, colors):
        if self.subsample_rate == 1.0:
            return stack_tensors(points), stack_tensors(colors)
        new_points, new_colors = [], []
        for idx in range(len(points)):
            sample_mask = np.random.rand(len(points[idx])) < self.subsample_rate
            sample_mask = sample_mask & (~np.any(np.isinf(points[idx]), axis=1))
            new_points.append(points[idx][sample_mask])
            new_colors.append(colors[idx][sample_mask])
        return stack_tensors(new_points), stack_tensors(new_colors)


    def visualize_points(self, points, colors):
        assert len(points) == len(colors)
        if isinstance(points, np.ndarray):
            points, colors = [points], [colors]
        self.add_points_and_colors(points, colors)
        self.scene.show(
            line_settings={"point_size": self.point_size}
        )

    def visualize_cams(self, poses, focals=None, intrinsics=None):
        if isinstance(poses, np.ndarray):
            poses = [poses]
        for cam_idx in range(len(poses)):
            cam_color = self.cam_colors[cam_idx % len(self.cam_colors)]
            # if colors[cam_idx].size/3 == self.image_size[0] * self.image_size[1]:
            #     image = colors[cam_idx].reshape(*self.image_size, 3)
            # else:
            #     image = None
            image = None
            if focals is not None:
                if isinstance(focals, list):
                    focal = focals[cam_idx]
                else:
                    focal = focals
            elif intrinsics is not None:
                if isinstance(intrinsics, list):
                    focal = intrinsics[cam_idx][0, 0]
                else:
                    focal = intrinsics[0, 0]
            else:
                focal = None
            self.add_camera(poses[cam_idx], focal, image, cam_color)
        self.scene.show(
            line_settings={"point_size": self.point_size}
        )

    def visualize(self, points, colors, poses=None, focals=None, intrinsics=None):
        assert len(points) == len(colors)
        if isinstance(points, np.ndarray):
            points, colors = [points], [colors]
        self.add_points_and_colors(points, colors)

        if poses is not None:
            if isinstance(poses, np.ndarray):
                poses = [poses]
            for cam_idx in range(len(poses)):
                cam_color = self.cam_colors[cam_idx % len(self.cam_colors)]
                # if colors[cam_idx].size/3 == self.image_size[0] * self.image_size[1]:
                #     image = colors[cam_idx].reshape(*self.image_size, 3)
                # else:
                #     image = None
                image = None
                if focals is not None:
                    if isinstance(focals, list):
                        focal = focals[cam_idx]
                    else:
                        focal = focals
                elif intrinsics is not None:
                    if isinstance(intrinsics, list):
                        focal = intrinsics[cam_idx][0, 0]
                    else:
                        focal = intrinsics[0, 0]
                else:
                    focal = None
                # focal = None
                self.add_camera(poses[cam_idx], focal, image, cam_color)

        self.scene.show(
            line_settings={"point_size": self.point_size}
        )

    def add_points_and_colors(self, points, colors):
        vertices, colors = self.preprocess(points, colors)
        pcd = trimesh.PointCloud(vertices, colors)
        self.add_geometry(pcd)

    def add_geometry(self, geometry):
        self.scene.add_geometry(geometry, transform=Cam_to_Trimesh)

    def save_to_glb(self, file_path: Union[str, Path]):
        # # Apply 180° rotation around X-axis to fix orientation (upside-down issue)
        # rotation_matrix_x = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
        # scene_3d.apply_transform(rotation_matrix_x)
        self.scene.export(file_path)


    def add_camera(self, pose, focal=None, image=None, cam_color=None, marker=None):
        # pose is in camera to world
        # image is in [H, W, 3] format or [3, H, W] format
        if image is not None:
            image = to_image(image)
            W, H = image.size
        elif self.image_size is not None:
            W, H = self.image_size
        elif focal is not None:
            W = H = focal / 1.1
        else:
            W = H = 1

        if cam_color is None:
            cam_color = self.cam_colors[0]

        if isinstance(focal, np.ndarray):
            focal = focal[0]

        if focal is None:
            focal = min(H, W) * 1.1

        # create fake camera
        height = max(self.cam_size / 10, focal * self.cam_size / H)
        width = self.cam_size*0.5**0.5
        rot45 = np.eye(4)
        rot45[:3, :3] = Rotation.from_euler("z", np.deg2rad(45)).as_matrix()
        rot45[2, 2] = height   # set the tip of the cone = optical center

        aspect_ratio = np.eye(4)
        aspect_ratio[0, 0] = W/H

        transform = pose @ OPENGL @ aspect_ratio @ rot45
        cam = trimesh.creation.cone(width, height, sections=4)

        if image is not None:
            vertices = geometric_transform(cam.vertices[[4, 5, 1, 3]], transform)
            faces = np.array([[0, 1, 2], [0, 2, 3], [2, 1, 0], [3, 2, 0]])
            img = trimesh.Trimesh(vertices=vertices, faces=faces)
            uv_coords = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
            img.visual = trimesh.visual.TextureVisuals(uv_coords, image=image)
            self.add_geometry(img)

        # this is the camera mesh
        rot2 = np.eye(4)
        rot2[:3, :3] = Rotation.from_euler('z', np.deg2rad(2)).as_matrix()
        vertices = np.r_[cam.vertices, 0.95*cam.vertices, geometric_transform(cam.vertices, rot2)]
        vertices = geometric_transform(vertices, transform)
        faces = []
        for face in cam.faces:
            if 0 in face:
                continue
            a, b, c = face
            a2, b2, c2 = face + len(cam.vertices)
            a3, b3, c3 = face + 2*len(cam.vertices)

            # add 3 pseudo-edges
            faces.append((a, b, b2))
            faces.append((a, a2, c))
            faces.append((c2, b, c))

            faces.append((a, b, b3))
            faces.append((a, a3, c))
            faces.append((c3, b, c))

        # no culling
        faces += [(c, b, a) for a, b, c in faces]

        cam = trimesh.Trimesh(vertices=vertices, faces=faces)
        cam.visual.face_colors[:, :3] = cam_color
        self.add_geometry(cam)

        if marker == 'o':
            marker = trimesh.creation.icosphere(3, radius=self.cam_size/4)
            marker.vertices += pose[:3,3]
            marker.visual.face_colors[:,:3] = cam_color
            self.add_geometry(marker)



class MatplotlibViewer:
    def __init__(self):
        pass

    def save_image(self, image, save_path: Union[str, Path]):
        image = to_image(image)
        image.save(save_path)

    def save_figure(self, save_path, dpi=300):
        plt.tight_layout(pad=0.3)
        plt.savefig(save_path, dpi=dpi)
        plt.close()

    def visualize_image(self, image):
        image = to_image(image)

        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
        plt.show()
        plt.close()

    def show_figure(self):
        plt.show()
        plt.close()

    def plot_features(self, image, feats, save_path):
        plt.imshow(image)
        plt.scatter(feats[:, 0], feats[:, 1], s=0.8, c='red')
        plt.xticks([])
        plt.yticks([])
        self.save_figure(save_path, dpi=200)


    def plot_features_correspondences(self, image1, image2, kpts1, kpts2, save_path):
        h0, w0, _ = image1.shape
        h1, w1, _ = image2.shape

        new_img0, new_img1 = image1.copy(), image2.copy()
        offset_w, offset_h = 0, 0
        img0_moved, img1_moved = False, False

        if w1 > w0 or h1 > h0:
            offset_w = w1 - w0
            new_img0 = np.zeros((h1, w1, 3), dtype=np.uint8)
            new_img0[:h0, offset_w:w0, :] = image1
            img0_moved = True

        elif w0 > w1 or h0 > h1:
            new_img1 = np.zeros((h0, w0, 3), dtype=np.uint8)
            new_img1[:h1, :w1, :] = image2
            img1_moved = True

        img = np.concatenate([new_img0, new_img1], axis=1)
        offset = img.shape[1] / 2
        plt.imshow(img)
        for p1, p2 in zip(kpts1, kpts2):
            if img0_moved:
                plt.scatter([p1[0]+offset_w, p2[0] + offset], [p1[1]+offset_h, p2[1]], s=0.8)
                plt.plot([p1[0], p2[0] + offset], [p1[1], p2[1]], linewidth=1, markersize=3)
            elif img1_moved:
                plt.scatter([p1[0], p2[0] + offset + offset_w], [p1[1], p2[1] + offset_h], s=0.8)
                plt.plot([p1[0], p2[0] + offset], [p1[1], p2[1]], linewidth=1, markersize=3)
            else:
                plt.scatter([p1[0], p2[0] + offset], [p1[1], p2[1]], s=0.8)
                plt.plot([p1[0], p2[0] + offset], [p1[1], p2[1]], linewidth=1, markersize=3)
        plt.xticks([])
        plt.yticks([])
        self.save_figure(save_path, dpi=300)


    def visualize_pixels_matches(self, image1, image2, warp, certainty, save_path=None):
        H, W, _ = image1.shape

        x1, x2 = to_image_tensor(image1, normalize=True), to_image_tensor(image2, normalize=True)
        warp1, warp2 = torch.tensor(warp[:, W:, :2]), torch.tensor(warp[:, :W, 2:])

        x1_to_x2 = F.grid_sample(x1[None], warp1[None], mode='bilinear', align_corners=False).squeeze()
        x2_to_x1 = F.grid_sample(x2[None], warp2[None], mode='bilinear', align_corners=False).squeeze()

        x12 = torch.cat([x2_to_x1, x1_to_x2], dim=2).numpy()
        x12 = certainty[None] * x12 + (1 - certainty[None]) * np.ones_like(x12)
        if save_path is not None:
            self.save_image(x12, save_path)
        else:
            self.visualize_image(x12)


class ReRunViewer:
    def __init__(self):
        pass


class ViserViewer:
    def __init__(self):
        pass