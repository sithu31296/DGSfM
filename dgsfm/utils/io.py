import torch
import cv2
import re
import h5py
import yaml
import PIL.Image
import numpy as np
# import open3d as o3d
from plyfile import PlyData, PlyElement
import pillow_heif
# from box import Box
from torch import Tensor
from pathlib import Path
from PIL import ExifTags, Image, TiffTags
from pillow_heif import register_heif_opener
from typing import Dict, Any, List, Union


# Spherical harmonic constant
C0 = 0.28209479177387814

register_heif_opener()

img_extensions = ['.png', '.jpg', '.jpeg', 'heif', 'heic']



def rgb_to_spherical_harmonic(rgb):
    return (rgb-0.5) / C0


def spherical_harmonic_to_rgb(sh):
    return sh*C0 + 0.5


def save_ply(path, means, scales, rotations, rgbs, opacities, normals=None):
    if normals is None: normals = np.zeros_like(means)
    colors = rgb_to_spherical_harmonic(rgbs)
    if scales.shape[1] == 1: scales = np.tile(scales, (1, 3))

    attrs = ['x', 'y', 'z',
             'nx', 'ny', 'nz',
             'f_dc_0', 'f_dc_1', 'f_dc_2',
             'opacity',
             'scale_0', 'scale_1', 'scale_2',
             'rot_0', 'rot_1', 'rot_2', 'rot_3',]

    dtype_full = [(attribute, 'f4') for attribute in attrs]
    elements = np.empty(means.shape[0], dtype=dtype_full)

    attributes = np.concatenate((means, normals, colors, opacities, scales, rotations), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)
    print(f"Saved PLY format Splat to {path}")


def ply_to_points(file):
    pcd = o3d.io.read_point_cloud(file)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) * 255
    return points, colors.astype(np.uint8)


def points_to_ply(points: np.ndarray, colors: np.ndarray, save_path: str):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(save_path, pcd)


def extract_exif(img_pil: Image) -> Dict[str, Any]:
    """Return exif information as a dictionary.

    Args:
    ----
        img_pil: A Pillow image.

    Returns:
    -------
        A dictionary with extracted EXIF information.

    """
    # Get full exif description from get_ifd(0x8769):
    # cf https://pillow.readthedocs.io/en/stable/releasenotes/8.2.0.html#image-getexif-exif-and-gps-ifd
    img_exif = img_pil.getexif().get_ifd(0x8769)
    exif_dict = {ExifTags.TAGS[k]: v for k, v in img_exif.items() if k in ExifTags.TAGS}

    tiff_tags = img_pil.getexif()
    tiff_dict = {
        TiffTags.TAGS_V2[k].name: v
        for k, v in tiff_tags.items()
        if k in TiffTags.TAGS_V2
    }
    return {**exif_dict, **tiff_dict}

def fpx_from_f35(width: float, height: float, f_mm: float = 50) -> float:
    """Convert a focal length given in mm (35mm film equivalent) to pixels."""
    return f_mm * np.sqrt(width**2.0 + height**2.0) / np.sqrt(36**2 + 24**2)


def rgb2gray(img: Union[np.ndarray, torch.Tensor]):
    """img in shape [C, H, W] in rgb format"""
    img = img.squeeze()
    if isinstance(img, torch.Tensor):
        img = img.permute(1, 2, 0).numpy()
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def tensor_to_image(x: Union[Tensor, Image.Image, np.ndarray]) -> Image:
    if isinstance(x, Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(x, Image.Image):
        return x
    if x.max() <= 1.01:
        x *= 255
    if x.shape[0] == 3:
        x = x.transpose((1, 2, 0))
    x = x.astype(np.uint8)
    return Image.fromarray(x)


def resize_image(image, long_edge_size):
    S = max(image.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / S)) for x in image.size)
    return image.resize(new_size, interp)


def rescale_keypoints(kpts, original_size, new_size):
    kpts[:, 0] *= original_size[1] / new_size[1]
    kpts[:, 1] *= original_size[0] / new_size[0]
    return kpts.astype(np.uint32)

def read_image(path: Union[str, Path], size: int = None,
               auto_rotate: bool = True, remove_alpha: bool = True):
    """Read the image and also extract EXIF data if exists
    Args:
    ----
        path: The url to the image to load.
        auto_rotate: Rotate the image based on the EXIF data, default is True.
        remove_alpha: Remove the alpha channel, default is True.

    Returns:
    -------
        img: The image loaded as a numpy array.
        f_px: The optional focal length in pixels, extracting from the exif data.
    """
    path = Path(path)
    if path.suffix.lower() in [".heic"]:
        heif_file = pillow_heif.open_heif(path, convert_hdr_to_8bit=True)
        img_pil = heif_file.to_pillow()
    else:
        img_pil = Image.open(path)

    img_exif = extract_exif(img_pil)
    if auto_rotate:
        exif_orientation = img_exif.get("Orientation", 1)
        if exif_orientation == 3:
            img_pil = img_pil.transpose(Image.ROTATE_180)
        elif exif_orientation == 6:
            img_pil = img_pil.transpose(Image.ROTATE_270)
        elif exif_orientation == 8:
            img_pil = img_pil.transpose(Image.ROTATE_90)

    if size is not None:
        # W1, H1 = img_pil.size
        img_pil = resize_image(img_pil, size)
        W, H = img_pil.size
        cx, cy = W//2, H//2
        halfw = ((2 * cx) // 16) * 16 / 2
        halfh = ((2 * cy) // 16) * 16 / 2
        # if not (square_ok) and W == H:
        #     halfh = 3*halfw/4
        img_pil = img_pil.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))

    img = np.array(img_pil)
    # Convert to RGB if single channel.
    if img.ndim < 3 or img.shape[2] == 1:
        img = np.dstack((img, img, img))

    if remove_alpha:
        img = img[:, :, :3]

    # Extract the focal length from exif data.
    f_35mm = img_exif.get("FocalLengthIn35mmFilm",
        img_exif.get("FocalLenIn35mmFilm", img_exif.get("FocalLengthIn35mmFormat", None)),
    )
    if f_35mm is not None and f_35mm > 0:
        # print(f"\tfocal length @ 35mm film: {f_35mm}mm")
        f_px = fpx_from_f35(img.shape[1], img.shape[0], f_35mm)
    else:
        f_px = None
    return img, f_px


# def read_image_paths(folder: Union[str, Path]):
#     folder = Path(folder)
#     # image_paths = sorted([x for x in folder.iterdir() if x.suffix.lower() in img_extensions], key=lambda x: int(re.search(r'\d+', Path(x).stem).group()))
#     image_paths = sorted([x for x in folder.iterdir() if x.suffix.lower() in img_extensions])
#     return image_paths


def read_image_paths(root: str | Path) -> list[Path]:
    root = Path(root)
    root_image_paths = []
    for folder in root.iterdir():
        if folder.is_dir():
            image_paths = sorted([x for x in folder.iterdir() if not x.name.startswith('.') and x.suffix.lower() in img_extensions])
            root_image_paths.extend(image_paths)
        else:
            return sorted([x for x in root.iterdir() if not x.name.startswith('.') and x.suffix.lower() in img_extensions])
    return root_image_paths


def get_grouped_filenames(paths: list[Path], nested=False) -> list[str]:
    if not nested:
        return [f"{path.name}" for path in paths]
    # this assume that there are only one level of folder with files
    return [f"{path.parts[-2]}/{path.name}" for path in paths]

def read_images(path: Union[str, Path, list], *args, **kwargs):
    if isinstance(path, list):
        image_paths = path
        if isinstance(image_paths[0], np.ndarray):
            return image_paths
    else:
        path = Path(path)
        if path.is_dir():
            image_paths = read_image_paths(path)
        elif path.is_file():
            image_paths = [path]

    images, focals = [], []
    for image_path in image_paths:
        image, focal = read_image(image_path, *args, **kwargs)
        images.append(image)
        focals.append(focal)
    return images, focals


def depth2points(depth: np.ndarray, intrinsic: np.ndarray = None, focal: float = None, cu: float = None, cv: float = None, mask=None):
    H, W = depth.shape

    if intrinsic is None:
        assert focal is not None
        if cu is None:
            cu = (W - 1) * 0.5
        if cv is None:
            cv = (H - 1) * 0.5
    else:
        focal, cu, cv = intrinsic[0, 0], intrinsic[0, 2], intrinsic[1, 2]

    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    points = np.zeros((H, W, 3), dtype=np.float32)
    if mask is None:
        mask = np.ones((H, W), dtype=bool)

    points[mask, 0] = depth[mask] * (uu[mask] - cu) / focal
    points[mask, 1] = depth[mask] * (vv[mask] - cv) / focal
    points[mask, 2] = depth[mask]
    return points


def yaml_join(loader, node):
    seq = loader.construct_sequence(node)
    return ''.join([str(i) for i in seq])



# def read_config(path: str | Path) -> Box:
#     yaml.add_constructor('!join', yaml_join)
#     with open(path) as f:
#         data = yaml.full_load(f)
#         config = Box(data)
#     return config


def load_h5(file_path, transform_slash=True):
    with h5py.File(file_path, 'r') as f:
        data = {k if not transform_slash else k.replace('+', '/'): v.__array__() \
                    for k, v in f.items()}
    return data

# if __name__ == '__main__':
#     config_file = "./configs/config.yaml"
#     config = read_config(config_file)
#     print(config.database.image_path.__repr__)
