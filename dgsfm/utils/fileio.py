import numpy as np
from pathlib import Path
from dgsfm.utils.recon_utils import export_reconstruction
from dgsfm.database.structures import FocalSource, ImageBatch, CameraBatch


def load_depthmap(dir: Path, image: str):
    data = np.load(dir / f"{Path(image).stem}.npz")
    return data['depth'], data['intrinsic']

def load_dense_matches(match_dir, image1, image2):
    data = np.load(match_dir / f"{image1}_{image2}.npz")
    return data['dense_match'], data['certainty']

def load_features(feats_dir: Path, image: str):
    data = np.load(feats_dir / f"{Path(image).stem}.npz")
    return data['feats'], data['scores']

def load_correspondences(match_dir: Path, image1: str, image2: str):
    data = np.load(match_dir / f"{image1}_{image2}.npz")
    return data['indices']

def load_relpose(feats_dir: Path, image1: str, image2: str):
    data = np.load(feats_dir / f"{image1}_{image2}_pose.npz")
    return data['scale'], data['rotation'], data['translation'], data['intrinsic1'], data['intrinsic2'], data['inliers'], data['inliers_mask'], data['inliers_ratio']


def save_colmap_model(output_path, image_path, cameras, images, tracks):
    print("Saving the reconstruction as a COLMAP model...")
    export_reconstruction(output_path, cameras, images, tracks, image_path, export_txt=True)

def load_image_pairs(filepath):
    lines = Path(filepath).read_text().splitlines()
    image_pairs = []
    for line in lines:
        image1, image2 = line.split("<>")
        image_pairs.append((image1, image2))
    return image_pairs

def load_image_pairs_with_scores(filepath):
    lines = Path(filepath).read_text().splitlines()
    image_pairs, scores = [], []
    for line in lines:
        image1, image2, score = line.split("<>")
        image_pairs.append((image1, image2))
        scores.append(float(score))
    return image_pairs, scores

def save_image_pairs(image_pairs: list[tuple[str, str]], save_path: Path) -> None:
    pairs = [f"{img1}<>{img2}\n" for img1, img2 in image_pairs]
    with open(save_path, "w") as f:
        f.writelines(pairs)

def save_image_pairs_with_scores(image_pairs: list[tuple[str, str]], scores: list[float], save_path: Path) -> None:
    pairs = [f"{img1}<>{img2} {score:.2f}\n" for (img1, img2), score in zip(image_pairs, scores)]
    with open(save_path, "w") as f:
        f.writelines(pairs)

def get_feats_depths(depthmap: np.ndarray, feats: np.ndarray) -> np.ndarray:
    assert feats.shape[-1] == 2
    try:
        depths = depthmap[feats[:, 1], feats[:, 0]]
    except IndexError:
        print("Error: Feature coordinates are out of bounds for the depth map.")
        depths = None
    return depths

def sample_correspondences(feats1: np.ndarray, feats2: np.ndarray, matches: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return feats1[matches[:, 0]], feats2[matches[:, 1]]

def load_image_focals(path: Path):
    lines = path.read_text().splitlines()
    focals = {}
    for line in lines[1:]:
        fname, known, mde, solver, graph, final = line.split(" ")
        focals[fname] = FocalSource(known, mde, solver, graph, final)
    return focals

def save_image_focals(path: Path, cameras: CameraBatch, images: ImageBatch):
    focals = []
    for image in images:
        cam = cameras[image.cam_id]
        focals.append(f"{image.filename} {cam.focal_known:.2f} {cam.focal_mde:.2f} {cam.focal_solver:.2f} {cam.focal_graph:.2f} {cam.focal_final:.2f}\n")
    with open(path, "w") as f:
        f.writelines(["filename EXIF MDE SOLVER GRAPH FINAL\n"]+focals)
