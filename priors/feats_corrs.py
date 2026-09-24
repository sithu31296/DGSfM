import torch
import h5py
import argparse
import numpy as np
from tqdm import tqdm
from dgsfm.database.config import DGSfMConfig, load_dgsfm_config
from dgsfm.database.dataset import COLMAPDataset, Scene
from dgsfm.models.lomatch import LoMa
from dgsfm.models.roma import sample_feats, sample_feats_v2, to_pixel_coordinates, match_keypoints
from dgsfm.utils.io import read_image


class CorrespondenceGenerator:
    def __init__(self, cfg: DGSfMConfig) -> None:
        self.cfg = cfg
        self.features_detectors = ['SIFT', 'LoMaB', 'LoMaG', 'LoMaL']
        self.dense_matchers = ['RoMav1', 'RoMav2', 'UFM']
        self.sparse_matchers = ['NNMatcher', 'LoMaB', 'LoMaG', 'LoMaL']

        self.feats_model = None
        if "LoMa" in cfg.viewgraph.feats_model:
            if "L" in cfg.viewgraph.feats_model:
                self.feats_model = LoMa("L", cfg.viewgraph.num_feats)
            elif "G" in cfg.viewgraph.feats_model:
                self.feats_model = LoMa("G", cfg.viewgraph.num_feats)
            else:
                self.feats_model = LoMa("B", cfg.viewgraph.num_feats)


    def to_tensor(self, x: np.ndarray, dtype: str) -> torch.Tensor:
        return torch.from_numpy(x).to(dtype=dtype, device=self.cfg.device)

    def to_numpy(self, x: torch.Tensor, dtype: str) -> np.ndarray:
        return x.detach().cpu().numpy().astype(dtype)

    def generate_correspondences(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        """
        Generate correspondences between two images using the specified matching model.
        """
        dense_match, certainty = self.model(image1, image2, save_path=None)
        return dense_match

    def filter_bounds(self, feats1: np.ndarray, feats2: np.ndarray, scores: np.ndarray, h1: int, w1: int, h2: int, w2: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Filter features that are out of image bounds.
        """
        mask = (feats1[:, 0] > 0) & (feats1[:, 0] < w1-1) & (feats1[:, 1] > 0) & (feats1[:, 1] < h1-1) & \
               (feats2[:, 0] > 0) & (feats2[:, 0] < w2-1) & (feats2[:, 1] > 0) & (feats2[:, 1] < h2-1)
        return feats1[mask], feats2[mask], scores[mask]

    def sample_features_from_dense_matches(self, dense_match: np.ndarray, certainty: np.ndarray, H1: int, W1: int, H2: int, W2: int) -> tuple[np.ndarray, np.ndarray]:
        # Implementation for sampling features from dense matches
        if self.cfg.viewgraph.match_model == "RoMav1":
            matches, scores = sample_feats(dense_match, certainty, self.cfg.viewgraph.num_feats, self.cfg.viewgraph.feats_confidence)
        elif self.cfg.viewgraph.match_model == "RoMav2":
            matches, scores = sample_feats_v2(dense_match, certainty, self.cfg.viewgraph.num_feats, self.cfg.viewgraph.feats_confidence)
        else:
            raise ValueError(f"Unknown matching model: {self.cfg.viewgraph.match_model}")
        feats1, feats2 = to_pixel_coordinates(matches, H1, W1, H2, W2)
        feats1, feats2, scores = self.filter_bounds(feats1, feats2, scores, H1, W1, H2, W2)
        return feats1, feats2, scores

    def get_dense_feature_indices(self, 
                                  image_file: str, 
                                  new_feats: np.ndarray, new_scores: np.ndarray, 
                                  image_features: dict[str, torch.Tensor], image_features_scores: dict[str, torch.Tensor]) -> np.ndarray:
        # Implementation for getting dense feature indices
        existing_feats = image_features[image_file]
        num_existing_feats = existing_feats.shape[0]
        match_indices = torch.arange(new_feats.shape[0], device=self.cfg.device)
        if num_existing_feats == 0:
            image_features[image_file] = torch.vstack([image_features[image_file], new_feats])
            image_features_scores[image_file] = torch.vstack([image_features_scores[image_file], new_scores])
        else:
            # compute pairwise L2 distances (num_cur, num_existing)
            dist_matrix = torch.cdist(new_feats, existing_feats)
            # find the closest existing keypoint for each new keypoint
            min_dist_idx = torch.argmin(dist_matrix, dim=1)
            min_dists = dist_matrix[match_indices, min_dist_idx]
            is_duplicate = min_dists <= 1
    
            # extract truly unique kpts
            new_non_dup_feats = new_feats[~is_duplicate]
            # prepare indices: if duplicate, use existing idx; else, assign new ones
            match_indices = torch.where(is_duplicate, min_dist_idx, -1)
            # assign new indices to non-duplicates
            non_dup_indices = torch.arange(num_existing_feats, num_existing_feats + len(new_non_dup_feats), device=self.cfg.device)
            match_indices[match_indices == -1] = non_dup_indices
            if len(new_non_dup_feats) > 0:
                image_features[image_file] = torch.vstack([image_features[image_file], new_non_dup_feats])
                image_features_scores[image_file] = torch.vstack([image_features_scores[image_file], new_scores[~is_duplicate]])
        return self.to_numpy(match_indices, self.cfg.int_dtype)

    def generate_sparse_features_and_matches(self, scene: Scene) -> None:
        image_features, image_descriptors = {}, {}
        with h5py.File(scene.image_feats_path, 'w') as file:
            for img_id in tqdm(range(scene.num_images)):
                img, img_path = scene.image_filenames[img_id], scene.image_paths[img_id]
                feats, descriptors, h, w = self.feats_model.extract(img_path)
                feats_px = self.feats_model.to_pixel_coords(feats, h, w).astype(self.cfg.int_dtype)
                scores = np.ones((len(feats_px), 1), dtype=np.float16)

                image_features[img] = feats
                image_descriptors[img] = descriptors

                grp = file.create_group(img)
                grp.create_dataset("feats", data=feats_px, compression="gzip", compression_opts=4)
                grp.create_dataset("scores", data=scores, compression="gzip", compression_opts=4)

        if self.cfg.viewgraph.match_model in self.sparse_matchers:
            self.generate_matches_from_descriptors(scene, image_features, image_descriptors)
        else:
            self.generate_matches_from_warping(scene, image_features)

    def generate_matches_from_warping(self, scene: Scene, features: dict[str, np.ndarray]):
        with (h5py.File(scene.dense_match_path, 'r') as file,
            h5py.File(scene.feats_match_path, 'w') as save_file):
            for pair_idx in tqdm(range(len(scene.image_pairs))):
                image_file1, image_file2 = scene.image_pairs[pair_idx]
                pair_key = f"{image_file1}_{image_file2}"
                dense_match = self.to_tensor(file[pair_key]["dense_match"][()], torch.float32)
                dense_match_certainty = self.to_tensor(file[pair_key]["certainty"][()], torch.float16)
                image1, image2 = read_image(scene.image_dir / image_file1)[0], read_image(scene.image_dir / image_file2)[0]
                feats1, feats2 = features[image_file1], features[image_file2]
                image1_ids, image2_ids = match_keypoints(feats1, feats2, 
                                                         dense_match, dense_match_certainty, 
                                                         max_dist=0.005, cert_th=0)
                image1_ids, image2_ids = self.to_numpy(image1_ids, self.cfg.int_dtype), self.to_numpy(image2_ids, self.cfg.int_dtype)
                matches = np.stack([image1_ids, image2_ids], axis=-1)
                grp = save_file.create_group(pair_key)
                grp.create_dataset("matches", data=matches, compression="gzip", compression_opts=4)
                
    def generate_matches_from_descriptors(self, scene: Scene, features, descriptors):
        with h5py.File(scene.feats_match_path, 'w') as save_file:
            for pair_idx in tqdm(range(len(scene.image_pairs))):
                image_file1, image_file2 = scene.image_pairs[pair_idx]
                matches = self.feats_model.match_from_feats(
                    features[image_file1], features[image_file2], 
                    descriptors[image_file1], descriptors[image_file2]
                )
                grp = save_file.create_group(f"{image_file1}_{image_file2}")
                grp.create_dataset("matches", data=matches, compression="gzip", compression_opts=4)
        
        
    def generate_dense_features_and_matches(self, scene: Scene) -> None:
        image_features, image_features_scores = {}, {}
        for image_file in scene.image_filenames:
            zero_feats = torch.zeros((0, 2), dtype=torch.float32, device=self.cfg.device)
            zero_scores = torch.zeros((0, 1), dtype=torch.float16, device=self.cfg.device)
            image_features[image_file] = zero_feats
            image_features_scores[image_file] = zero_scores

        with (h5py.File(scene.dense_match_path, 'r') as file,
                h5py.File(scene.feats_match_path, 'w') as save_file):
            for pair_idx in tqdm(range(len(scene.image_pairs))):
                image_file1, image_file2 = scene.image_pairs[pair_idx]
                pair_key = f"{image_file1}_{image_file2}"
                dense_match = self.to_tensor(file[pair_key]["dense_match"][()], torch.float32)
                dense_match_certainty = self.to_tensor(file[pair_key]["certainty"][()], torch.float16)
                image1, image2 = read_image(scene.image_dir / image_file1)[0], read_image(scene.image_dir / image_file2)[0]
                feats1, feats2, scores = self.sample_features_from_dense_matches(dense_match, dense_match_certainty, image1.shape[0], image1.shape[1], image2.shape[0], image2.shape[1])

                image1_ids = self.get_dense_feature_indices(image_file1, feats1, scores, image_features, image_features_scores)
                image2_ids = self.get_dense_feature_indices(image_file2, feats2, scores, image_features, image_features_scores)
                matches = np.stack([image1_ids, image2_ids], axis=-1)

                grp = save_file.create_group(pair_key)
                grp.create_dataset("matches", data=matches, compression="gzip", compression_opts=4)

        with h5py.File(scene.image_feats_path, 'w') as file:
            for img in image_features.keys():
                feats = self.to_numpy(image_features[img], self.cfg.int_dtype)
                scores = self.to_numpy(image_features_scores[img], np.float16)
                grp = file.create_group(img)
                grp.create_dataset("feats", data=feats, compression="gzip", compression_opts=4)
                grp.create_dataset("scores", data=scores, compression="gzip", compression_opts=4)

    def generate(self, dataset: COLMAPDataset) -> None:
        print(f"Generating correspondences for dataset with {self.cfg.viewgraph.feats_model}-{self.cfg.viewgraph.match_model}...")
        for scene in dataset:
            print(scene.scene_name)
            if scene.scene_name not in ['nyhavn', 'pagoda_river']:
                continue
            if self.cfg.viewgraph.feats_model in self.features_detectors:
                self.generate_sparse_features_and_matches(scene)
            else:
                self.generate_dense_features_and_matches(scene)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="Path to the configuration file", default="configs/eth3d.yaml")
    args = parser.parse_args()
    cfg = load_dgsfm_config(args.config)
    correspondence_generator = CorrespondenceGenerator(cfg)

    dataset = COLMAPDataset(cfg)
    correspondence_generator.generate(dataset)
