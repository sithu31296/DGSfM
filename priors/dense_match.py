import h5py
import argparse
from tqdm import tqdm
from dgsfm.database.config import DGSfMConfig, load_dgsfm_config
from dgsfm.utils.io import read_image
from dgsfm.models.roma import RoMav1, RoMav2
from dgsfm.database.dataset import COLMAPDataset


class DenseMatching:
    def __init__(self, cfg: DGSfMConfig) -> None:
        self.model = eval(cfg.viewgraph.match_model)(device=cfg.device)

    def pairs_matching(self, dataset: COLMAPDataset) -> None:
        for scene in dataset:
            print(scene.scene_name)
            scene.create_pairs()

            with h5py.File(scene.dense_match_path, 'w') as f:
                for pair_idx in tqdm(range(len(scene.image_pairs))):
                    image_file1, image_file2 = scene.image_pairs[pair_idx]
                    image1, image2 = read_image(scene.image_dir / image_file1)[0], read_image(scene.image_dir / image_file2)[0]
                    dense_match, certainty = self.model(image1, image2, save_path=None)
                    grp = f.create_group(f"{image_file1}_{image_file2}")
                    grp.create_dataset("dense_match", data=dense_match, compression="gzip", compression_opts=4)
                    grp.create_dataset("certainty", data=certainty, compression="gzip", compression_opts=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="Path to the configuration file", default="configs/eth3d.yaml")
    args = parser.parse_args()
    """
    WARNING: RoMa dense matching warp size is not the same as the image size.
    """
    cfg = load_dgsfm_config(args.config)
    matching = DenseMatching(cfg)

    dataset = COLMAPDataset(cfg)
    matching.pairs_matching(dataset)
