import h5py
import argparse
from tqdm import tqdm
from dgsfm.utils.io import read_image
from dgsfm.models.depth_estimators import UniDepth2, MoGe1, MoGe2, MoGe3, ZoeDepth
from dgsfm.database.config import DGSfMConfig, load_dgsfm_config
from dgsfm.database.dataset import COLMAPDataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="Path to the configuration file", default="configs/eth3d.yaml")
    args = parser.parse_args()
    cfg = load_dgsfm_config(args.config)
    dataset = COLMAPDataset(cfg)
    model = eval(cfg.viewgraph.depth_model)()
    print(f"Generating monocular depth estimation with {cfg.viewgraph.depth_model}..")

    for scene in dataset:
        print(scene.scene_name)
        with h5py.File(scene.monodepth_path, 'w') as save_file:
            for i in tqdm(range(len(scene.image_paths))):
                image, focal = read_image(scene.image_paths[i])
                depth, conf, intrinsic = model(image, focal)
                grp = save_file.create_group(scene.image_filenames[i])
                grp.create_dataset("depth", data=depth, compression="gzip", compression_opts=4)
                grp.create_dataset("conf", data=conf, compression="gzip", compression_opts=4)
                grp.create_dataset("intrinsic", data=intrinsic, compression="gzip", compression_opts=4)
