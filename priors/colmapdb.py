import h5py
import os
import argparse
import pycolmap
import numpy as np
from tqdm import tqdm
from dgsfm.database.config import DGSfMConfig, load_dgsfm_config
from dgsfm.database.dataset import COLMAPDataset, Scene


def create_colmap_database(scene: Scene):
    if scene.colmapdb_path.exists():
        os.remove(scene.colmapdb_path)
    colmap_db = pycolmap.Database.open(scene.colmapdb_path)
    colmap_db.clear_all_tables()

    # sensor = pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA)
    with h5py.File(scene.image_feats_path, 'r') as file:
        for i in tqdm(range(scene.num_images)):
            # colmap uses 1-based indices
            image_id = cam_id = i + 1
            image_name = scene.image_filenames[i]

            cam = pycolmap.infer_camera_from_image(scene.image_paths[i])
            sensor = pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=cam_id)
            rig = pycolmap.Rig()
            image = pycolmap.Image()
            frame = pycolmap.Frame()

            cam.camera_id = cam_id
            image.name = image_name
            image.image_id = image_id
            image.camera_id = cam_id
            frame.frame_id = image_id
            frame.rig_id = cam_id
            rig.rig_id = cam_id
            rig.add_ref_sensor(sensor)

            image_data = pycolmap.data_t()
            sensor.id = cam_id
            image_data.sensor_id = sensor
            image_data.id = image_id
            frame.add_data_id(image_data)

            colmap_db.write_camera(cam, use_camera_id=True)
            colmap_db.write_rig(rig, use_rig_id=True)
            colmap_db.write_image(image, use_image_id=True)
            colmap_db.write_frame(frame, use_frame_id=True)

            image2 = colmap_db.read_image(image_id=image_id)
            image2.frame_id = image_id
            colmap_db.update_image(image2)

            feats = file[image_name]["feats"][()]
            scores = file[image_name]["scores"][()]
            keypoints = np.concatenate([feats, scores], axis=-1)
            colmap_db.write_keypoints(image_id=image_id, keypoints=keypoints)

    with h5py.File(scene.feats_match_path, 'r') as file:
        for i in tqdm(range(len(scene.image_pairs))):
            image_name1, image_name2 = scene.image_pairs[i]
            pair_key = f"{image_name1}_{image_name2}"
            match_indices = file[pair_key]["matches"][()]
            image_id1, image_id2 = scene.image_filenames_to_ids[image_name1], scene.image_filenames_to_ids[image_name2]
            colmap_db.write_matches(image_id1=image_id1+1, image_id2=image_id2+1, matches=match_indices)
    colmap_db.close()



def run(cfg: DGSfMConfig):
    dataset = COLMAPDataset(cfg)
    for scene in dataset:
        print(scene.scene_name)
        create_colmap_database(scene)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="Path to the configuration file", default="configs/imc2021.yaml")
    args = parser.parse_args()
    cfg = load_dgsfm_config(args.config)
    run(cfg)