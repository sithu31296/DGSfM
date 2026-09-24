# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# colmap mapper/colmap point_triangulator/glomap mapper from mast3r matches
# --------------------------------------------------------
import pycolmap
import os
import os.path as path
import kapture.io
import kapture.io.csv
import subprocess
import PIL
from tqdm import tqdm
import PIL.Image
import numpy as np
from typing import List, Tuple, Union

import torch
import copy
from scipy.special import softmax

from mast3r.model import AsymmetricMASt3R
from mast3r.colmap.database import export_matches, get_im_matches

import mast3r.utils.path_to_dust3r  # noqa
from dust3r_visloc.datasets.utils import get_resize_function

import kapture
from kapture.converter.colmap.database_extra import get_colmap_camera_ids_from_db, get_colmap_image_ids_from_db
from kapture.utils.paths import path_secure

from dust3r.datasets.utils.transforms import ImgNorm
from mast3r.inference import inference


def scene_prepare_images(root: str, maxdim: int, patch_size: int, image_paths: List[str]):
    images = []
    # image loading
    for idx in tqdm(range(len(image_paths))):
        rgb_image = PIL.Image.open(os.path.join(root, image_paths[idx])).convert('RGB')

        # resize images
        W, H = rgb_image.size
        resize_func, _, to_orig = get_resize_function(maxdim, patch_size, H, W)
        rgb_tensor = resize_func(ImgNorm(rgb_image))

        # image dictionary
        images.append({'img': rgb_tensor.unsqueeze(0),
                       'true_shape': np.int32([rgb_tensor.shape[1:]]),
                       'to_orig': to_orig,
                       'idx': idx,
                       'instance': image_paths[idx],
                       'orig_shape': np.int32([H, W])})
    return images


def remove_duplicates(images, image_pairs):
    pairs_added = set()
    pairs = []
    for (i, _), (j, _) in image_pairs:
        smallidx, bigidx = min(i, j), max(i, j)
        if (smallidx, bigidx) in pairs_added:
            continue
        pairs_added.add((smallidx, bigidx))
        pairs.append((images[i], images[j]))
    return pairs


def run_mast3r_matching(model: AsymmetricMASt3R, maxdim: int, patch_size: int, device,
                        kdata: kapture.Kapture, root_path: str, image_pairs_kapture: List[Tuple[str, str]],
                        colmap_db,
                        dense_matching: bool, pixel_tol: int, conf_thr: float, skip_geometric_verification: bool,
                        min_len_track: int):
    assert kdata.records_camera is not None
    image_paths = kdata.records_camera.data_list()
    image_path_to_idx = {image_path: idx for idx, image_path in enumerate(image_paths)}
    image_path_to_ts = {kdata.records_camera[ts, camid]: (ts, camid) for ts, camid in kdata.records_camera.key_pairs()}

    images = scene_prepare_images(root_path, maxdim, patch_size, image_paths)
    image_pairs = [((image_path_to_idx[image_path1], image_path1), (image_path_to_idx[image_path2], image_path2))
                   for image_path1, image_path2 in image_pairs_kapture]
    matching_pairs = remove_duplicates(images, image_pairs)

    colmap_camera_ids = get_colmap_camera_ids_from_db(colmap_db, kdata.records_camera)
    colmap_image_ids = get_colmap_image_ids_from_db(colmap_db)
    im_keypoints = {idx: {} for idx in range(len(image_paths))}

    im_matches = {}
    image_to_colmap = {}
    for image_path, idx in image_path_to_idx.items():
        _, camid = image_path_to_ts[image_path]
        colmap_camid = colmap_camera_ids[camid]
        colmap_imid = colmap_image_ids[image_path]
        image_to_colmap[idx] = {
            'colmap_imid': colmap_imid,
            'colmap_camid': colmap_camid
        }

    # compute 2D-2D matching from dust3r inference
    for chunk in tqdm(range(0, len(matching_pairs), 4)):
        pairs_chunk = matching_pairs[chunk:chunk + 4]
        output = inference(pairs_chunk, model, device, batch_size=1, verbose=False)
        pred1, pred2 = output['pred1'], output['pred2']
        # TODO handle caching
        im_images_chunk = get_im_matches(pred1, pred2, pairs_chunk, image_to_colmap,
                                         im_keypoints, conf_thr, not dense_matching, pixel_tol)
        im_matches.update(im_images_chunk.items())

    # filter matches, convert them and export keypoints and matches to colmap db
    colmap_image_pairs = export_matches(
        colmap_db, images, image_to_colmap, im_keypoints, im_matches, min_len_track, skip_geometric_verification)
    colmap_db.commit()

    return colmap_image_pairs


def pycolmap_run_triangulator(colmap_db_path, prior_recon_path, recon_path, image_root_path):
    print("running mapping")
    reconstruction = pycolmap.Reconstruction(prior_recon_path)
    pycolmap.triangulate_points(
        reconstruction=reconstruction,
        database_path=colmap_db_path,
        image_path=image_root_path,
        output_path=recon_path,
        refine_intrinsics=False,
    )


def pycolmap_run_mapper(colmap_db_path, recon_path, image_root_path):
    print("running mapping")
    reconstructions = pycolmap.incremental_mapping(
        database_path=colmap_db_path,
        image_path=image_root_path,
        output_path=recon_path,
        options=pycolmap.IncrementalPipelineOptions({'multiple_models': False,
                                                     'extract_colors': True,
                                                     })
    )


def glomap_run_mapper(glomap_bin, colmap_db_path, recon_path, image_root_path):
    print("running mapping")
    args = [
        'mapper',
        '--database_path',
        colmap_db_path,
        '--image_path',
        image_root_path,
        '--output_path',
        recon_path
    ]
    args.insert(0, glomap_bin)
    glomap_process = subprocess.Popen(args)
    glomap_process.wait()

    if glomap_process.returncode != 0:
        raise ValueError(
            '\nSubprocess Error (Return code:'
            f' {glomap_process.returncode} )')


def kapture_import_image_folder_or_list(images_path: Union[str, Tuple[str, List[str]]], use_single_camera=False) -> kapture.Kapture:
    images = kapture.RecordsCamera()

    if isinstance(images_path, str):
        images_root = images_path
        file_list = [path.relpath(path.join(dirpath, filename), images_root)
                     for dirpath, dirs, filenames in os.walk(images_root)
                     for filename in filenames]
        file_list = sorted(file_list)
    else:
        images_root, file_list = images_path

    sensors = kapture.Sensors()
    for n, filename in enumerate(file_list):
        # test if file is a valid image
        try:
            # lazy load
            with PIL.Image.open(path.join(images_root, filename)) as im:
                width, height = im.size
                model_params = [width, height]
        except (OSError, PIL.UnidentifiedImageError):
            # It is not a valid image: skip it
            print(f'Skipping invalid image file {filename}')
            continue

        camera_id = f'sensor'
        if use_single_camera and camera_id not in sensors:
            sensors[camera_id] = kapture.Camera(kapture.CameraType.UNKNOWN_CAMERA, model_params)
        elif use_single_camera:
            assert sensors[camera_id].camera_params[0] == width and sensors[camera_id].camera_params[1] == height
        else:
            camera_id = camera_id + f'{n}'
            sensors[camera_id] = kapture.Camera(kapture.CameraType.UNKNOWN_CAMERA, model_params)

        images[(n, camera_id)] = path_secure(filename)  # don't forget windows

    return kapture.Kapture(sensors=sensors, records_camera=images)


def convert_to_numpy_array(preds):
    if isinstance(preds, torch.Tensor):
        # Directly convert tensor to numpy array
        return preds.detach().cpu().numpy()
    elif isinstance(preds, list):
        # Convert list of tensors to a single numpy array
        return np.stack([p.detach().cpu().numpy() for p in preds])
    else:
        raise ValueError("Input must be a torch.Tensor or a list of torch.Tensors.")
                

def run_mast3r_matching_doppelgangers(model: AsymmetricMASt3R, maxdim: int, patch_size: int, device,
                        kdata: kapture.Kapture, root_path: str, image_pairs_kapture: List[Tuple[str, str]],
                        colmap_db,
                        dense_matching: bool, pixel_tol: int, conf_thr: float, skip_geometric_verification: bool,
                        min_len_track: int, dopp_pred_path: str, dopp_thresh: float):
    assert kdata.records_camera is not None
    image_paths = kdata.records_camera.data_list()
    image_path_to_idx = {image_path: idx for idx, image_path in enumerate(image_paths)}
    image_path_to_ts = {kdata.records_camera[ts, camid]: (ts, camid) for ts, camid in kdata.records_camera.key_pairs()}

    images = scene_prepare_images(root_path, maxdim, patch_size, image_paths)
    image_pairs = [((image_path_to_idx[image_path1], image_path1), (image_path_to_idx[image_path2], image_path2))
                   for image_path1, image_path2 in image_pairs_kapture]
    matching_pairs = remove_duplicates(images, image_pairs)

    colmap_camera_ids = get_colmap_camera_ids_from_db(colmap_db, kdata.records_camera)
    colmap_image_ids = get_colmap_image_ids_from_db(colmap_db)
    im_keypoints = {idx: {} for idx in range(len(image_paths))}

    im_matches = {}
    image_to_colmap = {}
    for image_path, idx in image_path_to_idx.items():
        _, camid = image_path_to_ts[image_path]
        colmap_camid = colmap_camera_ids[camid]
        colmap_imid = colmap_image_ids[image_path]
        image_to_colmap[idx] = {
            'colmap_imid': colmap_imid,
            'colmap_camid': colmap_camid,
            'image_path': image_path
        }
    doppelgangers_propability_list = {}
    # compute 2D-2D matching from dust3r inference
    for chunk in tqdm(range(0, len(matching_pairs), 4)):
        pairs_chunk = matching_pairs[chunk:chunk + 4]
        output = inference(pairs_chunk, model, device, batch_size=1, verbose=False) 
            
        pred1, pred2 = output['pt_pred1'], output['pt_pred2']
        dopp_pred1, dopp_pred2 = output['pred1'], output['pred2']

        dopp_pred1_np = convert_to_numpy_array(dopp_pred1)
        dopp_pred2_np = convert_to_numpy_array(dopp_pred2)

        # Compute softmax scores
        score_s1 = softmax(dopp_pred1_np, axis=1)
        score_s2 = softmax(dopp_pred2_np, axis=1)

        pred = np.zeros(score_s1.shape[0])
        
        vote_0 = (score_s1[:, 0] > score_s1[:, 1]).astype(int) + (score_s2[:, 0] > score_s2[:, 1]).astype(int)
        vote_1 = (score_s1[:, 1] > score_s1[:, 0]).astype(int) + (score_s2[:, 1] > score_s2[:, 0]).astype(int)

        # Compute predictions
        pred = np.where(
            vote_1 > vote_0,
            np.maximum(score_s1[:, 1], score_s2[:, 1]),
            np.where(
                vote_1 < vote_0,
                np.minimum(score_s1[:, 1], score_s2[:, 1]),
                (score_s1[:, 1] + score_s2[:, 1]) / 2
            )
        )
        
        dopp_mask = (pred > dopp_thresh)
            
        im_images_chunk = get_im_matches(pred1, pred2, pairs_chunk, image_to_colmap,
                                         im_keypoints, conf_thr, not dense_matching, pixel_tol, dopp_mask=dopp_mask)
        im_matches.update(im_images_chunk.items())
        
        for pair_index, pair in enumerate(pairs_chunk):
            imidx0 = pair[0]['idx']
            imidx1 = pair[1]['idx']
            imid0 = copy.deepcopy(image_to_colmap[imidx0]['colmap_imid'])
            imid1 = copy.deepcopy(image_to_colmap[imidx1]['colmap_imid'])
            if imid0 > imid1:
                imid0, imid1 = imid1, imid0
                imidx0, imidx1 = imidx1, imidx0
            imid0, imid1 = image_to_colmap[imidx0]['colmap_imid'], image_to_colmap[imidx1]['colmap_imid']
            doppelgangers_propability_list[(imid0, imid1)] = {
                'pred': pred[pair_index],
                'score_s1': score_s1[pair_index],
                'score_s2': score_s2[pair_index],
                'dopp_pred1_np': dopp_pred1_np[pair_index],
                'dopp_pred2_np': dopp_pred2_np[pair_index],
                'image_path1': image_to_colmap[imidx0]['image_path'],
                'image_path2': image_to_colmap[imidx1]['image_path']
            }
               
    np.save(dopp_pred_path, doppelgangers_propability_list)
    colmap_image_pairs = export_matches(
        colmap_db, images, image_to_colmap, im_keypoints, im_matches, min_len_track, skip_geometric_verification)
    colmap_db.commit()
    

    return colmap_image_pairs