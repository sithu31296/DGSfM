import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="torch.library",
)
import argparse
import h5py
import pycolmap
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from dgsfm.database.config import DGSfMConfig, load_dgsfm_config
from dgsfm.database.dataset import COLMAPDataset, Scene
from dgsfm.database.structures import ImageBatch, CameraBatch, TrackBatch, FocalSource
from dgsfm.database.viewgraph import ViewGraph, ImagePair
from dgsfm.database.tracks import TrackManager
from dgsfm.solvers.reposed import RePoseDEstimator
from dgsfm.utils.metrics import PoseMetrics
from dgsfm.utils.fileio import *
from dgsfm.utils.io import *
from dgsfm.utils.geometry import lift_and_transform, is_valid_focal, focal2intrinsic
from dgsfm.optimizers.rotation_averaging import RotationAveraging
from dgsfm.optimizers.bundle_adjustment import BundleAdjusterBAE
from dgsfm.optimizers.global_positioning import GlobalPositioning
from dgsfm.optimizers.scale_averaging import ScaleAveraging
from evaluate import eval_dataset



def read_colmap_database(cfg: DGSfMConfig, scene: Scene):
    print(f"Reading COLMAP database from {scene.colmapdb_path.name}...")
    colmap_db = pycolmap.Database.open(scene.colmapdb_path)

    num_images, num_cameras = colmap_db.num_images(), colmap_db.num_cameras()
    gimages, gcameras, viewgraph = ImageBatch(num_images), CameraBatch(num_cameras), ViewGraph()
    colmap_cameras, colmap_images = colmap_db.read_all_cameras(), colmap_db.read_all_images()
    cam_mapping = {}
    with tqdm(total=num_cameras, desc="Reading Cameras") as pbar:
        for cam_id, colmap_cam in enumerate(colmap_cameras):
            cam_mapping[colmap_cam.camera_id] = cam_id
            # gcameras.focal_estimates[cam_id][FocalSource.KNOWN] = colmap_cam.focal_length
            gcameras.principal_points[cam_id] = np.array([colmap_cam.principal_point_x, colmap_cam.principal_point_y])
            pbar.update(1)

    image_mapping = {}
    with (h5py.File(scene.monodepth_path, 'r') as file,
        tqdm(total=num_images, desc="Reading Images") as pbar):
        for image_id, colmap_image in enumerate(colmap_images):
            colmap_image_id, colmap_cam_id = colmap_image.image_id, colmap_image.camera_id
            cam_id = cam_mapping[colmap_cam_id]
            image_mapping[colmap_image_id] = image_id
            image_name = Path(colmap_image.name).name

            depthmap = file[image_name]["depth"][()]
            depthmap_conf = file[image_name]["conf"][()]
            intrinisic_mde = file[image_name]["intrinsic"][()]

            colmap_cam = colmap_db.read_camera(colmap_cam_id)
            width, height = colmap_cam.width, colmap_cam.height

            keypoints = colmap_db.read_keypoints(colmap_image_id)
            feats, feats_conf = keypoints[:, :2].astype(cfg.int_dtype), keypoints[:, 2].astype(cfg.float_dtype)
            depths, depths_conf = get_feats_depths(depthmap, feats).astype(cfg.float_dtype), get_feats_depths(depthmap_conf, feats).astype(cfg.float_dtype)

            gcameras.focal_estimates[cam_id][FocalSource.MDE] = intrinisic_mde[0, 0] if intrinisic_mde.ndim == 2 else intrinisic_mde[0]
            gimages.filenames[image_id] = image_name
            gimages.cam_ids[image_id] = cam_id
            gimages.image_sizes[image_id] = (height, width)
            gimages.features[image_id] = feats
            gimages.features_confs[image_id] = feats_conf
            gimages.depths[image_id] = depths
            gimages.depths_confs[image_id] = depths_conf
            pbar.update(1)
            
    colmap_pair_ids, colmap_matches = colmap_db.read_all_matches()
    with tqdm(total=len(colmap_matches), desc="Reading Image Pairs") as pbar:
        for colmap_pair_id, matches in zip(colmap_pair_ids, colmap_matches):
            colmap_image_id1, colmap_image_id2 = pycolmap.pair_id_to_image_pair(colmap_pair_id)
            image_id1, image_id2 = image_mapping[colmap_image_id1], image_mapping[colmap_image_id2]
            viewgraph.image_pairs[(image_id1, image_id2)] = ImagePair(image_id1, image_id2, matches=matches)
            pbar.update(1)
    colmap_db.close()
    viewgraph.filter_pairs_by_matches_validity(min_num_matches=15, silent=True)
    return viewgraph, gcameras, gimages



class GlobalMapper:
    def __init__(self, cfg: DGSfMConfig):
        self.cfg = cfg

    def tracks_generation(self, view_graph: ViewGraph, images: ImageBatch):
        gtracks = TrackManager().establish_full_tracks(view_graph, images)
        gtracks.filter_tracks_by_length(self.cfg.mapping.min_num_views_per_track)
        return gtracks

    def rotation_averaging(self, view_graph: ViewGraph, images: ImageBatch) -> None:
        rotation_averager = RotationAveraging()
        for _ in range(self.cfg.mapping.ra_num_runs):
            rotation_averager.optimize(view_graph, images)
            view_graph.filter_pairs_by_rotation_error(images, self.cfg.mapping.max_rotation_error)
            view_graph.keep_largest_connected_component(images)

    def scale_averaging(self, view_graph: ViewGraph, images: ImageBatch) -> None:
        scale_averager = ScaleAveraging(self.cfg)
        scale_averager.initialize_from_maximum_spanning_tree(view_graph, images)
        scale_averager.optimize(view_graph, images)

    def global_positioning(self, view_graph: ViewGraph, cameras: CameraBatch, images: ImageBatch, tracks: TrackBatch):
        global_positioner = GlobalPositioning(self.cfg)
        global_positioner.initialize(view_graph, tracks, cameras, images)
        global_positioner.optimize(cameras, images, tracks)
        tracks.filter_observations_by_angle(images, cameras, self.cfg.mapping.max_angle_error)

    def global_bundle_adjustment(self, cameras: CameraBatch, images: ImageBatch, tracks: TrackBatch, *, fixed_cam_id: int):
        bundle_adjuster = BundleAdjusterBAE(self.cfg)
        for iter in range(self.cfg.mapping.ba_num_runs):
            bundle_adjuster.solve(cameras, images, tracks, fixed_cam_id=fixed_cam_id)
            tracks.filter_observations_by_reproj_error(images, cameras, self.cfg.mapping.max_reproj_error_norm * max(1, self.cfg.mapping.ba_num_runs - iter))
        tracks.filter_tracks_by_triangulation_angle(images, self.cfg.mapping.min_triang_angle)



class DGSfM:
    def __init__(self, cfg: DGSfMConfig) -> None:
        self.cfg = cfg
        self.relpose_estimator = RePoseDEstimator(cfg.viewgraph.epipolar_error, cfg.viewgraph.reprojection_error)
        self.pose_metrics = PoseMetrics(cfg.metric_error_type, cfg.pose_error_thresholds)
        self.global_mapper = GlobalMapper(cfg)

    def run_mapping_on_dataset(self, dataset: COLMAPDataset):
        for scene in dataset:
            # if scene.scene_name not in ['botanical_garden', 'boulders']:
            #     continue
            self.run_mapping(scene, dataset.name)

    def run_mapping_on_scene(self, scene_name: str, dataset: COLMAPDataset):
        for scene in dataset:
            if scene_name == scene.scene_name:
                self.run_mapping(scene, dataset.name)
                break

    def focal_averaging(self, viewgraph: ViewGraph, gcameras: CameraBatch, gimages: ImageBatch):
        # average solver focals
        solver_focals = defaultdict(list)
        for image_pair in viewgraph.image_pairs.values():
            solver_focals[image_pair.image_id1].append(image_pair.intrinsic1[0, 0])
            solver_focals[image_pair.image_id2].append(image_pair.intrinsic2[0, 0])
        with tqdm(total=len(solver_focals), desc="Focal Averaging") as pbar:
            for image_id, focals in solver_focals.items():
                gcameras[gimages[image_id].cam_id].focal_solver = np.median(focals)
                pbar.update(1)

        # decide which focals to use
        initial_focals = []
        for gcamera in gcameras:
            # focal_graph = np.mean([gcamera.focal_mde, gcamera.focal_solver])
            focal_graph = gcamera.focal_solver
            gcamera.focal_graph = focal_graph
            gcamera.focal = focal_graph
            initial_focals.append(focal_graph)
        print(f"Initial focals: min {min(initial_focals):.2f}, median {np.median(initial_focals):.2f}, max {max(initial_focals):.2f}")
    
    def compute_relative_depth_scale_errors(self, pair: ImagePair, feats1, depths1, depths2, intrinsic1):
        # from 1 to 2
        points12 = lift_and_transform(feats1, depths1, intrinsic1, pair.rotation, pair.translation)
        scaled_depths2 = pair.scale * depths2
        return np.abs(points12[:, 2] - scaled_depths2) / scaled_depths2

    def compute_bidir_relative_depth_scale_errors(self, pair: ImagePair, feats1, feats2, depths1, depths2, intrinsic1, intrinsic2):
        # from 1 to 2
        points12 = lift_and_transform(feats1, depths1, intrinsic1, pair.rotation, pair.translation)
        scaled_depths2 = pair.scale * depths2
        errors12 = np.abs(points12[:, 2] - scaled_depths2) / scaled_depths2
        # from 2 to 1
        points21 = lift_and_transform(feats2, scaled_depths2, intrinsic2, pair.rotation.T, -pair.rotation.T @ pair.translation)
        errors21 = np.abs(points21[:, 2] - depths1) / depths1
        return (errors12 + errors21) / 2
    
    def filter_pairs_by_doppelgangers(self, viewgraph: ViewGraph, gimages: ImageBatch, pairs_path: Path):
        doppelganger_scores = {}
        if self.cfg.viewgraph.disambiguation_method == 'doppelgangers' and pairs_path.exists():
            dopp_image_pairs, dopp_scores = load_image_pairs_with_scores(pairs_path)
            for pair, score in zip(dopp_image_pairs, dopp_scores):
                doppelganger_scores[pair] = score
            viewgraph.filter_pairs_by_disambiguation_scores(gimages, self.cfg.viewgraph.doppelgangers_threshold, doppelganger_scores)

    def filter_inliers_by_depth_consistency(self, viewgraph: ViewGraph, gimages: ImageBatch):
        for pair in tqdm(viewgraph.image_pairs.values(), desc=f"Scaled-Depth Filtering (> {self.cfg.viewgraph.depth_error_threshold} depth error)", total=viewgraph.num_pairs):
            feats1, feats2 = sample_correspondences(gimages[pair.image_id1].features, gimages[pair.image_id2].features, pair.matches[pair.inliers])
            depths1, depths2 = sample_correspondences(gimages[pair.image_id1].depths, gimages[pair.image_id2].depths, pair.matches[pair.inliers])
            depth_errors = self.compute_bidir_relative_depth_scale_errors(pair, feats1, feats2, depths1, depths2, pair.intrinsic1, pair.intrinsic2)
            pair.inliers = pair.inliers[depth_errors < self.cfg.viewgraph.depth_error_threshold]
            pair.weight = len(pair.inliers) / len(pair.matches)
        viewgraph.filter_pairs_by_inliers_validity(min_num_inliers=self.cfg.viewgraph.min_num_inliers, min_inlier_ratio=self.cfg.viewgraph.min_inlier_ratio)
    
    def estimate_two_view_geometry(self, viewgraph: ViewGraph, gimages: ImageBatch, gcameras: CameraBatch):
        for pair_key in tqdm(viewgraph.image_pairs.keys(), desc="RelPose with RePoseD", total=viewgraph.num_pairs):
            image_pair = viewgraph.image_pairs[pair_key]
            gimage1, gimage2 = gimages[image_pair.image_id1], gimages[image_pair.image_id2]
            gcam1, gcam2 = gcameras[gimage1.cam_id], gcameras[gimage2.cam_id]
            kpts1, kpts2 = sample_correspondences(gimage1.features, gimage2.features, image_pair.matches)
            kpts1_d, kpts2_d = sample_correspondences(gimage1.depths, gimage2.depths, image_pair.matches)
            depth_conf1, depth_conf2 = sample_correspondences(gimage1.depths_confs, gimage2.depths_confs, image_pair.matches)
            valid_depths = (
                np.isfinite(kpts1_d) & (kpts1_d > 0)
                & np.isfinite(kpts2_d) & (kpts2_d > 0)
                & np.isfinite(depth_conf1) & (depth_conf1 > 0)
                & np.isfinite(depth_conf2) & (depth_conf2 > 0)
            )
            valid_match_indices = np.flatnonzero(valid_depths)
            # Keep the same minimum input size as database pair filtering.
            if len(valid_match_indices) < max(15, self.cfg.viewgraph.min_num_inliers):
                image_pair.is_valid = False
                image_pair.inliers = np.empty(0, dtype=np.uint32)
                image_pair.weight = 0.0
                continue
            kpts1, kpts2 = kpts1[valid_depths], kpts2[valid_depths]
            kpts1_d, kpts2_d = kpts1_d[valid_depths], kpts2_d[valid_depths]
            relpose = self.relpose_estimator.solve(kpts1, kpts2, kpts1_d, kpts2_d, gimage1.image_size, gimage2.image_size, None, None, self.cfg.viewgraph.share_focal)

            if not is_valid_focal(relpose.intrinsic1[0, 0], relpose.intrinsic2[0, 0], gimage1.image_size[1], gimage2.image_size[1]):
                # print(f"Not valid focal: image1 -> {relpose.intrinsic1[0, 0]:.2f} image2 -> {relpose.intrinsic2[0, 0]:.2f} whereas \nfrom MDE image1 -> {gcam1.focal_mde:.2f} image2 -> {gcam2.focal_mde:.2f}")
                k_mde1, k_mde2 = focal2intrinsic(gcam1.focal_mde, *gcam1.principal_point), focal2intrinsic(gcam2.focal_mde, *gcam2.principal_point)
                relpose = self.relpose_estimator.solve(kpts1, kpts2, kpts1_d, kpts2_d, gimage1.image_size, gimage2.image_size, k_mde1, k_mde2, self.cfg.viewgraph.share_focal)

            image_pair.scale = relpose.scale
            image_pair.rotation = relpose.rotation
            image_pair.translation = relpose.translation
            # Solver inliers index the filtered arrays; downstream code indexes matches.
            image_pair.inliers = valid_match_indices[relpose.inliers]
            image_pair.weight = len(image_pair.inliers) / len(image_pair.matches)
            image_pair.intrinsic1 = relpose.intrinsic1
            image_pair.intrinsic2 = relpose.intrinsic2

        viewgraph.filter_pairs_by_inliers_validity(min_num_inliers=self.cfg.viewgraph.min_num_inliers, min_inlier_ratio=self.cfg.viewgraph.min_inlier_ratio)
        viewgraph.filter_pairs_by_scale_validity()

    def run_mapping(self, scene: Scene, dataset_name: str):
        print(f"Running DGSfM on {scene.scene_name} of {dataset_name} dataset...")
        #####################################################################################
        # # VIEWGRAPH CREATION
        #####################################################################################
        print("="*40, "\nVIEWGRAPH CREATION\n", "="*40)
        viewgraph, gcameras, gimages = read_colmap_database(self.cfg, scene)
        # Doppelgangers++ Edge Pruning
        self.filter_pairs_by_doppelgangers(viewgraph, gimages, scene.doppelgangers_pairs_path)
        # RePoseD 
        self.estimate_two_view_geometry(viewgraph, gimages, gcameras)
        # Scaled-Depth Consistency Filtering
        self.filter_inliers_by_depth_consistency(viewgraph, gimages)
        # Triplet-guided Edge Pruning
        viewgraph.verify_triplets(gimages, self.cfg.viewgraph.min_edge_score)
        #####################################################################################
        # # GLOBAL MAPPING
        #####################################################################################
        print("="*40, "\nGLOBAL MAPPING\n", "="*40)
        self.focal_averaging(viewgraph, gcameras, gimages)
        viewgraph.keep_largest_connected_component(gimages) 
        self.global_mapper.rotation_averaging(viewgraph, gimages)
        self.global_mapper.scale_averaging(viewgraph, gimages)
        gtracks = self.global_mapper.tracks_generation(viewgraph, gimages)
        self.global_mapper.global_positioning(viewgraph, gcameras, gimages, gtracks)
        self.global_mapper.global_bundle_adjustment(gcameras, gimages, gtracks, fixed_cam_id=viewgraph.fixed_cam_id)
        # SAVE RESULTS
        for gcamera in gcameras: gcamera.focal_final = gcamera.focal
        save_image_focals(scene.focals_list_path, gcameras, gimages)
        save_colmap_model(scene.colmap_dir, scene.image_dir, gcameras, gimages, gtracks)
        #####################################################################################
        # # POSE EVALUATION
        #####################################################################################
        if cfg.dataset_name in ['ETH3D', 'IMC2021']:
            print("="*40, "\nPOSE EVALUATION\n", "="*40)
            gt_model, pd_model = scene.load_models()
            errors = self.pose_metrics.compute_errors(gt_model, pd_model)
            pose_results = {"aucs": self.pose_metrics.compute_AUC(errors), "num_images": gt_model.num_images(), "num_reg_images": pd_model.num_images() if pd_model is not None else 0}
            self.pose_metrics.create_result_table({scene.scene_name: pose_results}, dataset.name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="Path to the configuration file", default="configs/eth3d.yaml")
    args = parser.parse_args()
    cfg = load_dgsfm_config(args.config)
    dataset = COLMAPDataset(cfg)

    dgsfm = DGSfM(cfg)
    # dgsfm.run_mapping_on_dataset(dataset)
    dgsfm.run_mapping_on_scene("london_bridge", dataset)
    
    if cfg.dataset_name in ['ETH3D', 'IMC2021']:
        eval_dataset(cfg)
