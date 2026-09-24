import multiprocessing
import functools
from dgsfm.database.dataset import COLMAPDataset, Scene
from dgsfm.utils.metrics import PoseMetrics
from dgsfm.database.config import DGSfMConfig, load_dgsfm_config


def process_scene(scene: Scene, metric):
    gt_model, pd_model = scene.load_models()
    errors = metric.compute_errors(gt_model, pd_model)
    data = {
        "aucs": metric.compute_AUC(errors),
        "num_images": gt_model.num_images(),
        "num_reg_images": pd_model.num_images() if pd_model is not None else 0,
    }
    return scene.scene_name, data


def eval_dataset(cfg: DGSfMConfig):
    dataset = COLMAPDataset(cfg)
    metric = PoseMetrics("relative_auc", cfg.pose_error_thresholds)

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as p:
        results = dict(
            p.imap_unordered(
                functools.partial(
                    process_scene,
                    metric=metric
                ),
                dataset,
                chunksize=1
            )
        )
    results = {k: v for k, v in sorted(results.items(), key=lambda item: item[0])}
    
    # results = {
    #     "botanical_garden": {"aucs": np.array([68.78, 88.96, 93.36]), "num_images": 30, "num_reg_images": 30},
    #     "boulders": {"aucs": np.array([71.23, 88.84, 93.10]), "num_images": 26, "num_reg_images": 26},
    #     "bridge": {"aucs": np.array([43.18, 60.58, 70.27]), "num_images": 110, "num_reg_images": 110},
    #     "courtyard": {"aucs": np.array([53.90, 81.52, 88.81]), "num_images": 38, "num_reg_images": 38},
    #     "delivery_area": {"aucs": np.array([62.00, 85.40, 90.90]), "num_images": 44, "num_reg_images": 44},
    #     "door": {"aucs": np.array([87.84, 92.77, 95.04]), "num_images": 7, "num_reg_images": 7},
    #     "electro": {"aucs": np.array([51.85, 77.51, 85.22]), "num_images": 45, "num_reg_images": 45},
    #     "exhibition_hall": {"aucs": np.array([31.69, 70.40, 80.96]), "num_images": 68, "num_reg_images": 68},
    #     "facade": {"aucs": np.array([70.31, 89.21, 93.42]), "num_images": 76, "num_reg_images": 76},
    #     "kicker": {"aucs": np.array([63.68, 85.90, 91.27]), "num_images": 31, "num_reg_images": 31},
    #     "lecture_room": {"aucs": np.array([58.66, 83.73, 90.12]), "num_images": 23, "num_reg_images": 23},
    #     "living_room": {"aucs": np.array([74.86, 90.46, 94.09]), "num_images": 65, "num_reg_images": 65},
    #     "lounge": {"aucs": np.array([61.13, 83.12, 89.67]), "num_images": 10, "num_reg_images": 10},
    #     "meadow": {"aucs": np.array([67.82, 88.68, 93.21]), "num_images": 15, "num_reg_images": 15},
    #     "office": {"aucs": np.array([30.42, 51.42, 60.98]), "num_images": 26, "num_reg_images": 26},
    #     "old_computer": {"aucs": np.array([39.62, 73.14, 83.13]), "num_images": 54, "num_reg_images": 54},
    #     "pipes": {"aucs": np.array([61.34, 85.28, 90.73]), "num_images": 14, "num_reg_images": 14},
    #     "playground": {"aucs": np.array([75.05, 91.10, 94.62]), "num_images": 38, "num_reg_images": 38},
    #     "statue": {"aucs": np.array([67.68, 89.23, 93.54]), "num_images": 11, "num_reg_images": 11},
    #     "terrace": {"aucs": np.array([64.47, 87.37, 92.42]), "num_images": 23, "num_reg_images": 23},
    #     "terrace_2": {"aucs": np.array([77.43, 91.06, 94.58]), "num_images": 13, "num_reg_images": 13},
    #     "terrains": {"aucs": np.array([59.30, 85.49, 91.14]), "num_images": 42, "num_reg_images": 42}
    # }

    aucs = metric.compute_avg_metrics(results)
    results['average'] = {"aucs": aucs, "num_images": "-", "num_reg_images": "-"}
    metric.create_result_table(results, "ETH3D")


if __name__ == '__main__':
    cfg = load_dgsfm_config("configs/imc2021.yaml")
    cfg.run_name = "glomap_LoMaB_LoMaB3/0"
    # cfg.run_name = "ours_loma_lomab"
    # print(cfg.viewgraph.depth_model)
    eval_dataset(cfg)
