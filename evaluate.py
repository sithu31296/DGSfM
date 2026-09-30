import multiprocessing
import functools
import argparse
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

    aucs = metric.compute_avg_metrics(results)
    results['average'] = {"aucs": aucs, "num_images": "-", "num_reg_images": "-"}
    metric.create_result_table(results, "ETH3D")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="Path to the configuration file", default="configs/eth3d.yaml")
    args = parser.parse_args()
    cfg = load_dgsfm_config(args.config)
    
    # cfg.run_name = "ours_MoGe2_LoMaB_LoMaB"
    eval_dataset(cfg)
