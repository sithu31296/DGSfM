import multiprocessing
import functools
from globalsfm.database.dataset import ETH3D, IMC2021, LaMAR
from globalsfm.utils.metrics import TriangulationMetrics


def process_scene(scene, metric):
    print(scene.scene_name)
    data = metric.compute_accuracy(scene.scene_root, scene.colmap_dir)
    return scene.scene_name, data


def eval_dataset(run_name="ours"):
    dataset = ETH3D(run_name=run_name)
    metric = TriangulationMetrics(dataset.pcd_tolerances)

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
    accs, comps = metric.compute_avg_metrics(results)
    results['average'] = {"accuracy": accs, "completeness": comps}
    metric.create_result_table(results, "ETH3D")

def eval_scene(run_name="ours"):
    dataset = ETH3D(run_name=run_name)
    metric = TriangulationMetrics(dataset.pcd_tolerances)

    results = {}
    for scene in dataset:
        if scene.scene_name not in [
            'courtyard', 'delivery_area', 'electro',
            'facade', 'kicker', 'meadow', 'office',
            'pipes', 'playground', 'terrace', 'terrains'
        ]: continue
        print(scene.scene_name)
        data = metric.compute_accuracy(run_name, scene.scene_root, scene.colmap_gt_dir, scene.colmap_dir)
        results[scene.scene_name] = data
    accs, comps = metric.compute_avg_metrics(results)
    results['average'] = {"accuracy": accs, "completeness": comps}
    metric.create_result_table(results, "ETH3D")

if __name__ == '__main__':
    # run_name = "ours"
    # run_name = 'ours_aliked_lightglue'
    # run_name = "ours_loma_loma"
    # run_name = "ours_loma_lomab_gtk"
    # run_name = "ours_MoGev2K_RoMav2_gtK"
    run_name = "ours_MoGev2_RoMav2"
    # run_name = "ours_MoGev2_RoMav1"

    # run_name = "colmap/0"
    # run_name = "glomap_aliked_lightglue/0"
    # run_name = "glomap_loma_lomag/0"

    # run_name = "instantsfm_sift_nn/0"
    # run_name = "instantsfm_aliked_lightglue/0"
    # run_name = "instantsfm_loma_lomab/0"

    # run_name = "mpsfm_sp_roma_gt/rec"

    # run_name = "colmap_gt"

    # run_name = "da3"

    # eval_dataset(run_name)
    eval_scene(run_name)