import torch
import copy
import shutil
import pycolmap
import subprocess
import numpy as np
# import open3d as o3d
from pathlib import Path
from tabulate import tabulate, SEPARATING_LINE
from scipy.spatial import KDTree
from dgsfm.utils.geometry import normalize, calculate_angle_between_poses
from dgsfm.utils.colmap_utils import read_points3D_binary, read_points3D_text


EPS = 1e-9


def rotation_error(R_gt, R_pd):
    sin_angle = np.linalg.norm(R_gt - R_pd) / (2 * np.sqrt(2))
    sin_angle = max(min(1.0, sin_angle), -1.0)
    error = np.rad2deg(2 * np.arcsin(sin_angle))
    return error


def translation_error(t_gt, t_pd):
    t_gt = normalize(t_gt)
    t_pd = normalize(t_pd)

    loss = np.maximum(1e-15, (1.0 - np.sum(t_pd * t_gt)**2))
    error = np.rad2deg(np.arccos(np.sqrt(1 - loss)))
    return error



def rel_angle_error(angle_12, angle_1, angle_2):
    est = (angle_2 - angle_1) - angle_12
    while est >= np.pi:
        est -= 2*np.pi
    while est < -np.pi:
        est += 2*np.pi

    # inject random noise if the angle is too close to the boundary
    # to break the possible balance at the local minima
    if (est > np.pi - 0.01) or (est < -np.pi + 0.01):
        if est < 0:
            est += (np.random.rand() % 1000) / 1000.0 * 0.01
        else:
            est -= (np.random.rand() % 1000) / 1000.0 * 0.01
    return est



def colmap_alignment(
    sparse_path: Path,
    sparse_gt_path: Path,
    sparse_aligned_path: Path,
    max_ref_model_error: float = 0.001,
) -> None:
    if sparse_aligned_path.exists():
        shutil.rmtree(sparse_aligned_path)

    if sparse_path.exists():
        sparse_aligned_path.mkdir(parents=True, exist_ok=True)
        subprocess.call(
            [
                "colmap",
                "model_aligner",
                "--input_path",
                sparse_path,
                "--ref_model_path",
                sparse_gt_path,
                "--output_path",
                sparse_aligned_path,
                "--alignment_max_error",
                str(max_ref_model_error),
            ]
        )



class DepthMetrics:
    """Depth Map Evaluation Metrics

    1. RMSE - Root Mean Square Error
    2. RMSE_log
    3. A.Rel - Absolute Mean Relative Error
    4. delta_i - Percentage of inlier pixels with threshold 1.25^i
    5. SI_log - Scale-Invariant Error in log scale = 100 * sqrt(Var(epsilon_log))

    """
    def __init__(self):
        pass

    def evaluate(self, gts: torch.Tensor, preds: torch.Tensor,
                 masks: torch.Tensor, max_depth=None):
        metrics = {}
        preds = torch.nn.functional.interpolate(preds, gts.shape[-2:], mode='bilinear')

        for i, (gt, pred, mask) in enumerate(zip(gts, preds, masks)):
            if max_depth is not None:
                mask = mask & (gt <= max_depth)

            tau = self.tau(gt[mask], pred[mask])
            tau_ssi = self.tau(gt[mask], self.ssi(gt[mask], pred[mask]))
            tau_si = self.tau(gt[mask], self.si(gt[mask], pred[mask]))

            d1 = self.tau(gt[mask], pred[mask])
            d1_ssi = self.delta_1(gt[mask], self.ssi(gt[mask], pred[mask]))
            d1_si = self.delta_1(gt[mask], self.si(gt[mask], pred[mask]))

            arel = self.tau(gt[mask], pred[mask])
            arel_ssi = self.arel(gt[mask], self.ssi(gt[mask], pred[mask]))
            arel_si = self.arel(gt[mask], self.si(gt[mask], pred[mask]))




    def rmse(self, tensor1, tensor2):
        return torch.sqrt(((tensor1 - tensor2)**2).mean())

    def rmse_log(self, tensor1, tensor2):
        return torch.sqrt(((tensor1.log() - tensor2.log())**2).mean())

    def delta_i(self, tensor1, tensor2, exponent=1):
        ratios = torch.maximum((tensor1 / tensor2), (tensor2 / tensor1))
        inliers = (ratios < 1.25**exponent)
        return inliers.to(torch.float32).mean()

    def delta_1(self, tensor1, tensor2):
        return self.delta_i(tensor1, tensor2, 1)

    def delta_2(self, tensor1, tensor2):
        return self.delta_i(tensor1, tensor2, 2)

    def delta_3(self, tensor1, tensor2):
        return self.delta_i(tensor1, tensor2, 3)

    def delta_auc(self, tensor1, tensor2):
        exponents = torch.linspace(0.01, 5.0, steps=100, device=tensor1.device)
        deltas = [self.delta(tensor1, tensor2, exponent) for exponent in exponents]
        return torch.trapz(torch.tensor(deltas, device=tensor1.device), exponents) / 5.0

    def tau(self, tensor1, tensor2, prec=0.03):
        ratios = torch.maximum((tensor1 / tensor2), (tensor2 / tensor1))
        inliers = (ratios < (1.0 + prec))
        return inliers.to(torch.float32).mean()

    def ssi(self, tensor1, tensor2):
        tensor2_homo = torch.stack([tensor2.detach(), torch.ones_like(tensor2).detach()], dim=1)
        scale_shift = torch.linalg.inv(tensor2_homo.T @ tensor2_homo + EPS) @ (tensor2_homo.T @ tensor1.unsqueeze(1))
        scale, shift = scale_shift.squeeze().chunk(2, dim=0)
        return tensor2 * scale + shift

    def si(self, tensor1, tensor2):
        return tensor2 * torch.median(tensor1) / torch.median(tensor2)

    def arel(self, tensor1, tensor2):
        # tensor2 = self.si(tensor1, tensor2)
        # return (torch.abs(tensor1 - tensor2) / tensor1).mean()
        return (torch.abs(tensor1 - tensor2) / tensor1).mean()

    def sqrel(self, tensor1, tensor2):
        return (((tensor1 - tensor2)**2) / tensor1).mean()

    def log10(self, tensor1, tensor2):
        return torch.abs(torch.log10(tensor1) - torch.log10(tensor2)).mean()

    def si_log(self, tensor1, tensor2):
        return 100 * torch.std(torch.log(tensor2) - torch.log(tensor1)).mean()

    def median_log(self, tensor1, tensor2):
        return 100 * (torch.log(tensor2) - torch.log(tensor1)).median().abs()





class PointCloudMetrics:
    """Point Cloud (Reconstruction) Metrics

    1. CD - Chamfer Distance
    2. F_A - F-score
    """
    def __init__(self):
        pass


    def chamfer_distance(self, pcd1, pcd2):
        A, B = np.asarray(pcd1), np.asarray(pcd2)

        # build KD-Trees for efficient nearest-neighbor search
        tree_a = KDTree(A)
        tree_b = KDTree(B)

        # query nearest neighbors
        dist_a_to_b, _ = tree_b.query(A)    # distance from A to nearest in B
        dist_b_to_a, _ = tree_a.query(B)    # distance from B to nearest in A

        return np.mean(dist_a_to_b) + np.mean(dist_b_to_a)


    def load_point_cloud(self, file_path: str | Path):
        file_path = Path(file_path)
        if file_path.suffix == ".ply":
            pcd = o3d.io.read_point_cloud(str(file_path))
            return np.asarray(pcd.points)
        elif file_path.suffix == ".txt":
            points3d = read_points3D_text(file_path)
            return np.stack([point.xyz for point in points3d.values()])
        elif file_path.suffix == ".bin":
            points3d = read_points3D_binary(file_path)
            return np.stack([point.xyz for point in points3d.values()])



class TriangulationMetrics:
    def __init__(self, error_thresholds=[0.01, 0.02, 0.05]) -> None:
        self.tool_path = Path("/home/aungsith/projects/multi-view-evaluation/build/ETH3DMultiViewEvaluation")
        self.error_thresholds = error_thresholds

    def _evaluate(self, ply_path, scan_path):
        cmd = [
            str(self.tool_path),
            "--reconstruction_ply_path",
            str(ply_path),
            "--ground_truth_mlp_path",
            str(scan_path),
            "--tolerances",
            ",".join(map(str, self.error_thresholds))
        ]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        out, err = p.communicate()
        lines = out.decode().split("\n")
        accuracy, completeness = None, None
        for line in lines:
            if line.startswith("Accuracies"):
                accuracy = list(map(float, line.replace("Accuracies: ", "").split(" ")))
            if line.startswith("Completenesses"):
                completeness = list(map(float, line.replace("Completenesses: ", "").split(" ")))

        accuracy_list = np.zeros(len(self.error_thresholds), dtype=np.float32)
        completeness_list = np.zeros(len(self.error_thresholds), dtype=np.float32)
        for i, tolerance in enumerate(self.error_thresholds):
            accuracy_list[i] = accuracy[i]
            completeness_list[i] = completeness[i]
        return {"accuracy": accuracy_list*100, "completeness": completeness_list*100}

    def compute_accuracy(self, run_name, scene_root: Path, gt_model_path, pd_model_path: Path):
        ply_path = Path(pd_model_path) / "reconstruction.ply"
        pd_model_path2 = scene_root / f"{run_name}_aligned"
        colmap_alignment(pd_model_path, gt_model_path, pd_model_path2)
        try:
            pd_model = pycolmap.Reconstruction(str(pd_model_path2))
            pd_model.export_PLY(ply_path)
            gt_scan_path = scene_root / "dslr_scan_eval" / "scan_alignment.mlp"
            results = self._evaluate(ply_path, gt_scan_path)
        except ValueError:
            results = {"accuracy": np.array([0.0, 0.0, 0.0]), "completeness": np.array([0.0, 0.0, 0.0])}
        return results

    def compute_avg_metrics(self, all_metrics: dict[str, dict]):
        acc_sum, com_sum = None, None
        for metrics in all_metrics.values():
            if acc_sum is None:
                acc_sum = copy.copy(metrics['accuracy'])
            else:
                acc_sum += metrics['accuracy']
            if com_sum is None:
                com_sum = copy.copy(metrics["completeness"])
            else:
                com_sum += metrics['completeness']
        return np.array(acc_sum) / len(all_metrics), np.array(com_sum) / len(all_metrics)

    def create_result_table(self, all_results: dict[str, dict], dataset_name: str):
        table = []
        for scene, results in all_results.items():
            acc_scores = results['accuracy'].tolist()
            com_scores = results['completeness'].tolist()
            if scene == 'average':
                table.append(SEPARATING_LINE)
            table.append([scene,
                *[f"{score:.2f}" for score in acc_scores],
                *[f"{score:.2f}" for score in com_scores]])

        # print(f"{dataset_name}                 AUC @ X deg (%)             images")
        header = ['scene', *[f"{th}" for th in self.error_thresholds], *[f"{th}" for th in self.error_thresholds]]
        print(tabulate(table, headers=header, tablefmt='rst'))



class PoseMetrics:
    def __init__(self, error_type: str, error_thresholds: list[int]):
        self.error_type = error_type    # relative_auc, absolute_auc, relative_recall, absolute_recall
        self.error_thresholds = np.array(error_thresholds)

    def compute_rel_errors(self, gt_model: pycolmap.Reconstruction, pd_model: pycolmap.Reconstruction):
        images = {}
        if pd_model is not None:
            for image in pd_model.images.values():
                images[image.name] = image

        dts, dRs = [], []
        for this_image_gt in gt_model.images.values():
            if this_image_gt.name not in images:
                for _ in range(gt_model.num_images() - 1):
                    dts.append(np.inf)
                    dRs.append(180)
                continue

            this_image = images[this_image_gt.name]
            for other_image_gt in gt_model.images.values():
                if this_image_gt.image_id == other_image_gt.image_id:
                    continue
                if other_image_gt.name not in images:
                    dts.append(np.inf)
                    dRs.append(180)
                    continue
                other_image = images[other_image_gt.name]
                other_from_this = (other_image.cam_from_world() * this_image.cam_from_world().inverse())
                other_from_this_gt = (other_image_gt.cam_from_world() * this_image_gt.cam_from_world().inverse())
                estimated_from_gt = other_from_this.inverse() * other_from_this_gt
                dt = calculate_angle_between_poses(other_from_this.translation, other_from_this_gt.translation)
                dR = np.rad2deg(estimated_from_gt.rotation.angle())
                dts.append(dt)
                dRs.append(dR)
        return np.array(dts), np.array(dRs)


    def compute_abs_errors(self, gt_model: pycolmap.Reconstruction, pd_model: pycolmap.Reconstruction):
        dts = np.full(len(gt_model.images), fill_value=np.inf, dtype=np.float32)
        dRs = np.full(len(gt_model.images), fill_value=180, dtype=np.float32)
        images = {}
        if pd_model is not None:
            for image in pd_model.images.values():
                images[image.name] = image

        for i, image_gt in enumerate(gt_model.images.values()):
            if image_gt.name not in images:
                continue
            image = images[image_gt.name]
            estimated_from_gt = (image.cam_from_world() * image_gt.cam_from_world().inverse())
            dts[i] = np.linalg.norm(estimated_from_gt.translation)
            dRs[i] = np.rad2deg(estimated_from_gt.rotation.angle())
        return dts, dRs

    def compute_errors(self, gt_model: pycolmap.Reconstruction, pd_model: pycolmap.Reconstruction):
        dts, dRs = self.compute_abs_errors(gt_model, pd_model) if "absolute" in self.error_type else self.compute_rel_errors(gt_model, pd_model)
        return np.maximum(dts, dRs)

    def compute_AUC(self, errors: np.ndarray, min_error: float = 0.):
        errors = np.sort(errors)
        aucs = np.zeros(len(self.error_thresholds), dtype=np.float32)
        for i, t in enumerate(self.error_thresholds):
            relevant = errors[errors <= t]
            points = np.concatenate([[0], relevant, [t]])
            accs = np.searchsorted(errors, points, side="right") / len(errors)
            aucs[i] = np.trapezoid(accs, points) / t
        return aucs * 100

    # def compute_AUC(self, errors: np.ndarray, min_error: float = 0.):
    #     # errors = np.sort(errors)
    #     aucs = np.zeros(len(self.error_thresholds), dtype=np.float32)
    #     for i, t in enumerate(self.error_thresholds):   
    #         bins = np.arange(t + 1)
    #         histogram, _ = np.histogram(errors, bins=bins)
    #         num_pairs = float(len(errors))
    #         normalized_histogram = histogram.astype(float) / num_pairs
    #         aucs[i] = np.mean(np.cumsum(normalized_histogram))
    #     return aucs * 100

    def compute_recall(self, errors: np.ndarray):
        recalls = np.zeros(len(self.error_thresholds), dtype=np.float64)
        for i, t in enumerate(self.error_thresholds):
            recalls[i] = np.sum(errors <= t) / len(errors)
        return recalls * 100

    # def compute_avg_metrics(self, all_metrics: dict[str, dict]):
    #     auc_sum, recall_sum = None, None
    #     for metrics in all_metrics.values():
    #         if auc_sum is None:
    #             auc_sum = copy.copy(metrics['aucs'])
    #             recall_sum = copy.copy(metrics['recalls'])
    #         else:
    #             auc_sum += metrics['aucs']
    #             recall_sum += metrics['recalls']
    #     return np.array(auc_sum) / len(all_metrics), np.array(recall_sum) / len(all_metrics)

    def compute_avg_metrics(self, all_metrics: dict[str, dict]):
        auc_sum = None
        for metrics in all_metrics.values():
            if auc_sum is None:
                auc_sum = copy.copy(metrics['aucs'])
            else:
                auc_sum += metrics['aucs']
        return np.array(auc_sum) / len(all_metrics)

    def create_result_table(self, all_results: dict[str, dict], dataset_name: str):
        table = []
        for scene, results in all_results.items():
            scores = results['aucs'].tolist() if self.error_type.endswith("auc") else results['recalls'].tolist()
            if scene == 'average':
                table.append(SEPARATING_LINE)
            table.append([scene, *[f"{score:.2f}" for score in scores], results['num_reg_images'], results['num_images']])

        print(f"{dataset_name}              AUC @ X deg (%)      images")
        header = ['scene', *[f"{th}" for th in self.error_thresholds], "reg", "all"]
        print(tabulate(table, headers=header, tablefmt='rst'))

