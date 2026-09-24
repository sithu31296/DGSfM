import pycolmap
import numpy as np
import networkx as nx
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict
from dataclasses import dataclass, field
from dgsfm.utils.geometry import calculate_angle_between_rotations, calculate_angle_between_translations
from dgsfm.utils.colmap_utils import rotmat2qvec
from dgsfm.database.structures import ImageBatch



def _empty_matches() -> np.ndarray:
    return np.empty((0, 2), dtype=np.uint32)


def _empty_inliers() -> np.ndarray:
    return np.empty(0, dtype=np.uint32)


def _identity_rotation() -> np.ndarray:
    return np.eye(3, dtype=np.float32)


def _zero_translation() -> np.ndarray:
    return np.zeros(3, dtype=np.float32)


@dataclass(slots=True)
class ImagePair:
    """Mutable two-view geometry associated with an image-graph edge."""

    image_id1: int
    image_id2: int
    pair_id: int = field(init=False)
    is_valid: bool = True

    matches: np.ndarray = field(default_factory=_empty_matches)
    inliers: np.ndarray = field(default_factory=_empty_inliers)
    weight: float = 0.0

    scale: float = 1.0
    rotation: np.ndarray = field(default_factory=_identity_rotation)
    translation: np.ndarray = field(default_factory=_zero_translation)
    intrinsic1: np.ndarray = field(default_factory=_identity_rotation)
    intrinsic2: np.ndarray = field(default_factory=_identity_rotation)
    

    def __post_init__(self) -> None:
        self.image_id1 = int(self.image_id1)
        self.image_id2 = int(self.image_id2)
        if self.image_id1 == self.image_id2:
            raise ValueError("An image pair must contain two different images")

        self.pair_id = int(pycolmap.image_pair_to_pair_id(self.image_id1, self.image_id2))
        self.matches = np.asarray(self.matches)
        self.inliers = np.asarray(self.inliers)
        self.rotation = np.asarray(self.rotation)
        self.translation = np.asarray(self.translation)
        self.intrinsic1 = np.asarray(self.intrinsic1)
        self.intrinsic2 = np.asarray(self.intrinsic2)
        self.weight = float(self.weight)
        self.scale = float(self.scale)

        if self.matches.ndim != 2 or self.matches.shape[1] != 2:
            raise ValueError("matches must have shape (N, 2)")
        if self.inliers.ndim != 1:
            raise ValueError("inliers must be a one-dimensional array")
        if self.rotation.shape != (3, 3):
            raise ValueError("rotation must have shape (3, 3)")
        if self.translation.shape != (3,):
            raise ValueError("translation must have shape (3,)")
        if self.intrinsic1.shape != (3, 3) or self.intrinsic2.shape != (3, 3):
            raise ValueError("intrinsics must have shape (3, 3)")

    @property
    def relpose(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.rotation
        T[:3, -1] = self.translation
        return T

    @property
    def rot_quat(self):
        return rotmat2qvec(self.rotation)


class ViewGraph:
    """A stricter, cycle-aware alternative to ``ViewGraph`` for comparison."""
    def __init__(self):
        self.image_pairs: dict[tuple[int, int], ImagePair] = {}
        self.graph = nx.Graph()

        # fixed camera parameters
        self.fixed_cam_id = -1
        self.fixed_cam_scale = 1.0
        self.fixed_cam_rot = np.eye(3)
        self.fixed_cam_trans = np.zeros(3)

    @property
    def num_pairs(self) -> int:
        """Number of stored pairs, derived from the edge dictionary."""
        return len(self.image_pairs)

    def filter_pairs(self):
        self.image_pairs = {pair_id: pair for pair_id, pair in self.image_pairs.items() if pair.is_valid}

    def filter_pairs_by_matches_validity(self, min_num_matches: int = 15, silent: bool = False):
        """Filter image pairs by minimum number of matches."""
        num_initial_pairs = len(self.image_pairs)
        for pair in self.image_pairs.values():
            if len(pair.matches) < min_num_matches:
                pair.is_valid = False
        self.filter_pairs()
        num_filtered_pairs = num_initial_pairs - len(self.image_pairs)
        if not silent:
            print(f"\tFiltered {num_filtered_pairs}/{num_initial_pairs} image pairs with fewer than {min_num_matches} matches.")

    def filter_pairs_by_inliers_validity(self, min_num_inliers: int = 15, min_inlier_ratio: float = 0.1):
        """Filter image pairs by minimum number of inliers or inliers ratio."""
        num_initial_pairs = len(self.image_pairs)
        for pair in self.image_pairs.values():
            if not pair.is_valid: continue
            if len(pair.inliers) < min_num_inliers or pair.weight < min_inlier_ratio:
                pair.is_valid = False
        self.filter_pairs()
        num_filtered_pairs = num_initial_pairs - len(self.image_pairs)
        print(f"\tFiltered {num_filtered_pairs}/{num_initial_pairs} image pairs with fewer than {min_num_inliers} inliers and {min_inlier_ratio} inlier ratio.")

    def filter_pairs_by_scale_validity(self):
        """Filter image pairs by scale validity."""
        num_initial_pairs = len(self.image_pairs)
        for pair in self.image_pairs.values():
            if pair.scale <= 0.1 or pair.scale >= 10.0:
                pair.is_valid = False
        self.filter_pairs()
        num_filtered_pairs = num_initial_pairs - len(self.image_pairs)
        print(f"\tFiltered {num_filtered_pairs}/{num_initial_pairs} image pairs with invalid scale.")

    def filter_pairs_by_disambiguation_scores(self, gimages: ImageBatch, min_disambiguation_score: float = 0.5, pair_scores: dict = None):
        """Filter image pairs by disambiguation scores."""
        num_initial_pairs = len(self.image_pairs)
        old_image_pairs = deepcopy(self.image_pairs)

        threshold = min_disambiguation_score
        while threshold > 0.3:
            with tqdm(total=len(self.image_pairs), desc=f"Edge Pruning with Doppelgangers++ (< {threshold} edge score)") as pbar:
                for (image_id1, image_id2), image_pair in self.image_pairs.items():
                    pair_key = (gimages[image_id1].filename, gimages[image_id2].filename)
                    if pair_key in pair_scores and pair_scores[pair_key] < threshold:
                        image_pair.is_valid = False
                    pbar.update(1)
            self.filter_pairs()

            connected_size = len(self.find_largest_connected_component(gimages))
            if connected_size < len(gimages):
                threshold -= 0.1
                self.image_pairs = deepcopy(old_image_pairs)
            else:
                break

        self.filter_pairs()
        num_filtered_pairs = num_initial_pairs - len(self.image_pairs)
        print(f"\tFiltered {num_filtered_pairs}/{num_initial_pairs} image pairs with disambiguation score below {threshold}.")

    def filter_pairs_by_rotation_error(self, images: ImageBatch, max_angle: float = 5.0):
        num_initial_pairs = len(self.image_pairs)
        for (image_id1, image_id2), pair in self.image_pairs.items():
            image1, image2 = images[image_id1], images[image_id2]
            if pair.is_valid and image1.is_registered and image2.is_registered:
                r12_est = image2.rotation @ image1.rotation.T
                if calculate_angle_between_rotations(r12_est, pair.rotation) > max_angle:
                    pair.is_valid = False
        self.filter_pairs()
        num_filtered_pairs = num_initial_pairs - len(self.image_pairs)
        print(f"\tFiltered {num_filtered_pairs}/{num_initial_pairs} image pairs with relative rotation error above {max_angle} degrees.")

    def filter_pairs_by_translation_error(self, images: ImageBatch, max_angle: float = 5.0):
        """Filter image pairs based on the angle between two translation vectors"""
        num_initial_pairs = len(self.image_pairs)
        for (image_id1, image_id2), pair in self.image_pairs.items():
            image1, image2 = images[image_id1], images[image_id2]
            if pair.is_valid and image1.is_registered and image2.is_registered:
                pose12_est = image2.world2cam @ np.linalg.inv(image1.world2cam)
                t12_est = pose12_est[:3, -1]
                if calculate_angle_between_translations(t12_est, pair.translation) > max_angle:
                    pair.is_valid = False
        self.filter_pairs()
        num_filtered_pairs = num_initial_pairs - len(self.image_pairs)
        print(f"\tFiltered {num_filtered_pairs}/{num_initial_pairs} image pairs with relative translation error above {max_angle} degrees.")
    

    def add_nodes(self, images: ImageBatch):
        self.graph.remove_nodes_from(list(self.graph.nodes))
        self.graph.add_nodes_from([image.id for image in images if image.is_registered])

    def add_edges(self):
        self.graph.remove_edges_from(list(self.graph.edges))
        for pair in self.image_pairs.values():
            if pair.is_valid and pair.image_id1 in self.graph and pair.image_id2 in self.graph:
                self.graph.add_edge(pair.image_id1, pair.image_id2, weight=pair.weight)
    
    def find_connected_components(self, images: ImageBatch | None = None):
        self.add_nodes(images)
        self.add_edges()
        return [c for c in sorted(nx.connected_components(self.graph), key=len, reverse=True)]

    def find_largest_connected_component(self, images: ImageBatch):
        components = self.find_connected_components(images)
        return set(max(components, key=len, default=[]))

    def keep_largest_connected_component(self, images: ImageBatch) -> int:
        largest = self.find_largest_connected_component(images)
        for image in images:
            image.is_registered = image.id in largest
        for pair in self.image_pairs.values():
            pair.is_valid = pair.is_valid and pair.image_id1 in largest and pair.image_id2 in largest

        self.filter_pairs()
        self.add_nodes(images)
        self.add_edges()
        
        if largest and self.fixed_cam_id not in largest:
            self.set_fixed_cam_params(images)

        print(f"{len(largest)}/{len(images)} are within the largest connected component.")

    def set_fixed_cam_params(self, images: ImageBatch) -> None:
        candidates = [image for image in images if image.is_registered]
        if not candidates:
            raise ValueError("Cannot select a fixed camera without a registered image")

        fixed = max(
            candidates,
            key=lambda image: (len(self.graph.adj[image.id]), len(image.features)),
        )
        self.fixed_cam_id = fixed.id
        self.fixed_cam_rot = fixed.rotation.copy()
        self.fixed_cam_trans = fixed.translation.copy()
        self.fixed_cam_scale = float(fixed.scale)
        print(
            f"Set fixed camera to {fixed.filename} with {len(fixed.features)} "
            f"features and {len(self.graph.adj[fixed.id])} edges."
        )

    def maximum_spanning_tree(self, images: ImageBatch):
        if not nx.is_connected(self.graph): return {}
        if self.fixed_cam_id not in self.graph: self.set_fixed_cam_params(images)
        mst = nx.maximum_spanning_tree(self.graph, weight="weight")
        parents = {node: -1 for node in mst.nodes}
        parents[self.fixed_cam_id] = self.fixed_cam_id
        for parent, child in nx.bfs_edges(mst, self.fixed_cam_id):
            parents[child] = parent
        return parents

    def find_triplets(self) -> list[tuple[int, int, int]]:
        triplets = []
        for node1, neighbors1 in self.graph.adj.items():
            for node2 in neighbors1:
                if node2 <= node1:
                    continue
                for node3 in neighbors1.keys() & self.graph.adj[node2].keys():
                    if node3 > node2:
                        triplets.append((node1, node2, node3))
        return triplets

    def find_tau(self, images: ImageBatch, min_edge_score: float = 0.6):
        registered = [image.id for image in images if image.is_registered]
        num_nodes = len(registered)
        max_degree = max((len(self.graph.adj[image_id]) for image_id in registered), default=0)
        degree_density = np.clip(max_degree / (num_nodes - 1), 0.0, 1.0)
        return min_edge_score * (1.0 - degree_density) + degree_density

    def get_triplet_pairs(self, image_id1: int, image_id2: int, image_id3: int):
        return ((image_id1, image_id2), (image_id2, image_id3), (image_id1, image_id3))

    def _pair_for(self, image_id1: int, image_id2: int) -> ImagePair:
        """Return an edge regardless of its tuple-key orientation."""
        for key in ((image_id1, image_id2), (image_id2, image_id1)):
            if key in self.image_pairs:
                return self.image_pairs[key]
        raise KeyError(f"Missing image pair ({image_id1}, {image_id2})")
        
    def relative_pose(self, pair_list: tuple[int, int]):
        """Return ``R, t, scale`` oriented from image 1 to image 2."""
        image_id1, image_id2 = pair_list
        pair = self._pair_for(*pair_list)
        if pair.image_id1 == image_id1 and pair.image_id2 == image_id2:
            return pair.rotation, pair.translation, float(pair.scale)
        if pair.image_id1 != image_id2 or pair.image_id2 != image_id1:
            raise ValueError("Pair endpoints do not agree with its dictionary key")

        rotation = pair.rotation.T
        translation = -rotation @ pair.translation
        scale = np.inf if pair.scale == 0 else 1.0 / float(pair.scale)
        return rotation, translation, scale

    def rotation_cycle_error(self, pair1, pair2, pair3):
        rotation12, _, _ = self.relative_pose(pair1)
        rotation23, _, _ = self.relative_pose(pair2)
        rotation13, _, _ = self.relative_pose(pair3)
        return calculate_angle_between_rotations(rotation23 @ rotation12, rotation13)

    def translation_cycle_error(self, image_id1: int, image_id2: int, image_id3: int):
        rotation12, translation12, _ = self.relative_pose(image_id1, image_id2)
        rotation23, translation23, _ = self.relative_pose(image_id2, image_id3)
        _, translation13, _ = self.relative_pose(image_id1, image_id3)
        translation13_est = rotation23 @ translation12 + translation23
        denominator = np.linalg.norm(translation12) + np.linalg.norm(translation23) + np.linalg.norm(translation13)
        return float(np.linalg.norm(translation13_est - translation13) / (denominator + 1e-12))

    def scale_cycle_error(self, image_id1: int, image_id2: int, image_id3: int):
        _, _, scale12 = self.relative_pose(image_id1, image_id2)
        _, _, scale23 = self.relative_pose(image_id2, image_id3)
        _, _, scale13 = self.relative_pose(image_id1, image_id3)
        if scale12 <= 0 or scale23 <= 0 or scale13 <= 0:
            return np.inf
        return float(abs(np.log((scale12 * scale23) / scale13)))

    def relative_inliers_score(self, pair1, pair2, pair3):
        pair1_inliers = max(len(self._pair_for(*pair1).inliers), 1e-6)
        pair2_inliers = max(len(self._pair_for(*pair2).inliers), 1e-6)
        pair3_inliers = max(len(self._pair_for(*pair3).inliers), 1e-6)
        max_inliers = max([pair1_inliers, pair2_inliers, pair3_inliers])
        return pair1_inliers/max_inliers, pair2_inliers/max_inliers, pair3_inliers/max_inliers

    def verify_triplets(
        self,
        images: ImageBatch,
        min_edge_score: float = 0.6,
        max_rotation_error: float = 5.0,
        max_translation_error: float | None = None,
        max_scale_error: float | None = None,
    ) -> None:
        self.add_nodes(images)
        self.add_edges()
        num_initial_pairs = len(self.image_pairs)
        old_image_pairs = deepcopy(self.image_pairs)
        triplets = self.find_triplets()
        if not triplets: return

        edge_support = defaultdict(list)
        edge_geometry = defaultdict(list)
        for triplet in tqdm(triplets, desc=f"Edge Pruning with Triplets-inliers (< {min_edge_score} edge score)", total=len(triplets)):
            triplet_pairs = self.get_triplet_pairs(*triplet)
            relative_scores = self.relative_inliers_score(*triplet_pairs)

            is_consistent = self.rotation_cycle_error(*triplet_pairs) <= max_rotation_error
            # if max_translation_error is not None:
            #     is_consistent &= self.translation_cycle_error(*triplet) <= max_translation_error
            # if max_scale_error is not None:
            #     is_consistent &= self.scale_cycle_error(*triplet) <= max_scale_error

            for pair_id, relative_score in zip(triplet_pairs, relative_scores):
                edge_support[pair_id].append(relative_score)
                edge_geometry[pair_id].append(float(is_consistent))

        # threshold = self.find_tau(images, min_edge_score)
        threshold = min_edge_score
        while threshold > 0.1:
            for pair_id in edge_support:
                support = float(np.median(edge_support[pair_id]))
                geometry = float(np.mean(edge_geometry[pair_id]))
                # if support * geometry < threshold:
                if support < threshold:
                    self.image_pairs[pair_id].is_valid = False
            self.filter_pairs()

            connected_size = len(self.find_largest_connected_component(images))
            if connected_size < len(images):
                threshold -= 0.1
                self.image_pairs = deepcopy(old_image_pairs)
            else:
                break
        self.filter_pairs()
        num_filtered_pairs = num_initial_pairs - len(self.image_pairs)
        print(f"\tFiltered {num_filtered_pairs}/{num_initial_pairs} image pairs with triplet consistency score below {threshold:.2f}.")
            
        self.add_edges()

    
