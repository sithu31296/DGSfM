import pycolmap
import itertools
from pathlib import Path
from dgsfm.utils.colmap_utils import colmap_alignment
from dgsfm.utils.io import read_image_paths, get_grouped_filenames
from dgsfm.utils.fileio import load_image_pairs, save_image_pairs
from dgsfm.database.config import DGSfMConfig


class Scene:
    def __init__(self, scene_name: str, cfg: DGSfMConfig) -> None:
        self.cfg = cfg
        self.position_accuracy_gt = 0.0
        self.metric_error_type = cfg.metric_error_type
        self.scene_name = scene_name
        self.match_type = cfg.matching_type
        self.run_name = cfg.run_name
        self.scene_root = Path(cfg.root) / scene_name
        self.image_dir = self.scene_root / cfg.image_folder
        self.colmap_dir = self.scene_root / cfg.run_name
        self.colmap_gt_dir = self.scene_root / "colmap_gt"
        self.priors_root = self.scene_root / "priors"
        self.database_dir = self.scene_root / "databases"

        # priors saving paths
        self.colmapdb_path = self.database_dir / f"{cfg.viewgraph.feats_model}_{cfg.viewgraph.match_model}.db"
        self.image_pairs_path = self.priors_root / f"{cfg.matching_type}_pairs.txt"
        self.doppelgangers_pairs_path = self.priors_root / "doppelgangers_pairs.txt"
        self.retrival_pairs_path = self.priors_root / f"{cfg.viewgraph.retrieval_model}_pairs.txt"
        self.dense_match_path = self.priors_root / f"{cfg.viewgraph.match_model}.h5"
        self.image_feats_path = self.priors_root / f"{cfg.viewgraph.feats_model}_feats.h5"
        self.feats_match_path = self.priors_root / f"{cfg.viewgraph.feats_model}_{cfg.viewgraph.match_model}.h5"
        self.monodepth_path = self.priors_root / f"{cfg.viewgraph.depth_model}.h5"
        self.focals_list_path = self.priors_root / "focals.txt"

        # create dirs
        self.makedir(self.colmap_dir)
        self.makedir(self.priors_root)
        self.makedir(self.database_dir)

        # read images information
        self.image_paths = read_image_paths(self.image_dir)
        self.num_images = len(self.image_paths)
        self.image_filenames = get_grouped_filenames(self.image_paths)
        self.image_filenames_to_ids = {fname: id for id, fname in enumerate(self.image_filenames)}
        self.image_pairs = load_image_pairs(self.image_pairs_path) if self.image_pairs_path.exists() else self.create_pairs()
        # self.image_pairs = self.create_pairs()

    def makedir(self, path: Path):
        if not path.exists():
            path.mkdir(exist_ok=True, parents=True)

    def create_sequential_pairs(self):
        # assuming image_filenames are already sorted
        image_pairs = []
        for i in range(self.num_images):
            for j in range(i+1, min(i+self.cfg.viewgraph.window_size, self.num_images)):
                image_pairs.append((self.image_filenames[i], self.image_filenames[j]))
        return image_pairs

    def create_exhaustive_pairs(self):
        pairs = list(itertools.combinations(range(self.num_images), r=2))
        image_pairs = [(self.image_filenames[id1], self.image_filenames[id2]) for id1, id2 in pairs]
        return image_pairs

    def create_pairs(self):
        if self.match_type == "exhaustive":
            self.image_pairs = self.create_exhaustive_pairs()
            save_image_pairs(self.image_pairs, self.priors_root / f"exhaustive_pairs.txt")
        elif self.match_type == 'sequential':
            self.image_pairs = self.create_sequential_pairs()
            save_image_pairs(self.image_pairs, self.priors_root / f"sequential_pairs.txt")
        else:
            self.image_pairs = load_image_pairs(self.image_pairs_path)
        return self.image_pairs

    def read_colmap_model(self, model_path: Path):
        model = None
        if model_path.exists() and (len(list(model_path.glob("*"))) > 1):
            model = pycolmap.Reconstruction(str(model_path))
            for image in model.images.values():
                image.name = str(Path(image.name).name)
        return model

    def load_models(self):
        gt_model = self.read_colmap_model(self.colmap_gt_dir)
        pd_model = self.read_colmap_model(self.colmap_dir)
        if self.metric_error_type.startswith("absolute"):
            pd_model_path = self.scene_root / f"{self.run_name}_aligned"
            colmap_alignment(self.colmap_dir, self.colmap_gt_dir, pd_model_path, self.position_accuracy_gt)
            pd_model = self.read_colmap_model(pd_model_path)
        return gt_model, pd_model


class COLMAPDataset:
    def __init__(self, cfg: DGSfMConfig) -> None:
        self.cfg = cfg
        self.name = cfg.dataset_name
        self.root = Path(cfg.root)
        self.scenes = self.list_scenes()

    def list_scenes(self):
        if self.cfg.dataset_name in ["IMC2021", "TPSfM"]:
            root_scenes = sorted(list(self.root.glob("*")))
            all_scenes = []
            for root_scene in root_scenes:
                sub_scenes = sorted(list(root_scene.glob("*")))
                for sub_scene in sub_scenes:
                    scene_name = f"{root_scene.stem}/{sub_scene.stem}"
                    all_scenes.append(scene_name)
            return all_scenes
        else:
            scenes = sorted(list(self.root.glob("*")))
            scene_names = [scene.stem for scene in scenes if scene.is_dir()]
            return scene_names

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index: int) -> Scene:
        return Scene(self.scenes[index], self.cfg)
