from pathlib import Path
from omegaconf import OmegaConf
from dataclasses import dataclass, field



@dataclass
class ViewGraphConfig:
    depth_model             : str = "MoGe2"
    feats_model             : str = "RoMav2"
    match_model             : str = "RoMav2" # "RoMav1", "RoMav2", "NNMatcher"
    retrieval_model         : str = "MegaLoc"    

    # features matching parameters
    num_feats               : int = 8192       # number of features to extract per image
    feats_confidence        : float = 0.5    # confidence threshold for feature extraction

    min_num_inliers         : int = 15
    min_inlier_ratio        : float = 0.1

    # RePoseD parameters
    epipolar_error          : float = 1.0
    reprojection_error      : float = 12.0
    share_focal             : bool = True

    # Scaled-Depth Consistency parameters
    depth_error_threshold   : float = 0.5

    # exhaustive matching parameters
    disambiguation_method   : str = "doppelgangers"    # "none", "doppelgangers"
    doppelgangers_threshold : float = 0.8
    min_edge_score          : float = 0.3
    
    # sequential matching parameters
    window_size             : int = 2

    # vocab tree matching parameters


@dataclass
class MappingConfig:
    # rotation averaging
    ra_num_runs             : int = 2
    ra_num_iters            : int = 50
    max_rotation_error      : float = 10.0

    # scale averaging
    sa_num_iters            : int = 10

    # tracks creation
    min_num_views_per_track : int = 3

    # global positioning
    gp_init                 : str = "depth"  # "depth" or "random"
    gp_num_iters            : int = 50
    max_angle_error         : float = 1.0

    # bundle adjustment
    ba_num_runs             : int = 3
    ba_num_iters            : int = 100
    max_reproj_error_norm   : float = 1e-2
    min_triang_angle        : float = 1.0


@dataclass
class DGSfMConfig:
    viewgraph               : ViewGraphConfig = field(default_factory=ViewGraphConfig)
    mapping                 : MappingConfig = field(default_factory=MappingConfig)

    root                    : str = ""
    run_name                : str = ""
    image_folder            : str = "images"
    dataset_name            : str = "ETH3D"  # "ETH3D", "IMC2021", "TPSfM", "ETH3DNViews"
    matching_type           : str = "exhaustive"  # "exhaustive", "vocab_tree", "sequential"

    # Evaluation parameters
    metric_error_type       : str = "relative_auc"
    pose_error_thresholds   : list[float] = field(default_factory=lambda: [1, 3, 5])
    point_error_thresholds  : list[float] = field(default_factory=lambda: [0.01, 0.02, 0.05])
    
    device                  : str = "cuda"  # "cuda" or "cpu"
    int_dtype               : str = "uint32"  # "uint16" for sparse feats or "uint32" for dense feats
    float_dtype             : str = "float32"  # "float16" to save space or "float32"

    def __post_init__(self):
        self.run_name = f"ours_{self.viewgraph.depth_model}_{self.viewgraph.feats_model}_{self.viewgraph.match_model}"


def load_dgsfm_config(config_path: str | Path) -> DGSfMConfig:
    """Load YAML overrides and construct the final typed configuration.

    Converting the merged OmegaConf object back to ``DGSfMConfig`` is
    intentional: it invokes ``DGSfMConfig.__post_init__`` only after all YAML
    values, including nested database settings, have been applied.
    """
    schema = OmegaConf.structured(DGSfMConfig)
    yaml_config = OmegaConf.load(config_path)
    merged_config = OmegaConf.merge(schema, yaml_config)
    return OmegaConf.to_object(merged_config)
