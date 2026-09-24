# DGSfM: Depth-Guided Scale-Aware Global Structure-from-Motion

DGSfM reconstructs a scene from an unordered image collection. It combines monocular depth, local or dense image correspondences, depth-aware relative pose estimation, and scaled-depth guided global optimization to produce a COLMAP-format reconstruction.

## Installation

From the repository root, install DGSfM and its dependencies with:

```bash
# create a conda environment
conda create -n dgsfm python=3.12
conda activate dgsfm
conda install nvidia::cuda-toolkit==12.8.0
pip install torch==2.8.0 torchvision==0.23.0 xformers==0.0.32.post1 --index-url https://download.pytorch.org/whl/cu128

# install additional dependencies for reposed
conda install cmake -c conda-forge --override-channels
conda install eigen -c conda-forge --override-channels

# install this project as a package
pip install -e .

# install pypose and bae
# first install cudss 0.5 for cuda 12.x and cudss 0.7.1 for cuda 13.x
conda install libcudss-dev==0.5.0.16 -c conda-forge --override-channels
pip install git+https://github.com/pypose/pypose.git
pip install git+https://github.com/pypose/bae.git --no-build-isolation
```
> Model weights are downloaded on their first use unless noted below.

## Quick start

1. Create a configuration from [`configs/custom.yaml`](configs/custom.yaml) and set `root` to your dataset directory.
2. Arrange your images as described in [Dataset layout](#dataset-layout).
3. Download the Doppelgangers++ [checkpoint](https://huggingface.co/doppelgangers25/doppelgangers_plusplus/tree/main) (save it to `third_party/checkpoints`) if you use the default disambiguation stage.
4. Run the commands in [Pipeline](#pipeline), always passing the same configuration file.

For example, after creating `configs/my_scene.yaml`:

```bash
python priors/monodepth.py -c configs/my_scene.yaml
python priors/disambiguation.py -c configs/my_scene.yaml
python priors/feats_corrs.py -c configs/my_scene.yaml
python priors/colmapdb.py -c configs/my_scene.yaml
python run_sfm_colmapdb.py -c configs/my_scene.yaml
```

This example uses the sparse LoMa configuration in `custom.yaml`, so it does not need the dense-matching command.

## Dataset layout

`root` is a directory containing one directory per scene. Each scene needs an `images/` directory and three writable working directories. The pipeline creates the reconstruction directory (in COLMAP format), `priors/` (depth, feats, matches in h5 files) and `databases/` (COLMAP database `.db` file).

```text
<root>/
└── <scene-name>/
    ├── images/                 # Input images: JPG, PNG, HEIF, or HEIC
    ├── priors/                 # Intermediate files: h5, txt
    ├── databases/              # COLMAP database file
    └── colmap_gt/              # Optional reference COLMAP model; required for evaluation
```

For a single scene named `garden`, prepare the directory with:

```bash
mkdir -p /path/to/dataset/garden/{images}
```

Then copy images to `/path/to/dataset/garden/images/` and set the following in your configuration:

```yaml
root: /path/to/dataset
dataset_name: Custom
image_folder: images
matching_type: exhaustive
```

## Configuration

Configuration files are YAML overrides of the defaults defined in [`dgsfm/database/config.py`](dgsfm/database/config.py). The provided [Custom](configs/custom.yaml), [ETH3D](configs/eth3d.yaml) and [IMC 2021](configs/imc2021.yaml) files are good starting points.

The settings most commonly changed for a new dataset are:

| Setting | Purpose | Typical values |
| --- | --- | --- |
| `root` | Dataset root containing scene directories. | Absolute path to the dataset. |
| `dataset_name` | Selects the scene-directory convention and evaluation labels. | `ETH3D`, `IMC2021`, `TPSfM`. |
| `matching_type` | Selects candidate image pairs. | `exhaustive` or `sequential`. |
| `viewgraph.depth_model` | Monocular depth estimator. | `MoGe2`, `MoGe1`, `MoGe3`, `UniDepth2`, `ZoeDepth`. |
| `viewgraph.feats_model` | Feature extractor. | `LoMaB`, `LoMaL`, `LoMaG`, or a dense matcher such as `RoMav2`. |
| `viewgraph.match_model` | Correspondence matcher. | `LoMaB`, `LoMaL`, `LoMaG`, `RoMav1`, `RoMav2`. |
| `viewgraph.num_feats` | Maximum features retained per image. | `4096` is used by the supplied configurations. |

With `matching_type: exhaustive`, all image pairs in a scene are considered. This grows quadratically with the image count. Use `sequential` and tune `viewgraph.window_size` for ordered sequences.

### Sparse and dense matching

The provided configurations use LoMa for both feature extraction and matching:

```yaml
viewgraph:
  feats_model: LoMaB
  match_model: LoMaB
```

For this sparse setup, skip `priors/dense_match.py`. To use dense RoMa matching, set both model fields to a RoMa variant, for example `RoMav2`, then include the dense-matching stage before generating features and correspondences.


## Pipeline

Run every command from the repository root and use the same `-c` configuration argument throughout. Stages write their intermediates under each scene's `priors/` and final database file excluding monodepth estimates in `databases/`.

| Step | Command | Produces |
| --- | --- | --- |
| 1. Monocular depth | `python priors/monodepth.py -c <config>` | Per-image depth, confidence, and intrinsics. |
| 2. Disambiguation | `python priors/disambiguation.py -c <config>` | Pairwise Doppelgangers++ scores. |
| 3. Dense matching | `python priors/dense_match.py -c <config>` | Dense RoMa warps and confidence maps. Required only for RoMa dense matching. |
| 4. Features and correspondences | `python priors/feats_corrs.py -c <config>` | Image features and pairwise feature matches. |
| 5. COLMAP database | `python priors/colmapdb.py -c <config>` | A COLMAP database containing images, keypoints, and matches. |
| 6. Global mapping | `python run_sfm_colmapdb.py -c <config>` | Filtered view graph, global reconstruction, and pose evaluation. |

For the default sparse configuration, run:

```bash
python priors/monodepth.py -c configs/custom.yaml
python priors/disambiguation.py -c configs/custom.yaml
python priors/feats_corrs.py -c configs/custom.yaml
python priors/colmapdb.py -c configs/custom.yaml
python run_sfm_colmapdb.py -c configs/custom.yaml
```

For a dense RoMa configuration, insert this command between disambiguation and feature generation:

```bash
python priors/dense_match.py -c configs/custom_dense.yaml
```

## Outputs

For a scene named `<scene-name>`, DGSfM writes the following files:

```text
<root>/<scene-name>/
├── priors/
│   ├── <depth-model>.h5
│   ├── doppelgangers_pairs.txt             # When disambiguation is enabled
│   ├── <matcher>.h5                        # For dense RoMa matching only
│   ├── <feature-model>_feats.h5
│   ├── <feature-model>_<matcher>.h5
│   └── focals.txt
├── databases/
│   └── <feature-model>_<matcher>.db
└── ours_<depth-model>_<feature-model>_<matcher>/
    ├── cameras.txt
    ├── images.txt
    └── points3D.txt
```

The reconstruction is exported as a text-format COLMAP model, so it can be inspected or converted with standard COLMAP tooling.

<!-- 
## Rendering a reconstruction

[`render_colmap.py`](render_colmap.py) uses [Open3D's offscreen renderer](https://www.open3d.org/docs/release/python_api/open3d.visualization.rendering.OffscreenRenderer.html) to save numbered PNG images of the colored point cloud. It reads COLMAP text or binary models using `pycolmap`. Sparse SfM models produce point-cloud views, not photorealistic novel views. For denser results, pass `--point-cloud /path/to/fused.ply`; the cloud must use the same coordinates as the reconstruction.

Point colors are treated as display RGB and rendered unlit with post-processing disabled, preserving the model's colors without tone mapping or an additional gamma transform. Antialiased edges can blend with the background; the H.264 video can differ slightly from the PNGs because of compression and chroma subsampling.

Install the renderer into your DGSfM environment with `pip install 'open3d>=0.18'` (or use `pip install -e '.[render]'`). The standalone script only needs `numpy`, `scipy`, `pycolmap`, and `open3d`.

```bash
# Automatic turntable: frame the scene and orbit around its center.
python render_colmap.py /path/to/scene/ours_MoGe2_LoMaB_LoMaB \
    --output renders/orbit --trajectory turntable --frames 180

# Fly through the registered camera poses, sorted lexically by image name.
python render_colmap.py /path/to/model \
    --output renders/flythrough --trajectory flythrough --frames 240

# Use selected cameras as keyframes in a deliberate order.
python render_colmap.py /path/to/model --trajectory flythrough \
    --image-list camera_order.txt --output renders/selected
```

`camera_order.txt` contains one registered image name per line, exactly as stored in COLMAP. Choose a spatially sensible order for unordered photo collections. Fly-throughs interpolate positions linearly and rotations with SLERP, spending equal time between each pair of keyframes; they do not avoid obstacles. All modes use a virtual pinhole camera with `--fov` (vertical degrees), `--width`, and `--height`, rather than the source images' intrinsics or distortion.

Use `--zoom 2` for a closeup or `--zoom 0.5` for a wider view. Zoom multiplies the rendering focal length while keeping the camera positions and orientations fixed, including for automatic turntables. To specify an absolute focal length, use `--focal-length 1000` (alias `--focal-length-px`) in **output-image pixels**, with `fx = fy`. Larger focal lengths zoom in; smaller ones zoom out. Pixel focal lengths depend on output resolution: at 720 pixels high, the default 60-degree vertical FOV corresponds to about 624 pixels. `--focal-length` and `--fov` are mutually exclusive; either may be combined with `--zoom`.

```bash
python render_colmap.py /path/to/model --trajectory flythrough \
    --zoom 2 --output renders/closeup --software-rendering
python render_colmap.py /path/to/model --trajectory flythrough \
    --zoom 0.5 --output renders/wide --software-rendering
python render_colmap.py /path/to/model --trajectory turntable \
    --focal-length 1000 --output renders/telephoto --software-rendering
```

For automatic turntables, `--fov` still sets the base scene framing; focal-length overrides and zoom are then applied without moving the orbit. The exported `fov` and `focal_length_px` are the effective values after zoom. When replaying a saved path, use its effective `--fov` or `--focal-length` with the default `--zoom 1` to reproduce the projection. Point diameters remain controlled by `--point-size` in pixels.

For a manual path, create `path.json` using **world coordinates in the reconstruction**:

```json
{
  "keyframes": [
    {"eye": [0, -5, 2], "target": [0, 0, 0], "up": [0, 0, 1]},
    {"eye": [5, 0, 2], "target": [0, 0, 0], "up": [0, 0, 1]},
    {"eye": [0, 5, 2], "target": [0, 0, 0], "up": [0, 0, 1]}
  ]
}
```

```bash
python render_colmap.py /path/to/model --trajectory manual \
    --keyframes path.json --output renders/manual --frames 120
```

Every run exports `trajectory.json` with the sampled eye/target/up poses, which can be edited and reused as `--keyframes`. Use `--dry-run` to generate this file without rendering first, then render into a different output directory. A single keyframe with `--frames 1` renders one view. Exported width/height/fov are informational; pass the same CLI values when replaying a path.

The orbit defaults to an up direction estimated from registered cameras. If the scene is tilted, set `--up 0 0 1` (or the appropriate world axis). Adjust `--center X Y Z`, `--radius R`, `--elevation 20`, `--start-angle 90`, `--point-size 3`, or `--background 1 1 1` as needed. Automatic framing uses the central 96% of point coordinates to reduce the influence of distant outliers; it does not remove points.

Add `--show-cameras` to include orange wireframe camera cones (frustums) in the PNGs and video:

```bash
python render_colmap.py /path/to/model --trajectory turntable \
    --show-cameras --camera-color 1 0.5 0 --camera-line-width 2 \
    --output renders/cameras --headless
```

Each cone shows a registered camera's position, orientation, and image aspect ratio using its pinhole intrinsics; lens distortion is omitted. All registered cameras are shown, including when `--image-list` selects only some cameras for the fly-through. Cone depth defaults to 5% of the robust point-cloud radius; override it with `--camera-size 0.2` in reconstruction units. Automatic turntable framing includes the cameras. Fly-through and manual camera paths remain as specified, so cones outside those views will not be visible.

On a Linux server with Mesa/EGL, add `--headless` to set `EGL_PLATFORM=surfaceless` before Open3D is imported. The renderer still needs working graphics drivers; see [Open3D's CPU rendering setup](https://www.open3d.org/docs/release/tutorial/visualization/cpu_rendering.html). Existing PNG sequences and trajectory files are protected from overwriting, so choose a fresh output directory for each run.

The script checks `XDG_RUNTIME_DIR` before rendering on Linux. If it is unset, missing, or unwritable, it creates a private temporary directory for the run. Some Mesa builds allocate graphics memory through files in this directory and can crash with `Failed to create anonymous file for memory allocations` when it is unusable. This setup works with both the EGL and Vulkan rendering backends. If an earlier attempt crashed after exporting `trajectory.json`, retry with a new `--output` directory.

If a graphics driver produces completely black images, try `--software-rendering` on Linux. This explicitly selects Mesa's CPU Vulkan driver (lavapipe) and software OpenGL before importing Open3D, and implies `--headless`. It requires an installed `lvp_icd*.json` driver manifest in the active Python environment or system Vulkan driver directory. `--headless` alone configures EGL and does not force a particular Vulkan device. Use a fresh output directory and `--frames 3 --no-video` for a quick check. The script prints its Python executable and Open3D version so runs on different environments can be compared.

After rendering, the script automatically runs FFmpeg to create `<output>/render.mp4` (H.264, 30 fps), keeping the PNGs. Install FFmpeg with `libx264` support and ensure `ffmpeg` is on `PATH`. Set `--fps 24` to change playback speed, or `--no-video` to save only images. `--dry-run` does not require FFmpeg. Odd image dimensions are padded by one pixel where needed for video encoding.

To encode an existing PNG sequence separately:

```bash
ffmpeg -framerate 30 -i renders/orbit/%06d.png \
    -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' \
    -c:v libx264 -pix_fmt yuv420p renders/orbit/render.mp4
``` -->

## Evaluation

`run_sfm_colmapdb.py` evaluates each reconstruction at the end of mapping. To enable it, place a reference COLMAP model in `<scene>/colmap_gt/`. The script reports pose AUC at the thresholds in `pose_error_thresholds`.

If your collection has no ground-truth COLMAP model, the reconstruction is still written before evaluation begins. The current entry point expects `colmap_gt/` for its final evaluation step; remove or bypass that step in [`run_sfm_colmapdb.py`](run_sfm_colmapdb.py) when reconstructing an unevaluated custom collection.


<!-- 
## Troubleshooting

- **`FileNotFoundError` for a prior or database:** run the prior stages in order and confirm that every scene contains writable `priors/` and `databases/` directories.
- **Missing Doppelgangers++ checkpoint:** confirm the exact path is `third_party/checkpoints/checkpoint-dg.pth`, or disable disambiguation in the YAML configuration.
- **Out-of-memory during matching:** lower `viewgraph.num_feats`, use `matching_type: sequential`, or use a smaller depth/matching model.
- **No CUDA device available:** DGSfM's supplied models and defaults target `cuda`; use a CUDA-enabled PyTorch environment. -->
