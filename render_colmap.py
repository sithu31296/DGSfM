"""Render a COLMAP point cloud along an orbit or an editable camera path.

Requires numpy, scipy, pycolmap, open3d, and FFmpeg for video encoding.
DGSfM is not imported.
Run ``python render_colmap.py --help`` for options.
"""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def software_vulkan_driver():
    """Locate Mesa's CPU Vulkan driver, preferring the active Python environment."""
    for directory in (Path(sys.prefix) / "share/vulkan/icd.d",
                      Path("/usr/local/share/vulkan/icd.d"),
                      Path("/usr/share/vulkan/icd.d")):
        for manifest in sorted(directory.glob("lvp_icd*.json")):
            if "i686" not in manifest.name:
                return manifest
    raise SystemExit("--software-rendering requires Mesa's lavapipe Vulkan driver (lvp_icd*.json) in this environment or system installation.")


@contextmanager
def rendering_environment(headless=False, enabled=True, software=False):
    """Give Mesa a writable runtime directory before Open3D initializes it."""
    old_runtime = os.environ.get("XDG_RUNTIME_DIR")
    old_egl = os.environ.get("EGL_PLATFORM")
    software_keys = ("VK_DRIVER_FILES", "VK_ICD_FILENAMES", "LIBGL_ALWAYS_SOFTWARE")
    old_software = {key: os.environ.get(key) for key in software_keys}
    headless = headless or software
    temporary = None
    try:
        if enabled and software:
            if not sys.platform.startswith("linux"):
                raise SystemExit("--software-rendering currently supports Linux/Mesa only.")
            driver = software_vulkan_driver()
            os.environ["VK_DRIVER_FILES"] = str(driver)
            os.environ["VK_ICD_FILENAMES"] = str(driver)
            os.environ["LIBGL_ALWAYS_SOFTWARE"] = "true"
            print(f"Using Mesa CPU rendering: {driver}", flush=True)
        if enabled and sys.platform.startswith("linux"):
            usable = False
            if old_runtime and Path(old_runtime).is_absolute():
                try:
                    # os.access alone does not detect read-only sandbox mounts.
                    with tempfile.TemporaryFile(dir=old_runtime):
                        pass
                    usable = True
                except OSError:
                    pass
            if not usable:
                temporary = tempfile.TemporaryDirectory(prefix="dgsfm-render-runtime-")
                os.environ["XDG_RUNTIME_DIR"] = temporary.name
                print(f"Using private graphics runtime directory: {temporary.name}", flush=True)
        if enabled and headless:
            os.environ["EGL_PLATFORM"] = "surfaceless"
        yield
    finally:
        if enabled and software:
            for key, value in old_software.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        if temporary is not None:
            if old_runtime is None:
                os.environ.pop("XDG_RUNTIME_DIR", None)
            else:
                os.environ["XDG_RUNTIME_DIR"] = old_runtime
            temporary.cleanup()
        if enabled and headless:
            if old_egl is None:
                os.environ.pop("EGL_PLATFORM", None)
            else:
                os.environ["EGL_PLATFORM"] = old_egl


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("Camera vectors must contain three finite numbers.")
    length = np.linalg.norm(vector)
    if length < 1e-10:
        raise ValueError("Camera direction/up vectors must be nonzero.")
    return vector / length


def look_rotation(eye, target, up):
    """Camera-to-world rotation, using COLMAP's right/down/forward axes."""
    forward = unit(np.asarray(target) - np.asarray(eye))
    right = unit(np.cross(forward, unit(up)))
    return np.column_stack((right, np.cross(forward, right), forward))


def interpolate_path(eyes, rotations, num_frames):
    """Linear position interpolation and spherical rotation interpolation."""
    eyes = np.asarray(eyes, dtype=float)
    rotations = np.asarray(rotations, dtype=float)
    if not len(eyes):
        raise ValueError("The camera path contains no keyframes.")
    if len(eyes) == 1:
        return np.repeat(eyes, num_frames, axis=0), np.repeat(rotations, num_frames, axis=0)
    times = np.arange(len(eyes))
    samples = np.linspace(0, len(eyes) - 1, num_frames)
    positions = np.column_stack([np.interp(samples, times, eyes[:, i]) for i in range(3)])
    orientations = Slerp(times, Rotation.from_matrix(rotations))(samples).as_matrix()
    return positions, orientations


def read_keyframes(path, default_up):
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or not isinstance(data.get("keyframes"), list):
        raise ValueError('Trajectory JSON must contain a "keyframes" list.')
    eyes, rotations = [], []
    for index, key in enumerate(data["keyframes"]):
        if not isinstance(key, dict) or "eye" not in key or "target" not in key:
            raise ValueError(f"Keyframe {index} needs eye and target vectors.")
        eye = np.asarray(key["eye"], dtype=float)
        target = np.asarray(key["target"], dtype=float)
        if eye.shape != (3,) or target.shape != (3,):
            raise ValueError(f"Keyframe {index}: eye and target must have three coordinates.")
        rotations.append(look_rotation(eye, target, key.get("up", default_up)))
        eyes.append(eye)
    return eyes, rotations


def camera_pose(image):
    # Older pycolmap versions expose cam_from_world as a property.
    pose = image.cam_from_world
    pose = pose() if callable(pose) else pose
    return np.asarray(image.projection_center()), np.asarray(pose.rotation.matrix()).T


def camera_frustums(reconstruction, size):
    """Batch wireframe pyramids using pinhole intrinsics and world camera poses.

    Size is the distance along camera +Z to the image plane, in model units.
    Lens distortion is omitted from these camera glyphs.
    """
    vertices, lines = [], []
    edges = np.array([[0, 1], [0, 2], [0, 3], [0, 4],
                      [1, 2], [2, 3], [3, 4], [4, 1]], dtype=np.int32)
    for image_id in sorted(reconstruction.reg_image_ids()):
        image = reconstruction.images[image_id]
        camera = reconstruction.cameras[image.camera_id]
        intrinsic = camera.calibration_matrix()
        corners = np.array([[0, 0, 1], [camera.width, 0, 1],
                            [camera.width, camera.height, 1], [0, camera.height, 1]], dtype=float)
        rays = np.linalg.solve(intrinsic, corners.T).T
        local = np.vstack([np.zeros(3), size * rays])
        eye, rotation = camera_pose(image)
        vertices.append(local @ rotation.T + eye)
        lines.append(edges + 5 * (len(vertices) - 1))
    if not vertices:
        return np.empty((0, 3)), np.empty((0, 2), dtype=np.int32)
    return np.concatenate(vertices), np.concatenate(lines)


def registered_poses(reconstruction, names_path=None):
    images = [reconstruction.images[i] for i in reconstruction.reg_image_ids()]
    images.sort(key=lambda image: image.name)
    if names_path:
        by_name = {image.name: image for image in images}
        names = [line.strip() for line in names_path.read_text().splitlines() if line.strip()]
        missing = [name for name in names if name not in by_name]
        if missing:
            raise ValueError(f"Image names are not registered in this model: {missing[:5]}")
        images = [by_name[name] for name in names]
    eyes, rotations = [], []
    for image in images:
        eye, rotation = camera_pose(image)
        eyes.append(eye)
        rotations.append(rotation)
    return eyes, rotations


def scene_bounds(points):
    if not len(points):
        raise ValueError("The model has no finite 3D points to render.")
    # Robust framing: a handful of distant SfM outliers should not shrink the scene.
    low, high = np.percentile(points, [2, 98], axis=0)
    center = (low + high) / 2
    radius = np.linalg.norm(high - low) / 2
    if radius < 1e-10:
        raise ValueError("The point cloud has no spatial extent.")
    return center, radius


def turntable_path(center, up, radius, elevation, start_angle, frames):
    up = unit(up)
    axis = np.eye(3)[np.argmin(np.abs(up))]
    right = unit(np.cross(up, axis))
    front = np.cross(right, up)
    angles = np.deg2rad(start_angle) + np.linspace(0, 2 * np.pi, frames, endpoint=False)
    elevation = np.deg2rad(elevation)
    eyes = np.asarray(center) + radius * (
        np.cos(elevation) * (np.cos(angles)[:, None] * front + np.sin(angles)[:, None] * right)
        + np.sin(elevation) * up
    )
    rotations = np.array([look_rotation(eye, center, up) for eye in eyes])
    return eyes, rotations


def save_trajectory(path, eyes, rotations, args):
    data = {
        "width": args.width,
        "height": args.height,
        "fov": args.fov,
        "keyframes": [
            {"eye": eye.tolist(), "target": (eye + rotation[:, 2]).tolist(),
             "up": (-rotation[:, 1]).tolist()}
            for eye, rotation in zip(eyes, rotations)
        ],
    }
    path.write_text(json.dumps(data, indent=2) + "\n")


def render_frames(points, colors, eyes, rotations, args, o3d, frustums=None):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    renderer = o3d.visualization.rendering.OffscreenRenderer(args.width, args.height)
    # COLMAP/PLY colors are already display RGB. Filament's post-processing
    # applies another color transform, lifting midtones and shifting hues even
    # with an unlit material. Pass the stored colors through unchanged instead.
    renderer.scene.view.set_post_processing(False)
    renderer.scene.set_background(np.array([*args.background, 1.0], dtype=np.float32))
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.sRGB_color = False  # Do not linearize display RGB with post-processing disabled.
    material.point_size = args.point_size
    renderer.scene.add_geometry("reconstruction", cloud, material)
    low, high = points.min(axis=0), points.max(axis=0)
    if frustums is not None and len(frustums[0]):
        vertices, edges = frustums
        cameras = o3d.geometry.LineSet()
        cameras.points = o3d.utility.Vector3dVector(vertices)
        cameras.lines = o3d.utility.Vector2iVector(edges)
        cameras.paint_uniform_color(args.camera_color)
        camera_material = o3d.visualization.rendering.MaterialRecord()
        camera_material.shader = "unlitLine"
        camera_material.line_width = args.camera_line_width
        renderer.scene.add_geometry("cameras", cameras, camera_material)
        low = np.minimum(low, vertices.min(axis=0))
        high = np.maximum(high, vertices.max(axis=0))
    # Explicit clipping avoids the renderer clipping a moving camera path based
    # on one static bounding box. Distances are in the reconstruction's units.
    scale = np.linalg.norm(high - low)
    near = args.near if args.near is not None else max(scale * 1e-5, 1e-7)
    for i, (eye, rotation) in enumerate(zip(eyes, rotations)):
        far = max(np.linalg.norm(low - eye), np.linalg.norm(high - eye)) + scale
        if near >= far:
            raise ValueError("--near must be smaller than the scene's far clipping distance.")
        renderer.setup_camera(args.fov, eye + rotation[:, 2], eye, -rotation[:, 1], near, far)
        output = args.output / f"{i:06d}.png"
        if not o3d.io.write_image(str(output), renderer.render_to_image()):
            raise RuntimeError(f"Failed to write {output}")
        if i == 0 or (i + 1) % 25 == 0 or i == len(eyes) - 1:
            print(f"Rendered {i + 1}/{len(eyes)} frames", flush=True)


def encode_video(output, fps, frames, ffmpeg):
    """Keep the PNGs and encode an H.264 MP4, padding odd dimensions as needed."""
    video = output.resolve() / "render.mp4"
    if video.exists():
        raise RuntimeError(f"Video already exists: {video}; choose another --output.")
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
         "-framerate", str(fps), "-start_number", "0",
         "-i", str(output.resolve() / "%06d.png"), "-frames:v", str(frames),
         "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264",
         "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)],
        capture_output=True, text=True,
    )
    if result.returncode:
        raise RuntimeError(f"FFmpeg video encoding failed; PNG frames remain in {output}.\n{result.stderr.strip()}")
    print(f"Saved video: {video} ({fps:g} fps)", flush=True)
    return video


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("model", type=Path, help="COLMAP model directory containing cameras/images/points3D .bin or .txt")
    parser.add_argument("--output", type=Path, default=Path("renders"), help="Directory for numbered PNGs, trajectory.json, and render.mp4")
    parser.add_argument("--trajectory", choices=["turntable", "flythrough", "manual"], default="turntable")
    parser.add_argument("--keyframes", type=Path, help="Manual trajectory JSON containing eye/target/up keyframes")
    parser.add_argument("--image-list", type=Path, help="Fly-through image names in desired order, one per line; default: lexical name order")
    parser.add_argument("--point-cloud", type=Path, help="Render a dense PLY cloud in the model's coordinate system instead of sparse points")
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--fps", type=float, default=30, help="Output video frame rate")
    parser.add_argument("--no-video", action="store_true", help="Save PNGs only, without running FFmpeg")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov", type=float, default=60, help="Vertical field of view in degrees; all paths use this virtual pinhole camera")
    parser.add_argument("--point-size", type=float, default=2, help="Point diameter in pixels")
    parser.add_argument("--show-cameras", action="store_true", help="Draw wireframe cones for all registered cameras")
    parser.add_argument("--camera-size", type=float, help="Camera cone depth in model units; default: 5%% of the robust scene radius")
    parser.add_argument("--camera-color", nargs=3, type=float, default=[1.0, 0.0, 0.0], metavar=("R", "G", "B"), help="Camera cone RGB color in [0, 1]")
    parser.add_argument("--camera-line-width", type=float, default=4, help="Camera cone line width in pixels")
    parser.add_argument("--background", nargs=3, type=float, default=[0.05, 0.05, 0.05], metavar=("R", "G", "B"))
    parser.add_argument("--up", nargs=3, type=float, help="World up; default: average registered camera up, or +Z")
    parser.add_argument("--center", nargs=3, type=float, help="Turntable look-at point; default: robust cloud center")
    parser.add_argument("--radius", type=float, help="Turntable eye-to-center distance; default: fit scene to field of view")
    parser.add_argument("--elevation", type=float, default=20, help="Turntable elevation in degrees")
    parser.add_argument("--start-angle", type=float, default=0, help="Turntable starting azimuth in degrees")
    parser.add_argument("--near", type=float, help="Near clipping distance in model units")
    parser.add_argument("--headless", action="store_true", help="Set EGL_PLATFORM=surfaceless before importing Open3D (Linux/Mesa)")
    parser.add_argument("--software-rendering", action="store_true", help="Explicitly select Mesa CPU rendering for Vulkan/OpenGL on Linux; implies --headless")
    parser.add_argument("--dry-run", action="store_true", help="Export trajectory.json without initializing the renderer")
    args = parser.parse_args(argv)
    if min(args.frames, args.width, args.height) < 1:
        parser.error("--frames, --width, and --height must be positive")
    if not 0 < args.fov < 180 or not -89 < args.elevation < 89:
        parser.error("--fov must be in (0, 180) and --elevation in (-89, 89)")
    for name in ("point_size", "radius", "near", "fps", "camera_size", "camera_line_width"):
        value = getattr(args, name)
        if value is not None and (not np.isfinite(value) or value <= 0):
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    for name in ("up", "center", "background", "start_angle", "camera_color"):
        value = getattr(args, name)
        if value is not None and not np.isfinite(value).all():
            parser.error(f"--{name.replace('_', '-')} must be finite")
    if any(c < 0 or c > 1 for c in args.background):
        parser.error("--background values must be in [0, 1]")
    if any(c < 0 or c > 1 for c in args.camera_color):
        parser.error("--camera-color values must be in [0, 1]")
    if (args.trajectory == "manual") != (args.keyframes is not None):
        parser.error("--trajectory manual requires --keyframes (and vice versa)")
    if args.image_list and args.trajectory != "flythrough":
        parser.error("--image-list requires --trajectory flythrough")
    return args


def run(args):
    try:
        ffmpeg = None
        if not args.dry_run and not args.no_video:
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg is None:
                raise RuntimeError("FFmpeg was not found on PATH. Install FFmpeg (with libx264) or use --no-video for PNGs only.")
        import pycolmap

        reconstruction = pycolmap.Reconstruction(str(args.model))
        camera_eyes, camera_rotations = registered_poses(reconstruction)
        o3d = None
        if args.point_cloud or not args.dry_run:
            import open3d as o3d
            print(f"Open3D {o3d.__version__}; Python: {sys.executable}", flush=True)

        if args.point_cloud:
            cloud = o3d.io.read_point_cloud(str(args.point_cloud))
            points = np.asarray(cloud.points)
            colors = np.asarray(cloud.colors) if cloud.has_colors() else np.full_like(points, 0.7)
        else:
            points = np.array([p.xyz for p in reconstruction.points3D.values()]).reshape(-1, 3)
            colors = np.array([p.color for p in reconstruction.points3D.values()]).reshape(-1, 3) / 255.0
        valid = np.isfinite(points).all(axis=1) & np.isfinite(colors).all(axis=1)
        points, colors = points[valid], colors[valid]
        center, scene_radius = scene_bounds(points)
        frustums = None
        if args.show_cameras:
            frustums = camera_frustums(reconstruction, args.camera_size or 0.05 * scene_radius)
            print(f"Showing {len(frustums[0]) // 5} reconstructed cameras", flush=True)
            if len(frustums[0]):
                # Include cameras in automatic orbit framing, even outside the cloud.
                low, high = np.percentile(points, [2, 98], axis=0)
                low = np.minimum(low, frustums[0].min(axis=0))
                high = np.maximum(high, frustums[0].max(axis=0))
                center, scene_radius = (low + high) / 2, np.linalg.norm(high - low) / 2
        if args.up is not None:
            up = unit(args.up)
        elif camera_rotations:
            average_up = -np.mean(np.array(camera_rotations)[:, :, 1], axis=0)
            up = unit(average_up) if np.linalg.norm(average_up) > 1e-6 else np.array([0., 0., 1.])
        else:
            up = np.array([0., 0., 1.])

        if args.trajectory == "turntable":
            target = center if args.center is None else np.asarray(args.center)
            half_fov = np.deg2rad(args.fov) / 2
            half_fov = min(half_fov, np.arctan(np.tan(half_fov) * args.width / args.height))
            radius = args.radius or (1.15 * (scene_radius + np.linalg.norm(target - center)) / np.sin(half_fov))
            eyes, rotations = turntable_path(target, up, radius, args.elevation, args.start_angle, args.frames)
        else:
            if args.trajectory == "manual":
                key_eyes, key_rotations = read_keyframes(args.keyframes, up)
            elif args.image_list:
                key_eyes, key_rotations = registered_poses(reconstruction, args.image_list)
            else:
                key_eyes, key_rotations = camera_eyes, camera_rotations
            eyes, rotations = interpolate_path(key_eyes, key_rotations, args.frames)

        # Refuse to mix sequences or overwrite an earlier render.
        if args.output.exists() and (any(args.output.glob("*.png")) or (args.output / "trajectory.json").exists()
                                     or (args.output / "render.mp4").exists()):
            raise ValueError(f"{args.output} already contains a render/path; choose another --output.")
        args.output.mkdir(parents=True, exist_ok=True)
        save_trajectory(args.output / "trajectory.json", eyes, rotations, args)
        print(f"Loaded {len(points):,} points; saved {len(eyes)} camera poses to {args.output / 'trajectory.json'}")
        if not args.dry_run:
            render_frames(points, colors, eyes, rotations, args, o3d, frustums)
            if ffmpeg is not None:
                encode_video(args.output, args.fps, len(eyes), ffmpeg)
    except ImportError as exc:
        raise SystemExit(f"Missing rendering dependency: {exc}. Install: pip install numpy scipy pycolmap 'open3d>=0.18'") from exc
    except (ValueError, RuntimeError, OSError) as exc:
        raise SystemExit(str(exc)) from exc


def main(argv=None):
    args = parse_args(argv)
    with rendering_environment(args.headless, enabled=not args.dry_run, software=args.software_rendering):
        run(args)


if __name__ == "__main__":
    main()
