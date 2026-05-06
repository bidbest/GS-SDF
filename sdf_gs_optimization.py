#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import struct
import subprocess
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


THIS_FILE = Path(__file__).resolve()
MODULE_ROOT = THIS_FILE.parent

DEFAULT_CONFIG_TEMPLATE = (
    MODULE_ROOT / "config" / "colmap" / "foundation_stereo_depth.yaml"
)
DEFAULT_BASE_CONFIG = MODULE_ROOT / "config" / "base.yaml"
DEFAULT_EVAL_DIR = MODULE_ROOT / "eval"
DEFAULT_DATASET_DIRNAME = "gs_sdf_colmap"
DEFAULT_RUNTIME_CONFIG_NAME = "foundation_stereo_depth_runtime.yaml"
DEFAULT_TRAIN_BINARY = Path("/opt/gs-sdf-build/build/neural_mapping_node")


@dataclass(frozen=True)
class ColmapImageEntry:
    image_id: int
    qw: str
    qx: str
    qy: str
    qz: str
    tx: str
    ty: str
    tz: str
    camera_id: str
    image_name: str


@dataclass(frozen=True)
class ColmapCamera:
    camera_id: int
    model_name: str
    gs_model: int
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    d0: float = 0.0
    d1: float = 0.0
    d2: float = 0.0
    d3: float = 0.0
    d4: float = 0.0


@dataclass(frozen=True)
class RuntimeCameraConfig:
    camera: ColmapCamera
    scale: float
    target_width: int
    target_height: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a GS-SDF COLMAP image-depth dataset from "
            "source_folder/undistorted and optionally launch training."
        )
    )
    parser.add_argument("source_folder", help="Source folder containing undistorted/")
    parser.add_argument(
        "--dataset-dirname",
        default=DEFAULT_DATASET_DIRNAME,
        help=f"Prepared dataset directory name under source_folder (default: {DEFAULT_DATASET_DIRNAME})",
    )
    parser.add_argument(
        "--config-template",
        default=str(DEFAULT_CONFIG_TEMPLATE),
        help=f"GS-SDF config template path (default: {DEFAULT_CONFIG_TEMPLATE})",
    )
    parser.add_argument(
        "--runtime-config-name",
        default=DEFAULT_RUNTIME_CONFIG_NAME,
        help=f"Generated config filename under the dataset root (default: {DEFAULT_RUNTIME_CONFIG_NAME})",
    )
    parser.add_argument(
        "--train-binary",
        default=str(DEFAULT_TRAIN_BINARY),
        help=f"GS-SDF train binary path inside the container (default: {DEFAULT_TRAIN_BINARY})",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare and validate the dataset but do not launch GS-SDF training",
    )
    return parser.parse_args()


def require_dir(path: Path, description: str) -> Path:
    if not path.is_dir():
        raise RuntimeError(f"Missing {description}: {path}")
    return path


def preview_paths(paths: list[Path], limit: int = 5) -> str:
    shown = [str(path) for path in paths[:limit]]
    suffix = "" if len(paths) <= limit else f" ... (+{len(paths) - limit} more)"
    return ", ".join(shown) + suffix


def has_colmap_binary_model(path: Path) -> bool:
    return (path / "cameras.bin").is_file() and (path / "images.bin").is_file()


def has_colmap_text_model(path: Path) -> bool:
    return (path / "cameras.txt").is_file() and (path / "images.txt").is_file()


def resolve_sparse_model_dir(sparse_root: Path) -> Path:
    if has_colmap_binary_model(sparse_root) or has_colmap_text_model(sparse_root):
        return sparse_root

    candidates = []
    for child in sorted(sparse_root.iterdir()):
        if not child.is_dir():
            continue
        if has_colmap_binary_model(child) or has_colmap_text_model(child):
            candidates.append(child)

    if not candidates:
        raise RuntimeError(
            "Could not find a COLMAP sparse model under "
            f"{sparse_root}. Expected cameras/images in sparse/ or sparse/0/."
        )

    zero_dir = sparse_root / "0"
    if zero_dir in candidates:
        return zero_dir
    if len(candidates) == 1:
        return candidates[0]

    raise RuntimeError(
        "Found multiple sparse model directories and could not choose one: "
        f"{preview_paths(candidates)}"
    )


def remove_if_exists(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def run_command(cmd: list[str]) -> None:
    print(f"[run] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def run_command_with_progress(cmd: list[str], label: str, interval_seconds: float = 10.0) -> None:
    print(f"[run] {label}: {' '.join(cmd)}", flush=True)
    start_time = time.monotonic()
    last_log_time = start_time
    process = subprocess.Popen(cmd)
    while True:
        return_code = process.poll()
        if return_code is not None:
            elapsed = time.monotonic() - start_time
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, cmd)
            print(f"[done] {label} finished in {elapsed:.1f}s", flush=True)
            return

        current_time = time.monotonic()
        if current_time - last_log_time >= interval_seconds:
            elapsed = current_time - start_time
            print(f"[progress] {label} still running after {elapsed:.1f}s", flush=True)
            last_log_time = current_time
        time.sleep(1.0)


def colmap_text_conversion_is_complete(dataset_root: Path) -> bool:
    required_files = ("cameras.txt", "images.txt", "points3D.txt")
    return all((dataset_root / filename).is_file() for filename in required_files)


def convert_sparse_model_to_text(source_model_dir: Path, dataset_root: Path) -> None:
    if colmap_text_conversion_is_complete(dataset_root):
        print(
            f"[skip] COLMAP text model already exists at {dataset_root}. "
            "Skipping binary-to-text conversion.",
            flush=True,
        )
        return

    for filename in ("cameras.txt", "images.txt", "points3D.txt"):
        remove_if_exists(dataset_root / filename)

    if has_colmap_binary_model(source_model_dir):
        run_command_with_progress(
            [
                "colmap",
                "model_converter",
                "--input_path",
                str(source_model_dir),
                "--output_path",
                str(dataset_root),
                "--output_type",
                "TXT",
            ],
            label=f"COLMAP model conversion to text ({source_model_dir} -> {dataset_root})",
        )
        return

    if has_colmap_text_model(source_model_dir):
        print(
            f"[step] Copying existing COLMAP text model from {source_model_dir} to {dataset_root}",
            flush=True,
        )
        for filename in ("cameras.txt", "images.txt", "points3D.txt"):
            source_path = source_model_dir / filename
            if source_path.is_file():
                shutil.copy2(source_path, dataset_root / filename)
        print(f"[done] Copied COLMAP text model into {dataset_root}", flush=True)
        return

    raise RuntimeError(f"COLMAP model directory is missing cameras/images data: {source_model_dir}")


def parse_colmap_images_txt(images_txt_path: Path) -> list[ColmapImageEntry]:
    entries: list[ColmapImageEntry] = []
    with images_txt_path.open("r", encoding="utf-8") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            parts = stripped.split()
            if len(parts) < 10:
                raise RuntimeError(f"Malformed COLMAP image line in {images_txt_path}: {stripped}")

            entries.append(
                ColmapImageEntry(
                    image_id=int(parts[0]),
                    qw=parts[1],
                    qx=parts[2],
                    qy=parts[3],
                    qz=parts[4],
                    tx=parts[5],
                    ty=parts[6],
                    tz=parts[7],
                    camera_id=parts[8],
                    image_name=parts[9],
                )
            )

            handle.readline()

    if not entries:
        raise RuntimeError(f"No COLMAP image entries found in {images_txt_path}")
    return entries


def parse_first_camera(cameras_txt_path: Path) -> ColmapCamera:
    with cameras_txt_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 5:
                raise RuntimeError(f"Malformed COLMAP camera line in {cameras_txt_path}: {stripped}")

            camera_id = int(parts[0])
            model_name = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(value) for value in parts[4:]]

            if model_name == "PINHOLE":
                if len(params) != 4:
                    raise RuntimeError(f"Expected 4 PINHOLE params in {cameras_txt_path}: {stripped}")
                fx, fy, cx, cy = params
                return ColmapCamera(camera_id, model_name, 0, width, height, fx, fy, cx, cy)

            if model_name == "OPENCV":
                if len(params) < 4:
                    raise RuntimeError(f"Expected OPENCV params in {cameras_txt_path}: {stripped}")
                fx, fy, cx, cy = params[:4]
                return ColmapCamera(camera_id, model_name, 0, width, height, fx, fy, cx, cy)

            if model_name == "OPENCV_FISHEYE":
                if len(params) != 8:
                    raise RuntimeError(
                        f"Expected 8 OPENCV_FISHEYE params in {cameras_txt_path}: {stripped}"
                    )
                fx, fy, cx, cy, d0, d1, d2, d3 = params
                return ColmapCamera(
                    camera_id,
                    model_name,
                    1,
                    width,
                    height,
                    fx,
                    fy,
                    cx,
                    cy,
                    d0,
                    d1,
                    d2,
                    d3,
                    0.0,
                )

            raise RuntimeError(
                "Unsupported COLMAP camera model for GS-SDF adapter: "
                f"{model_name}. Expected undistorted PINHOLE, OPENCV, or OPENCV_FISHEYE."
            )

    raise RuntimeError(f"No cameras found in {cameras_txt_path}")


def load_npy_depth_shape(depth_path: Path) -> tuple[int, int]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required to inspect FoundationStereo .npy depth files") from exc

    depth = np.load(depth_path, mmap_mode="r")
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise RuntimeError(f"Expected a 2D depth array in {depth_path}, got shape {depth.shape}")
    return int(depth.shape[0]), int(depth.shape[1])


def build_runtime_camera_config(
    camera: ColmapCamera,
    depth_source_dir: Path,
    sample_entry: ColmapImageEntry,
) -> RuntimeCameraConfig:
    sample_depth_path = depth_source_dir / build_depth_npy_filename(sample_entry)
    sample_depth_height, sample_depth_width = load_npy_depth_shape(sample_depth_path)
    scale = min(sample_depth_width / camera.width, sample_depth_height / camera.height)
    if scale <= 0:
        raise RuntimeError(
            f"Invalid runtime camera scale computed from {sample_depth_path}: {scale}"
        )

    target_width = max(1, int(camera.width * scale))
    target_height = max(1, int(camera.height * scale))
    print(
        f"[step] Runtime camera scale {scale:.6f} -> target resolution "
        f"{target_width}x{target_height}",
        flush=True,
    )
    return RuntimeCameraConfig(
        camera=camera,
        scale=scale,
        target_width=target_width,
        target_height=target_height,
    )


def foundation_stereo_output_stem(image_id: int, image_name: str) -> str:
    normalized_name = image_name.replace("\\", "/")
    stem = PurePosixPath(normalized_name).with_suffix("").as_posix().replace("/", "__")
    return f"{image_id:08d}_{stem}"


def build_depth_filename(entry: ColmapImageEntry) -> str:
    return f"{foundation_stereo_output_stem(entry.image_id, entry.image_name)}.png"


def build_depth_npy_filename(entry: ColmapImageEntry) -> str:
    return f"{foundation_stereo_output_stem(entry.image_id, entry.image_name)}.npy"


def write_depths_txt(depths_txt_path: Path, entries: list[ColmapImageEntry]) -> None:
    with depths_txt_path.open("w", encoding="utf-8") as handle:
        handle.write("# Depth list with two lines of data per depth:\n")
        handle.write("#   DEPTH_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        handle.write("#   (empty line)\n")
        handle.write(f"# Number of depths: {len(entries)}\n")
        for entry in entries:
            depth_filename = build_depth_filename(entry)
            handle.write(
                " ".join(
                    [
                        str(entry.image_id),
                        entry.qw,
                        entry.qx,
                        entry.qy,
                        entry.qz,
                        entry.tx,
                        entry.ty,
                        entry.tz,
                        entry.camera_id,
                        depth_filename,
                    ]
                )
            )
            handle.write("\n\n")


def make_png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + chunk_type
        + payload
        + struct.pack(">I", zlib.crc32(chunk_type + payload) & 0xFFFFFFFF)
    )


def write_png_u16(path: Path, depth_u16) -> None:
    height, width = depth_u16.shape
    depth_be = depth_u16.astype(">u2", copy=False)
    scanlines = b"".join(b"\x00" + depth_be[row].tobytes() for row in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 16, 0, 0, 0, 0)
    png_bytes = b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            make_png_chunk(b"IHDR", ihdr),
            make_png_chunk(b"IDAT", zlib.compress(scanlines)),
            make_png_chunk(b"IEND", b""),
        ]
    )
    path.write_bytes(png_bytes)


def resize_depth_nearest(depth, target_height: int, target_width: int):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required to resize FoundationStereo depth files") from exc

    if depth.shape == (target_height, target_width):
        return depth

    row_idx = np.clip(
        np.round(np.linspace(0, depth.shape[0] - 1, target_height)).astype(np.int64),
        0,
        depth.shape[0] - 1,
    )
    col_idx = np.clip(
        np.round(np.linspace(0, depth.shape[1] - 1, target_width)).astype(np.int64),
        0,
        depth.shape[1] - 1,
    )
    return depth[row_idx[:, None], col_idx[None, :]]


def convert_npy_depth_to_u16_png(
    source_path: Path,
    output_path: Path,
    target_height: int,
    target_width: int,
) -> None:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required to convert FoundationStereo .npy depth files") from exc

    depth = np.load(source_path, mmap_mode="r")
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise RuntimeError(f"Expected a 2D depth array in {source_path}, got shape {depth.shape}")

    depth = resize_depth_nearest(depth, target_height, target_width)
    depth_mm = np.clip(np.nan_to_num(depth, nan=0.0) * 1000.0, 0.0, 65535.0).astype(np.uint16)
    write_png_u16(output_path, depth_mm)


def populate_depth_pngs(
    dataset_depth_dir: Path,
    depth_source_dir: Path,
    entries: list[ColmapImageEntry],
    target_height: int,
    target_width: int,
) -> None:
    remove_if_exists(dataset_depth_dir)
    dataset_depth_dir.mkdir(parents=True, exist_ok=True)
    total = len(entries)
    print(
        f"[step] Generating {total} GS-SDF depth PNG files in {dataset_depth_dir} "
        f"at {target_width}x{target_height}",
        flush=True,
    )
    for index, entry in enumerate(entries, start=1):
        source_path = depth_source_dir / build_depth_npy_filename(entry)
        output_path = dataset_depth_dir / build_depth_filename(entry)
        convert_npy_depth_to_u16_png(source_path, output_path, target_height, target_width)
        if index == total or index % 100 == 0:
            print(f"[progress] Generated {index}/{total} depth PNG files", flush=True)
    print(f"[done] Depth PNG generation completed in {dataset_depth_dir}", flush=True)


def expected_relative_image_paths(entries: list[ColmapImageEntry]) -> list[Path]:
    return [Path(entry.image_name) for entry in entries]


def copy_registered_images(
    dataset_images_dir: Path,
    images_source_dir: Path,
    entries: list[ColmapImageEntry],
) -> None:
    expected_paths = expected_relative_image_paths(entries)
    if dataset_images_dir.is_dir() and not dataset_images_dir.is_symlink():
        if all((dataset_images_dir / rel_path).is_file() for rel_path in expected_paths):
            print(
                f"[skip] Registered images already exist under {dataset_images_dir}. "
                "Skipping image copy.",
                flush=True,
            )
            return

    remove_if_exists(dataset_images_dir)
    dataset_images_dir.mkdir(parents=True, exist_ok=True)

    total = len(expected_paths)
    print(f"[step] Copying {total} registered images into {dataset_images_dir}", flush=True)
    for index, rel_path in enumerate(expected_paths, start=1):
        source_path = images_source_dir / rel_path
        target_path = dataset_images_dir / rel_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        if index == total or index % 100 == 0:
            print(f"[progress] Copied {index}/{total} images", flush=True)
    print(f"[done] Image copy completed into {dataset_images_dir}", flush=True)


def validate_registered_files(
    images_source_dir: Path, depth_source_dir: Path, entries: list[ColmapImageEntry]
) -> None:
    missing_images: list[Path] = []
    missing_depths: list[Path] = []
    for entry in entries:
        image_path = images_source_dir / entry.image_name
        if not image_path.is_file():
            missing_images.append(image_path)
        depth_path = depth_source_dir / build_depth_npy_filename(entry)
        if not depth_path.is_file():
            missing_depths.append(depth_path)

    if missing_images:
        raise RuntimeError(
            "Missing registered color images for COLMAP entries: " f"{preview_paths(missing_images)}"
        )
    if missing_depths:
        raise RuntimeError(
            "Missing FoundationStereo depth NPYs for COLMAP entries: "
            f"{preview_paths(missing_depths)}"
        )


def validate_depth_png_headers(depth_png_dir: Path, entries: list[ColmapImageEntry]) -> None:
    sample_count = min(3, len(entries))
    for entry in entries[:sample_count]:
        depth_path = depth_png_dir / build_depth_filename(entry)
        with depth_path.open("rb") as handle:
            header = handle.read(29)
        if len(header) < 29 or header[:8] != b"\x89PNG\r\n\x1a\n":
            raise RuntimeError(f"Depth file is not a PNG: {depth_path}")
        if struct.unpack(">I", header[8:12])[0] != 13 or header[12:16] != b"IHDR":
            raise RuntimeError(f"Depth PNG is missing a valid IHDR chunk: {depth_path}")

        width, height = struct.unpack(">II", header[16:24])
        bit_depth = header[24]
        color_type = header[25]
        if bit_depth != 16:
            raise RuntimeError(f"Depth PNG is not 16-bit: {depth_path} (bit_depth={bit_depth})")
        if color_type != 0:
            raise RuntimeError(
                f"Depth PNG is not grayscale as expected: {depth_path} (color_type={color_type})"
            )
        print(f"[check] {depth_path.name}: {width}x{height}, 16-bit grayscale PNG", flush=True)


def render_runtime_config(
    template_path: Path,
    runtime_config_path: Path,
    runtime_camera: RuntimeCameraConfig,
) -> None:
    if not template_path.is_file():
        raise RuntimeError(f"Missing config template: {template_path}")

    template_text = template_path.read_text(encoding="utf-8")
    replacements = {
        "scale: 1.0": f"scale: {runtime_camera.scale!r}",
        "__PACKAGE_PATH__": MODULE_ROOT.as_posix(),
        "__CAMERA_MODEL__": str(runtime_camera.camera.gs_model),
        "__CAMERA_WIDTH__": str(runtime_camera.camera.width),
        "__CAMERA_HEIGHT__": str(runtime_camera.camera.height),
        "__CAMERA_FX__": repr(runtime_camera.camera.fx),
        "__CAMERA_FY__": repr(runtime_camera.camera.fy),
        "__CAMERA_CX__": repr(runtime_camera.camera.cx),
        "__CAMERA_CY__": repr(runtime_camera.camera.cy),
    }
    for old, new in replacements.items():
        template_text = template_text.replace(old, new)

    distortion_block = "\n".join(
        [
            "   d0: __CAMERA_D0__",
            "   d1: __CAMERA_D1__",
            "   d2: __CAMERA_D2__",
            "   d3: __CAMERA_D3__",
            "   d4: __CAMERA_D4__",
        ]
    )
    template_text = template_text.replace(
        distortion_block,
        "   # d0-d4 intentionally omitted for undistorted adapter data",
    )

    runtime_config_path.write_text(template_text, encoding="utf-8")


def prepare_runtime_config_paths(
    dataset_root: Path,
    runtime_config_name: str,
) -> tuple[Path, Path]:
    config_dir = dataset_root / "config"
    scene_config_dir = config_dir / "scene"
    scene_config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "base.yaml", scene_config_dir / runtime_config_name


def install_dataset_base_config(base_config_source: Path, dataset_base_config_path: Path) -> None:
    if not base_config_source.is_file():
        raise RuntimeError(f"Missing GS-SDF base config template: {base_config_source}")
    dataset_base_config_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base_config_source, dataset_base_config_path)
    print(f"[done] Installed dataset base config at {dataset_base_config_path}", flush=True)


def install_dataset_eval_scripts(eval_source_dir: Path, dataset_eval_dir: Path) -> None:
    if not eval_source_dir.is_dir():
        raise RuntimeError(f"Missing GS-SDF eval directory: {eval_source_dir}")

    required_eval_script = dataset_eval_dir / "draw_loss.py"
    if required_eval_script.is_file():
        print(
            f"[skip] GS-SDF eval scripts already exist under {dataset_eval_dir}. "
            "Skipping eval script copy.",
            flush=True,
        )
        return

    remove_if_exists(dataset_eval_dir)
    shutil.copytree(eval_source_dir, dataset_eval_dir)
    print(f"[done] Installed GS-SDF eval scripts at {dataset_eval_dir}", flush=True)


def validate_preconditions(source_folder: Path) -> tuple[Path, Path, Path, Path]:
    undistorted_dir = require_dir(source_folder / "undistorted", "undistorted directory")
    images_dir = require_dir(undistorted_dir / "images", "undistorted image directory")
    sparse_root = require_dir(undistorted_dir / "sparse", "undistorted sparse directory")
    depth_dir = require_dir(undistorted_dir / "depth", "FoundationStereo depth directory")
    return undistorted_dir, images_dir, sparse_root, depth_dir


def prepare_dataset(source_folder: Path, args: argparse.Namespace) -> tuple[Path, Path]:
    _, images_dir, sparse_root, depth_dir = validate_preconditions(source_folder)
    sparse_model_dir = resolve_sparse_model_dir(sparse_root)

    dataset_root = source_folder / args.dataset_dirname
    dataset_root.mkdir(parents=True, exist_ok=True)
    print(f"[step] Preparing GS-SDF dataset at {dataset_root}", flush=True)
    print(f"[step] Using COLMAP sparse model from {sparse_model_dir}", flush=True)

    convert_sparse_model_to_text(sparse_model_dir, dataset_root)

    images_txt_path = dataset_root / "images.txt"
    cameras_txt_path = dataset_root / "cameras.txt"
    entries = parse_colmap_images_txt(images_txt_path)
    camera = parse_first_camera(cameras_txt_path)
    runtime_camera = build_runtime_camera_config(camera, depth_dir, entries[0])

    validate_registered_files(images_dir, depth_dir, entries)

    copy_registered_images(dataset_root / "images", images_dir, entries)
    populate_depth_pngs(
        dataset_root / "depths",
        depth_dir,
        entries,
        runtime_camera.target_height,
        runtime_camera.target_width,
    )
    validate_depth_png_headers(dataset_root / "depths", entries)

    write_depths_txt(dataset_root / "depths.txt", entries)

    dataset_base_config_path, runtime_config_path = prepare_runtime_config_paths(
        dataset_root, args.runtime_config_name
    )
    install_dataset_base_config(DEFAULT_BASE_CONFIG, dataset_base_config_path)
    install_dataset_eval_scripts(DEFAULT_EVAL_DIR, dataset_root / "eval")
    render_runtime_config(
        Path(args.config_template).expanduser().resolve(),
        runtime_config_path,
        runtime_camera,
    )
    runtime_config_text = runtime_config_path.read_text(encoding="utf-8")
    runtime_config_text = runtime_config_text.replace(
        'base_config: "../base.yaml"',
        f'base_config: "{dataset_base_config_path.as_posix()}"',
    )
    runtime_config_path.write_text(runtime_config_text, encoding="utf-8")

    print(f"[done] Prepared GS-SDF dataset: {dataset_root}", flush=True)
    print(f"[done] Color poses: {len(entries)}", flush=True)
    print(f"[done] Depth poses: {len(entries)}", flush=True)
    print(f"[done] Runtime config: {runtime_config_path}", flush=True)
    return dataset_root, runtime_config_path


def run_training(dataset_root: Path, runtime_config_path: Path, train_binary: Path) -> None:
    if not train_binary.is_file():
        raise RuntimeError(f"Missing GS-SDF train binary: {train_binary}")
    print(f"[step] Launching GS-SDF training on {dataset_root}", flush=True)
    run_command([str(train_binary), "train", str(runtime_config_path), str(dataset_root)])


def main() -> int:
    args = parse_args()
    source_folder = Path(args.source_folder).expanduser().resolve()
    if not source_folder.is_dir():
        raise RuntimeError(f"Source folder is not a directory: {source_folder}")

    dataset_root, runtime_config_path = prepare_dataset(source_folder, args)
    if args.prepare_only:
        print("[done] Dataset preparation completed without launching GS-SDF training", flush=True)
        return 0

    run_training(
        dataset_root=dataset_root,
        runtime_config_path=runtime_config_path,
        train_binary=Path(args.train_binary).expanduser().resolve(),
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"\n[error] Command failed with exit code {exc.returncode}", file=sys.stderr)
        raise SystemExit(exc.returncode) from exc
    except RuntimeError as exc:
        print(f"\n[error] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
