#!/usr/bin/env python3
"""Unified CLI for training and evaluating a traffic light detector.

This script exposes a cohesive command line interface that orchestrates the
entire workflow required for the project:

* environment checks (`setup`)
* model training (`train`)
* validation only runs (`validate`)
* light-weight hyper-parameter search (`tune`)
* exporting the best checkpoint (`export`)
* video inference with optional SAHI tiling (`infer-video`)
* a short end-to-end smoke test (`demo`)
* notebook export that mirrors the CLI workflow (`export-notebook`)

All commands share a consistent configuration surface and write their
artifacts to the ``runs/`` or ``outputs/`` folders located next to this file.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import textwrap
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, MutableMapping, Optional, Sequence, Tuple

try:  # Optional but nice-to-have dependency for pretty YAML dumps.
    import yaml  # type: ignore
except Exception:  # pragma: no cover - yaml is optional, json fallback is used.
    yaml = None  # type: ignore

# --- Constants -----------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
RUNS_DIR = REPO_ROOT / "runs"
OUTPUTS_DIR = REPO_ROOT / "outputs"
DATA_CONFIG_DIR = RUNS_DIR / "data_configs"
DEFAULT_MODEL = os.environ.get("TRAFFIC_LIGHT_MODEL", "yolov8n.pt")
CLASS_NAMES = ["green", "off", "red", "wait_on", "yellow"]
NOTEBOOK_PATH = REPO_ROOT / "Project_1_object_detection_traffic_light.ipynb"

# --- Utility helpers -----------------------------------------------------------


def ensure_package(module: str, install_hint: str | None = None) -> None:
    """Abort the program with a clear message if a dependency is missing."""

    try:
        __import__(module)
    except Exception as exc:  # pragma: no cover - depends on environment.
        hint = f" Install via `{install_hint}`" if install_hint else ""
        raise SystemExit(f"Required dependency '{module}' is not available.{hint}\n{exc}") from exc


def optional_import(module: str) -> Any:
    """Attempt to import ``module`` returning ``None`` when unavailable."""

    try:
        return __import__(module)
    except Exception:  # pragma: no cover - non deterministic.
        return None


def set_global_seed(seed: Optional[int]) -> None:
    """Set deterministic seeds across popular libraries."""

    if seed is None:
        return

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:  # pragma: no cover - optional dependency.
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover - torch might not be installed.
        pass


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


@dataclass
class DatasetConfig:
    data_root: Path
    train_images: Path
    val_images: Path
    class_names: Sequence[str] = field(default_factory=lambda: CLASS_NAMES)

    @property
    def yaml_dict(self) -> Dict[str, Any]:
        return {
            "path": str(self.data_root.resolve()),
            "train": str(self.train_images.resolve()),
            "val": str(self.val_images.resolve()),
            "names": {idx: name for idx, name in enumerate(self.class_names)},
        }


# --- Dataset helpers -----------------------------------------------------------


def discover_dataset(data_root: Path) -> DatasetConfig:
    """Locate YOLO folders within ``data_root`` and build a config object."""

    train_images = data_root / "train" / "images"
    val_images = data_root / "valid" / "images"

    if not train_images.exists():
        # some datasets use ``val`` instead of ``valid``
        alt = data_root / "val" / "images"
        if alt.exists():
            val_images = alt

    if not train_images.exists():
        raise FileNotFoundError(f"Cannot locate train/images under {data_root}")

    if not val_images.exists():
        raise FileNotFoundError(
            f"Cannot locate validation images under {data_root}. Expected either `val/images` or `valid/images`."
        )

    return DatasetConfig(data_root=data_root, train_images=train_images, val_images=val_images)


def ensure_data_yaml(data_root: Path) -> Path:
    """Generate (and cache) a data.yaml file describing the dataset."""

    DATA_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg = discover_dataset(data_root)
    suffix = data_root.resolve().name.replace(" ", "_")
    yaml_path = DATA_CONFIG_DIR / f"traffic_light_{suffix}.yaml"

    if yaml:
        content = yaml.safe_dump(cfg.yaml_dict, sort_keys=False)
    else:
        lines = [
            f"path: {cfg.data_root.resolve()}",
            f"train: {cfg.train_images.resolve()}",
            f"val: {cfg.val_images.resolve()}",
            "names:",
        ]
        for idx, name in enumerate(cfg.class_names):
            lines.append(f"  {idx}: {name}")
        content = "\n".join(lines) + "\n"

    if not yaml_path.exists() or yaml_path.read_text() != content:
        yaml_path.write_text(content)

    return yaml_path




# --- Logging helpers -----------------------------------------------------------


def dump_args(destination: Path, args: MutableMapping[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if yaml:
        destination.write_text(yaml.safe_dump(dict(args), sort_keys=False))
    else:  # fallback to JSON if PyYAML is not available.
        destination.write_text(json.dumps(args, indent=2))


def save_metrics(destination: Path, metrics: Dict[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metrics, indent=2))


def extract_detection_metrics(result: Any) -> Dict[str, Any]:
    """Normalise Ultralytics metric objects into a JSON serialisable dictionary."""

    metrics: Dict[str, Any] = {}
    if result is None:
        return metrics

    metric_obj = getattr(result, "metrics", None)
    if metric_obj is None:
        return metrics

    box = getattr(metric_obj, "box", None)
    if box is None:
        return metrics

    try:
        metrics["map50-95"] = float(getattr(box, "map", None) or 0.0)
        metrics["map50"] = float(getattr(box, "map50", None) or 0.0)
        metrics["map75"] = float(getattr(box, "map75", None) or 0.0)
        per_class = getattr(box, "maps", None) or []
        metrics["per_class_ap"] = {
            CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else str(idx): float(ap)
            for idx, ap in enumerate(per_class)
        }
    except Exception:  # pragma: no cover - depends on Ultralytics internals.
        metrics = {}
    return metrics


# --- Ultralytics helpers -------------------------------------------------------


def load_model(weights: str | Path) -> Any:
    ensure_package("ultralytics", "pip install ultralytics")
    from ultralytics import YOLO

    return YOLO(str(weights))


def default_model() -> Any:
    return load_model(DEFAULT_MODEL)


# --- Command implementations ---------------------------------------------------


def command_setup(args: argparse.Namespace) -> None:
    print("Environment & dataset checks")
    ensure_package("torch", "pip install torch --extra-index-url https://download.pytorch.org/whl/cpu")
    ensure_package("ultralytics", "pip install ultralytics")
    optional_packages = ["opencv-python", "pandas", "numpy", "tqdm", "nbformat"]
    for pkg in optional_packages:
        try:
            ensure_package(pkg)
        except SystemExit as exc:  # degrade to warning for optional deps.
            print(f"[warning] {exc}")
    data_root = Path(args.data_root).expanduser().resolve()
    cfg = discover_dataset(data_root)
    print("Detected dataset configuration:")
    print(json.dumps(cfg.yaml_dict, indent=2))
    print("Class labels:", ", ".join(CLASS_NAMES))
    data_yaml = ensure_data_yaml(data_root)
    print(f"Data YAML available at {data_yaml}")


def command_train(args: argparse.Namespace) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root).expanduser().resolve()
    data_yaml = ensure_data_yaml(data_root)
    set_global_seed(args.seed)

    model = load_model(args.model or DEFAULT_MODEL)

    exp_name = f"train_{timestamp()}"
    overrides = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "lr0": args.lr0,
        "patience": args.patience,
        "device": args.device,
        "seed": args.seed,
        "project": str(RUNS_DIR),
        "name": exp_name,
        "verbose": True,
        "pretrained": True,
        "save": True,
    }

    if args.augment:
        overrides["augment"] = True
    if args.flipud is not None:
        overrides["flipud"] = args.flipud
    if args.fliplr is not None:
        overrides["fliplr"] = args.fliplr

    print("Training with overrides:")
    print(json.dumps(overrides, indent=2))

    results = model.train(**overrides)
    save_dir = Path(getattr(results, "save_dir", RUNS_DIR / exp_name))

    metrics = extract_detection_metrics(results)
    if metrics:
        print(f"Final validation mAP50-95: {metrics.get('map50-95', 0.0):.4f}")
    else:
        print("Unable to extract metrics from Ultralytics result object.")

    metrics_path = save_dir / "metrics.json"
    save_metrics(metrics_path, metrics)

    args_path = save_dir / "args.yaml"
    dump_args(args_path, vars(args))

    best_weight_src = save_dir / "weights" / "best.pt"
    if best_weight_src.exists():
        best_weight_dst = RUNS_DIR / "best.pt"
        shutil.copy(best_weight_src, best_weight_dst)
        print(f"Best checkpoint copied to {best_weight_dst}")
    else:
        print("Best checkpoint not found. Inspect the training logs for issues.")

    print(f"Training artifacts stored in {save_dir}")


def command_validate(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root).expanduser().resolve()
    weights = Path(args.weights or (RUNS_DIR / "best.pt"))
    if not weights.exists():
        raise SystemExit(f"Weights file not found: {weights}")

    model = load_model(weights)
    data_yaml = ensure_data_yaml(data_root)

    overrides = {
        "data": str(data_yaml),
        "imgsz": args.imgsz,
        "project": str(RUNS_DIR),
        "name": f"val_{timestamp()}",
        "device": args.device,
        "split": "val",
    }

    print(json.dumps(overrides, indent=2))
    results = model.val(**overrides)
    save_dir = Path(getattr(results, "save_dir", RUNS_DIR))
    metrics = extract_detection_metrics(results)
    metrics_path = save_dir / "val_metrics.json"
    save_metrics(metrics_path, metrics)

    if metrics:
        print("Validation metrics:")
        print(json.dumps(metrics, indent=2))
    else:
        print("Validation completed but metrics could not be parsed.")

    cm_path = save_dir / "confusion_matrix.png"
    if cm_path.exists():
        print(f"Confusion matrix saved to {cm_path}")


def command_tune(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root).expanduser().resolve()
    data_yaml = ensure_data_yaml(data_root)
    model = load_model(args.model or DEFAULT_MODEL)
    try:
        results = model.tune(
            data=str(data_yaml),
            imgsz=args.imgsz,
            epochs=args.epochs,
            iterations=args.trials,
            batch=args.batch,
            device=args.device,
            project=str(RUNS_DIR),
            name=f"tune_{timestamp()}",
        )
    except AttributeError:  # pragma: no cover - depends on Ultralytics version.
        raise SystemExit("This version of Ultralytics does not expose `YOLO.tune`. Consider upgrading.")

    save_dir = Path(getattr(results, "save_dir", RUNS_DIR))
    metrics = extract_detection_metrics(results)
    metrics_path = save_dir / "tune_metrics.json"
    save_metrics(metrics_path, metrics)
    print("Tuning finished. Inspect the generated hyp*.yaml file for best hyper-parameters.")


def draw_boxes(frame, detections: List[Tuple[int, float, Tuple[int, int, int, int]]]) -> None:
    import cv2

    for cls_id, conf, box in detections:
        x1, y1, x2, y2 = box
        color = [int(x) for x in ((cls_id * 53) % 255, (cls_id * 97) % 255, (cls_id * 193) % 255)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASS_NAMES[cls_id]} {conf:.2f}" if cls_id < len(CLASS_NAMES) else f"{cls_id} {conf:.2f}"
        cv2.putText(frame, label, (x1, max(10, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def smooth_majority(labels: List[str], window: int = 5) -> List[str]:
    smoothed: List[str] = []
    for idx in range(len(labels)):
        start = max(0, idx - window + 1)
        window_labels = labels[start : idx + 1]
        counter = Counter(window_labels)
        most_common = counter.most_common()
        if not most_common:
            smoothed.append("unknown")
            continue
        best_count = most_common[0][1]
        candidates = [label for label, count in most_common if count == best_count]
        # favour the most recent label when there is a tie
        for label in reversed(window_labels):
            if label in candidates:
                smoothed.append(label)
                break
        else:
            smoothed.append(most_common[0][0])
    return smoothed


def run_sahi_inference(frame, args, class_names) -> List[Tuple[int, float, Tuple[int, int, int, int]]]:
    sahi = optional_import("sahi")
    if sahi is None:
        raise RuntimeError("SAHI requested but the `sahi` package is not installed.")

    from sahi.predict import get_sliced_prediction
    from sahi.models.yolo import YOLODetectionModel

    detection_model = YOLODetectionModel(
        model_path=str(args.weights),
        confidence_threshold=args.conf_thres,
        device=args.device,
        image_size=args.imgsz,
    )

    prediction = get_sliced_prediction(
        image=frame[:, :, ::-1],  # convert BGR->RGB
        detection_model=detection_model,
        slice_height=args.tile_size,
        slice_width=args.tile_size,
        overlap_height_ratio=args.tile_overlap,
        overlap_width_ratio=args.tile_overlap,
        postprocess_match_metric=args.postprocess_match_metric,
    )

    detections: List[Tuple[int, float, Tuple[int, int, int, int]]] = []
    for obj in prediction.object_prediction_list:
        bbox = obj.bbox.to_xyxy()  # returns [x1, y1, x2, y2]
        cls_id = int(obj.category.id)
        conf = float(obj.score.value)
        detections.append((cls_id, conf, tuple(int(v) for v in bbox)))
    return detections


def run_yolo_inference(model, frame, args) -> List[Tuple[int, float, Tuple[int, int, int, int]]]:
    import numpy as np

    results = model.predict(frame, conf=args.conf_thres, iou=args.iou_thres, device=args.device, verbose=False)
    detections: List[Tuple[int, float, Tuple[int, int, int, int]]] = []
    if not results:
        return detections

    result = results[0]
    if not hasattr(result, "boxes"):
        return detections

    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)
    for cls_id, conf, bbox in zip(clss, confs, xyxy):
        x1, y1, x2, y2 = bbox.tolist()
        detections.append((int(cls_id), float(conf), (x1, y1, x2, y2)))
    return detections


def command_infer_video(args: argparse.Namespace) -> None:
    ensure_package("opencv-python", "pip install opencv-python")
    ensure_package("pandas", "pip install pandas")

    import cv2
    import pandas as pd

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    weights = Path(args.weights or (RUNS_DIR / "best.pt"))
    if not weights.exists():
        raise SystemExit(f"Weights file not found: {weights}")

    args.weights = weights
    model = None if args.sahi else load_model(weights)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    output_video_path = OUTPUTS_DIR / "annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    rows: List[Dict[str, Any]] = []
    predominant: List[str] = []

    frame_idx = 0
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if args.sahi:
            try:
                detections = run_sahi_inference(frame, args, CLASS_NAMES)
            except Exception as exc:
                print(f"[warning] SAHI failed ({exc}); falling back to vanilla prediction.")
                model = model or load_model(weights)
                args.sahi = False
                detections = run_yolo_inference(model, frame, args)
        else:
            model = model or load_model(weights)
            detections = run_yolo_inference(model, frame, args)

        draw_boxes(frame, detections)
        writer.write(frame)

        t_ms = (frame_idx / fps) * 1000.0
        area_per_class: Dict[int, float] = {cls: 0.0 for cls in range(len(CLASS_NAMES))}
        for cls_id, conf, (x1, y1, x2, y2) in detections:
            rows.append(
                {
                    "frame": frame_idx,
                    "t_ms": round(t_ms, 2),
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                    "cls_id": int(cls_id),
                    "cls_name": CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id),
                    "conf": float(conf),
                }
            )
            area = max(0, (x2 - x1)) * max(0, (y2 - y1)) * conf
            area_per_class[cls_id] = area_per_class.get(cls_id, 0.0) + area

        if detections:
            cls_id = max(area_per_class, key=area_per_class.get)
            predominant.append(CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unknown")
        else:
            predominant.append("unknown")

        frame_idx += 1

    cap.release()
    writer.release()

    detections_path = OUTPUTS_DIR / "detections.csv"
    if rows:
        detections_df = pd.DataFrame(rows)
        detections_df.to_csv(detections_path, index=False)
        print(f"Detections saved to {detections_path}")
    else:
        detections_df = pd.DataFrame(columns=["frame", "t_ms", "x1", "y1", "x2", "y2", "cls_id", "cls_name", "conf"])
        detections_df.to_csv(detections_path, index=False)
        print("No detections were produced.")

    summary_path = OUTPUTS_DIR / "summary_per_frame.csv"
    summary_df = pd.DataFrame({"frame": range(len(predominant)), "predominant_color": predominant})
    summary_df["smoothed_color"] = smooth_majority(predominant, window=args.smooth_window)
    summary_df.to_csv(summary_path, index=False)
    print(f"Per-frame summary saved to {summary_path}")

    elapsed = time.time() - start_time
    print(f"Processed {frame_idx} frames (input frames: {frame_count}) in {elapsed:.2f}s. Annotated video: {output_video_path}")


def command_export(args: argparse.Namespace) -> None:
    weights = Path(args.weights or (RUNS_DIR / "best.pt"))
    if not weights.exists():
        raise SystemExit(f"Weights not found: {weights}")

    model = load_model(weights)
    export_path = model.export(format="onnx", opset=args.opset, dynamic=args.dynamic)
    print(f"ONNX model exported to {export_path}")


def command_demo(args: argparse.Namespace) -> None:
    demo_args = argparse.Namespace(
        data_root=args.data_root,
        epochs=min(args.epochs, 5),
        imgsz=args.imgsz,
        batch=max(1, args.batch // 2),
        lr0=args.lr0,
        seed=args.seed,
        device=args.device,
        patience=max(5, args.patience // 2),
        augment=True,
        flipud=args.flipud,
        fliplr=args.fliplr,
        model=args.model,
    )

    print("Running demo training...")
    command_train(demo_args)

    print("Running demo validation...")
    command_validate(
        argparse.Namespace(
            data_root=args.data_root,
            weights=args.weights or (RUNS_DIR / "best.pt"),
            imgsz=args.imgsz,
            device=args.device,
        )
    )

    print("Running demo video inference...")
    command_infer_video(
        argparse.Namespace(
            weights=args.weights or (RUNS_DIR / "best.pt"),
            video=args.video,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            device=args.device,
            sahi=False,
            imgsz=args.imgsz,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            postprocess_match_metric=args.postprocess_match_metric,
            smooth_window=args.smooth_window,
        )
    )


def command_export_notebook(args: argparse.Namespace) -> None:
    ensure_package("nbformat", "pip install nbformat")
    import nbformat as nbf

    notebook_exists = NOTEBOOK_PATH.exists()
    if notebook_exists and not args.overwrite:
        print(
            f"Notebook already exists at {NOTEBOOK_PATH}. "
            "Re-run with --overwrite to replace it."
        )
        return

    title = textwrap.dedent(
        f"""
        # Project 1 – Traffic Light Object Detection

        Train, validate, and run inference for the traffic-light colour detector using the
        command line interface shipped with this repository. Execute each cell in order to
        reproduce the full workflow on your machine.

        - **Dataset root**: `{args.data_root}`
        - **Classes**: {', '.join(CLASS_NAMES)}
        - **Key files**:
          - `Project_1_object_detection_traffic_light.py` – primary CLI entrypoint
          - `Small Traffic Light.v1i.yolov11/` – YOLO-format dataset
          - `runs/` & `outputs/` – generated training and inference artefacts
        """
    ).strip()

    install_cell = textwrap.dedent(
        """
        from importlib import import_module, util
        from pathlib import Path
        import sys

        def guard(module, package=None, critical=False):
            if util.find_spec(module):
                return import_module(module)
            pkg = package or module
            if module == "torch":
                print("PyTorch missing. Install CPU build via `pip install torch --index-url https://download.pytorch.org/whl/cpu`.")
                return None
            msg = f"Install via `pip install {pkg}`." if critical else f"Optional dependency `{module}` missing."
            print(msg)
            if critical:
                raise ModuleNotFoundError(module)
            return None

        project_root = Path.cwd()
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        nbformat = guard("nbformat", critical=True)
        torch = guard("torch")
        ultralytics = guard("ultralytics")
        cv2 = guard("cv2", "opencv-python")
        numpy, pandas, tqdm = (guard(m) for m in ("numpy", "pandas", "tqdm"))
        sahi = guard("sahi")
        supervision = guard("supervision")
        """
    ).strip()

    params_cell = textwrap.dedent(
        f"""
        from pathlib import Path

        DATA_ROOT = Path("{args.data_root}")
        EPOCHS = 120
        IMGSZ = 640
        BATCH = 16
        DEVICE = "auto"
        SEED = 42
        USE_SAHI = True
        VIDEO_PATH = "inference_traffic_light_video.mp4"
        WEIGHTS_PATH = "runs/best.pt"
        CLASS_NAMES = ["green", "off", "red", "wait_on", "yellow"]
        """
    ).strip()

    dataset_cell = textwrap.dedent(
        """
        import subprocess, sys

        setup_cmd = [
            sys.executable,
            "Project_1_object_detection_traffic_light.py",
            "setup",
            "--data-root", str(DATA_ROOT),
        ]
        result = subprocess.run(setup_cmd, check=False)
        if result.returncode != 0:
            raise SystemExit(f"Setup command failed with exit code {result.returncode}")
        """
    ).strip()

    train_cell = textwrap.dedent(
        """
        import subprocess, sys

        train_cmd = [
            sys.executable,
            "Project_1_object_detection_traffic_light.py",
            "train",
            "--data-root", str(DATA_ROOT),
            "--epochs", str(EPOCHS),
            "--imgsz", str(IMGSZ),
            "--batch", str(BATCH),
            "--device", DEVICE,
            "--seed", str(SEED),
            "--patience", "20",
        ]
        result = subprocess.run(train_cmd, check=False)
        if result.returncode != 0:
            raise SystemExit(f"Train command failed with exit code {result.returncode}")
        """
    ).strip()

    validate_cell = textwrap.dedent(
        """
        import subprocess, sys

        validate_cmd = [
            sys.executable,
            "Project_1_object_detection_traffic_light.py",
            "validate",
            "--data-root", str(DATA_ROOT),
        ]
        result = subprocess.run(validate_cmd, check=False)
        if result.returncode != 0:
            raise SystemExit(f"Validate command failed with exit code {result.returncode}")
        """
    ).strip()

    infer_cell = textwrap.dedent(
        """
        import subprocess, sys

        infer_cmd = [
            sys.executable,
            "Project_1_object_detection_traffic_light.py",
            "infer-video",
            "--weights", WEIGHTS_PATH,
            "--video", VIDEO_PATH,
            "--sahi", str(USE_SAHI).lower(),
            "--conf-thres", "0.25",
            "--iou-thres", "0.5",
        ]
        result = subprocess.run(infer_cmd, check=False)
        if result.returncode != 0:
            raise SystemExit(f"Inference failed with exit code {result.returncode}")
        """
    ).strip()

    results_cell = textwrap.dedent(
        """
        import json
        from pathlib import Path

        metrics_files = sorted(Path("runs").rglob("metrics.json"), key=lambda p: p.stat().st_mtime)
        if metrics_files:
            latest = metrics_files[-1]
            with latest.open() as fh:
                metrics = json.load(fh)
            print("Latest metrics file:", latest)
            print("mAP50-95:", metrics.get("map50-95"))
        else:
            print("No metrics.json files found. Train the model first.")
        print("Annotated video:", Path("outputs/annotated.mp4"))
        print("Detections CSV:", Path("outputs/detections.csv"))
        print("Frame summary:", Path("outputs/summary_per_frame.csv"))
        """
    ).strip()

    viz_cell = textwrap.dedent(
        """
        if "cv2" not in globals() or cv2 is None:
            print("OpenCV not available. Install `opencv-python` to preview frames.")
        else:
            import matplotlib.pyplot as plt

            video_path = Path("outputs/annotated.mp4")
            if not video_path.exists():
                print("Annotated video not found. Run the inference step first.")
            else:
                cap = cv2.VideoCapture(str(video_path))
                frames = []
                for _ in range(3):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cap.release()
                if not frames:
                    print("No frames available for preview.")
                else:
                    fig, axes = plt.subplots(1, len(frames), figsize=(15, 5))
                    if len(frames) == 1:
                        axes = [axes]
                    for ax, frame in zip(axes, frames):
                        ax.imshow(frame)
                        ax.axis("off")
                    plt.show()
        """
    ).strip()

    notes_cell = textwrap.dedent(
        """
        ### Notes & Next steps

        - Explore larger YOLO backbones or RT-DETR for higher accuracy.
        - Extend augmentation strategies (colour jitter, histogram equalisation).
        - Investigate mixed-precision training once GPU acceleration is available.
        """
    ).strip()

    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(title),
        nbf.v4.new_code_cell(install_cell),
        nbf.v4.new_code_cell(params_cell),
        nbf.v4.new_code_cell(dataset_cell),
        nbf.v4.new_code_cell(train_cell),
        nbf.v4.new_code_cell(validate_cell),
        nbf.v4.new_code_cell(infer_cell),
        nbf.v4.new_code_cell(results_cell),
        nbf.v4.new_code_cell(viz_cell),
        nbf.v4.new_markdown_cell(notes_cell),
    ]

    NOTEBOOK_PATH.write_text(nbf.writes(nb))
    print(f"Notebook exported to {NOTEBOOK_PATH}")


# --- CLI glue ------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Traffic light object detection CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-root", type=str, default="Small Traffic Light.v1i.yolov11", help="Dataset root directory")

    parser_setup = sub.add_parser("setup", parents=[common], help="Verify dependencies and dataset layout")
    parser_setup.set_defaults(func=command_setup)

    parser_train = sub.add_parser("train", parents=[common], help="Train a YOLO model")
    parser_train.add_argument("--epochs", type=int, default=100)
    parser_train.add_argument("--imgsz", type=int, default=640)
    parser_train.add_argument("--batch", type=int, default=16)
    parser_train.add_argument("--lr0", type=float, default=0.01)
    parser_train.add_argument("--seed", type=int, default=42)
    parser_train.add_argument("--device", type=str, default="auto")
    parser_train.add_argument("--patience", type=int, default=50)
    parser_train.add_argument("--augment", action="store_true", help="Enable default YOLO augmentations")
    parser_train.add_argument("--flipud", type=float, default=None, help="Vertical flip probability override")
    parser_train.add_argument("--fliplr", type=float, default=None, help="Horizontal flip probability override")
    parser_train.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser_train.set_defaults(func=command_train)

    parser_val = sub.add_parser("validate", parents=[common], help="Run validation on the val split")
    parser_val.add_argument("--weights", type=str, default=None)
    parser_val.add_argument("--imgsz", type=int, default=640)
    parser_val.add_argument("--device", type=str, default="auto")
    parser_val.set_defaults(func=command_validate)

    parser_tune = sub.add_parser("tune", parents=[common], help="Hyper-parameter search using Ultralytics tuner")
    parser_tune.add_argument("--trials", type=int, default=4, help="Number of tuning trials")
    parser_tune.add_argument("--epochs", type=int, default=40)
    parser_tune.add_argument("--imgsz", type=int, default=640)
    parser_tune.add_argument("--batch", type=int, default=16)
    parser_tune.add_argument("--device", type=str, default="auto")
    parser_tune.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser_tune.set_defaults(func=command_tune)

    parser_export = sub.add_parser("export", help="Export a trained checkpoint to ONNX")
    parser_export.add_argument("--weights", type=str, default=None)
    parser_export.add_argument("--opset", type=int, default=12)
    parser_export.add_argument("--dynamic", action="store_true")
    parser_export.set_defaults(func=command_export)

    parser_infer = sub.add_parser("infer-video", help="Run inference on the provided video")
    parser_infer.add_argument("--weights", type=str, default=None)
    parser_infer.add_argument("--video", type=str, default="inference_traffic_light_video.mp4")
    parser_infer.add_argument("--conf-thres", type=float, default=0.25)
    parser_infer.add_argument("--iou-thres", type=float, default=0.5)
    parser_infer.add_argument("--device", type=str, default="auto")
    parser_infer.add_argument("--sahi", action="store_true")
    parser_infer.add_argument("--imgsz", type=int, default=640)
    parser_infer.add_argument("--tile-size", type=int, default=512)
    parser_infer.add_argument("--tile-overlap", type=float, default=0.2)
    parser_infer.add_argument("--postprocess-match-metric", type=str, default="IOU")
    parser_infer.add_argument("--smooth-window", type=int, default=5)
    parser_infer.set_defaults(func=command_infer_video)

    parser_demo = sub.add_parser("demo", parents=[common], help="Run a short end-to-end demo")
    parser_demo.add_argument("--epochs", type=int, default=5)
    parser_demo.add_argument("--imgsz", type=int, default=640)
    parser_demo.add_argument("--batch", type=int, default=8)
    parser_demo.add_argument("--lr0", type=float, default=0.01)
    parser_demo.add_argument("--seed", type=int, default=42)
    parser_demo.add_argument("--device", type=str, default="auto")
    parser_demo.add_argument("--patience", type=int, default=10)
    parser_demo.add_argument("--augment", action="store_true")
    parser_demo.add_argument("--flipud", type=float, default=None)
    parser_demo.add_argument("--fliplr", type=float, default=None)
    parser_demo.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser_demo.add_argument("--weights", type=str, default=None)
    parser_demo.add_argument("--video", type=str, default="inference_traffic_light_video.mp4")
    parser_demo.add_argument("--conf-thres", type=float, default=0.25)
    parser_demo.add_argument("--iou-thres", type=float, default=0.5)
    parser_demo.add_argument("--tile-size", type=int, default=512)
    parser_demo.add_argument("--tile-overlap", type=float, default=0.2)
    parser_demo.add_argument("--postprocess-match-metric", type=str, default="IOU")
    parser_demo.add_argument("--smooth-window", type=int, default=5)
    parser_demo.set_defaults(func=command_demo)

    parser_nb = sub.add_parser("export-notebook", parents=[common], help="Generate a companion notebook")
    parser_nb.add_argument("--overwrite", action="store_true", help="Replace an existing notebook")
    parser_nb.set_defaults(func=command_export_notebook)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
