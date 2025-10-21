# Traffic Light Object Detection

This repository bundles a full, scriptable workflow for training and
deploying a traffic light colour detector using the Ultralytics YOLO
family. The `Project_1_object_detection_traffic_light.py` CLI covers setup,
training, validation, hyper-parameter tuning, exporting, video inference
(with optional SAHI tiling), a reproducible demo run, and programmatic
notebook export.

## Dataset

The dataset lives in `Small Traffic Light.v1i.yolov11/` and follows the
YOLO directory convention:

```
Small Traffic Light.v1i.yolov11/
├── train/images
├── train/labels
├── val/images (or valid/images)
└── val/labels (or valid/labels)
```

Five classes are supported throughout the pipeline:

| ID | Label    |
|----|----------|
| 0  | green    |
| 1  | off      |
| 2  | red      |
| 3  | wait_on  |
| 4  | yellow   |

## CLI usage

The Python CLI is the canonical interface. Each command below matches the
project brief.

```bash
# 1) Environment and dataset checks
python Project_1_object_detection_traffic_light.py setup \
  --data-root "Small Traffic Light.v1i.yolov11"

# 2) Training (update hyper-parameters as needed)
python Project_1_object_detection_traffic_light.py train \
  --data-root "Small Traffic Light.v1i.yolov11" \
  --epochs 120 --imgsz 640 --batch 16 --device auto --seed 42 --patience 20 \
  --model path/to/your/yolo_backbone.pt

# 3) Validation only
python Project_1_object_detection_traffic_light.py validate \
  --data-root "Small Traffic Light.v1i.yolov11"

# 4) Video inference (SAHI enabled)
python Project_1_object_detection_traffic_light.py infer-video \
  --weights runs/best.pt \
  --video inference_traffic_light_video.mp4 \
  --sahi --conf-thres 0.25 --iou-thres 0.5

# 5) Export ONNX checkpoint
python Project_1_object_detection_traffic_light.py export --weights runs/best.pt

# 6) Generate the companion notebook
python Project_1_object_detection_traffic_light.py export-notebook [--overwrite]

# 7) Quick demo (train+val+infer with small budgets)
python Project_1_object_detection_traffic_light.py demo \
  --data-root "Small Traffic Light.v1i.yolov11"

> **Tip:** `--model` (train/tune/demo) and `--weights` (validate/infer/demo) accept absolute or relative paths, so you can start
> from any YOLO checkpoint stored in your workspace.
```

## Results

*Training command:* `python Project_1_object_detection_traffic_light.py train --data-root "Small Traffic Light.v1i.yolov11" --epochs 120 --imgsz 640 --batch 16 --device auto --seed 42 --patience 20`

*Validation metrics (val split)*

- mAP50–95: _pending (run training to populate `runs/*/metrics.json`)_ — target ≥ 0.48
- mAP50: _pending_
- Per-class APs are logged in the same JSON artefact.

*Generated artefacts*

- Annotated video: `outputs/annotated.mp4`
- Raw detections: `outputs/detections.csv`
- Predominant colour summary: `outputs/summary_per_frame.csv`

### Approach highlights

1. Ultralytics YOLOv8 backbone with reproducible seeds and configurable training overrides.
2. Automatic `data.yaml` synthesis from the supplied YOLO-formatted dataset layout.
3. Optional augment controls (flip probabilities, generic augment toggle) exposed through the CLI.
4. Hyper-parameter tuning hook via `yolo.tune` for lightweight searches.
5. SAHI tiled inference path with configurable tile size and overlap for improved small-object recall.
6. Structured logging of arguments and metrics per run under `runs/` for experiment tracking.
7. Notebook export built programmatically via `nbformat` to mirror the CLI flow and preview training samples with bounding boxes.
8. Video inference produces annotated media plus CSV summaries with smoothed predominant colour labels.
9. ONNX export for deployment and integration with downstream runtimes.
10. Compact demo command that exercises the end-to-end flow on a tight budget.

## Notebook

A runnable notebook is generated automatically:

```bash
python Project_1_object_detection_traffic_light.py export-notebook \
  --data-root "Small Traffic Light.v1i.yolov11" --overwrite
```

The command regenerates `Project_1_object_detection_traffic_light.ipynb`
programmatically using `nbformat`. Open the notebook in JupyterLab or VS Code
and execute the cells sequentially. The first code cell installs all
dependencies via `pip`, the dataset cell synthesises `data.yaml`, and a preview
cell renders a few training images with their YOLO bounding boxes. Training,
validation, and inference cells call the CLI through subprocesses, keeping the
workflow isolated yet reproducible. Update the configuration cell to point
`MODEL_PATH`, `WEIGHTS_PATH`, and other variables to your own files, and omit
`--overwrite` to keep an existing notebook untouched.
