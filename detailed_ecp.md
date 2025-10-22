# Detailed Explanation of Traffic Light Object Detection Notebook

This document provides a comprehensive walkthrough of the `Project_1_object_detection_traffic_light.ipynb` notebook. The notebook orchestrates the full workflow required to train a YOLOv8 detector on a custom traffic-light dataset and run inference on a sample video. Each section below mirrors the notebook structure and explains the motivation behind the code as well as notable implementation details.

## 1. Project Overview (Markdown Cell 0)
The notebook opens with a high-level summary of the project goals and assets:
- **Objective:** Train, validate, and run inference for a traffic-light colour detector using Ultralytics' YOLO models.
- **Dataset:** The YOLO-formatted dataset lives under `Small Traffic Light.v1i.yolov11` and contains class labels for `green`, `off`, `red`, `wait_on`, and `yellow`.
- **Generated artefacts:** Training outputs are written under `runs/` while inference deliverables such as videos and CSVs are stored in `outputs/`.

## 2. Environment Preparation (Code Cell 1)
To ensure reproducibility, the notebook installs every required dependency at runtime using `pip` commands executed through Python's `subprocess` module. The packages include:
- **Core framework:** `torch` (CPU wheels) and `ultralytics` for YOLO training and inference.
- **Utility libraries:** `opencv-python`, `pandas`, `numpy`, `tqdm`, `matplotlib`, `pyyaml`, and `nbformat`.
- **Optional tooling:** `sahi` and `supervision` for slicing-based inference and visualisation support.

By iterating over a list of commands, the cell prints the command being executed and launches the installer. The approach avoids manual shell interaction and works seamlessly within hosted notebook environments.

Once installation is complete, the cell imports all Python modules used later in the workflow. Grouping the imports in the same cell ensures the runtime state is fully configured before moving to dataset preparation and training.

## 3. Experiment Hyperparameters (Code Cell 2)
Key configuration options are declared as module-level constants:
- File system locations (`DATA_ROOT`, `VIDEO_PATH`, `WEIGHTS_PATH`, `MODEL_PATH`).
- Training hyperparameters (`EPOCHS`, `IMGSZ`, `BATCH`, `DEVICE`, `SEED`).
- Post-processing and inference flags (`USE_SAHI`).
- Class label list (`CLASS_NAMES`).

Centralising these values at the top of the notebook makes it easy to tweak runs without touching downstream logic.

## 4. Dataset Structure Validation (Code Cell 3)
The notebook validates that the expected YOLO directory structure exists. It checks for `train/images` and `valid/images` folders under the dataset root, gracefully handling datasets that use `val` instead of `valid`. If either directory is missing, a descriptive `FileNotFoundError` is raised to prevent silent failures later during training.

## 5. Data Configuration File Generation (Code Cells 4–5)
YOLO training requires a YAML configuration file specifying dataset paths and label names. These cells:
1. Build a dictionary with absolute paths to the dataset root, training images, validation images, and a class index-to-name mapping.
2. Write the dictionary to `runs/data_configs/traffic_light_<dataset>.yaml` using `yaml.safe_dump`. The file is only overwritten when the contents change, avoiding unnecessary filesystem churn.
3. Print a concise summary of the generated configuration, including class names and resolved directories, to confirm correctness before training starts.

## 6. Sample Visualisation Utilities (Code Cells 6–7)
To facilitate manual inspection, the notebook defines helper functions:
- `yolo_to_xyxy` converts YOLO-normalised bounding boxes to pixel coordinates, enabling straightforward OpenCV drawing routines.
- `preview_samples` randomly samples training images, overlays bounding boxes using Matplotlib, and displays them inline.

Calling `preview_samples(n=5)` offers a quick qualitative check that annotations are aligned with the imagery.

## 7. YOLOv8 Training Invocation (Code Cell 8)
Training is triggered through the Ultralytics CLI by constructing a command list and executing it with `subprocess.run`. Key arguments include the data YAML path, pretrained weights (`yolov8l.pt`), number of epochs, image size, batch size, target device, and random seed. Early stopping is configured via `patience=20`. Printing the command string prior to execution provides transparency and simplifies debugging should the CLI report an error.

## 8. Video Inference with Ultralytics API (Code Cell 9)
After training, the notebook loads the best-performing checkpoint (expected at `runs/detect/train3/weights/best.pt`) using the `YOLO` Python API. It then runs `model.predict` on `inference_traffic_light_video.mp4` with the following settings:
- Confidence threshold of 0.25 and IoU threshold of 0.5 to balance precision and recall.
- `save=True`, coupled with `project` and `name`, to ensure annotated video outputs are persisted under `runs/predict/video_infer`.
- `exist_ok=True` to allow reruns without errors.

Upon completion, the cell prints the directory containing the prediction artefacts.

## 9. Training Metrics Summary (Code Cell 10)
To report quantitative performance, the notebook searches recursively under `runs/` for the most recent `metrics.json`. If found, it prints the mAP@0.5:0.95 value. Additional print statements point users to the expected locations of the annotated video (`outputs/annotated.mp4`), detections CSV, and per-frame summary CSV, acting as a checklist for downstream analysis scripts.

## 10. Annotated Video Preview (Code Cell 11)
For environments with OpenCV support, the notebook provides a lightweight visual sanity check:
- Attempts to read the first three frames from `outputs/annotated.mp4`.
- Converts them from BGR to RGB and renders them with Matplotlib.
- Handles missing video files or failed reads gracefully by printing explanatory messages.

This step allows users to quickly assess inference quality without leaving the notebook environment.

## 11. Next Steps (Markdown Cell 12)
The notebook concludes with actionable suggestions for future experimentation, such as exploring larger backbone models, augmenting training data, and enabling mixed-precision training when GPU resources are available.

---

By following the structure outlined above, the notebook delivers an end-to-end pipeline—from environment setup and dataset validation through model training, evaluation, and qualitative review—tailored for traffic-light object detection tasks.
