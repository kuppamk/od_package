# BDD100K Object Detection Pipeline (Faster R-CNN, PyTorch)

This project implements an end-to-end object detection pipeline using the **BDD100K dataset** and **Faster R-CNN (ResNet50)**. It includes:

- Custom dataset handler for BDD100K annotations
- Training and validation pipeline
- Inference with confidence thresholding
- Bounding box visualization after inference
- Evaluation using mAP and IoU metrics
- JSON-based prediction saving
- YAML-based config file
- Integrated logging to console and file

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/bdk_object_detection.git
cd bdk_object_detection
```

### 2.  Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset

Download the **BDD100K images and labels** from the official website:

- **Images**: [https://bdd-data.berkeley.edu/](https://bdd-data.berkeley.edu/)
- **Labels**: [https://bdd-data.berkeley.edu/](https://bdd-data.berkeley.edu/)

After downloading and extracting, update the paths in `od_model/config.yaml:`:

```yaml
TRAIN_IMAGE_DIR: "/path/to/bdd100k/images/100k/train"
TRAIN_LABEL_JSON_PATH:  "/path/to/bdd100k_labels_images_train.json"
VALID_IMAGE_DIR: "/path/to/bdd100k/images/100k/val"
VALID_LABEL_JSON_PATH:  "/path/to/bdd100k_labels_images_val.json"
BASE_WEIGTHS:  "/path/to/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"
```

## Running the Project

The project supports a CLI interface via argparse.

### Available Flags
--train — Run the training pipeline

--inference — Run inference using a trained model

--eval — Evaluate predictions using metrics like mAP, Precision, Recall

--visualize — Visualize predicted and ground-truth bounding boxes

--model_path — Path to the .pth model file

--config_path — (Optional) Path to your YAML config file (default: config.yaml)

### Example Commands

```bash
# Run training, inference, and evaluation
python main.py --train --inference --eval --visualize --model_path artifacts/best_model.pth

# Run inference only
python main.py --inference --model_path artifacts/best_model.pth

# Run evaluation only
python main.py --eval

# Visualization only
python main.py --visualize
```

## Execution Modes & Requirements

Below are detailed instructions and prerequisites for each pipeline mode:

---

### `--train`: Train the Model

Runs the full training loop and saves the best model checkpoint.

#### Requirements

- Make sure paths to training and validation images/labels are correctly set in:
  - `od_model/config.yaml`

#### What it does

- Loads the BDD100K dataset
- Trains the model using Faster R-CNN model
- Saves the best model to `artifacts/best_model.pth`
- Logs progress to artifacts/run.log

#### Command

```bash
python main.py --train
```

### `--inference`: Run Inference

Performs inference on the validation set and saves predictions.

#### Requirements

- Trained model `.pth` file (e.g., `artifacts/best_model.pth`)
- Dataset must be available (image + label paths in `od_model/config.py`)
- Set `--model_path` to your `.pth` file

#### What it does

- Loads the model and validation images
- Runs inference with a score threshold (configurable in `config.py`)
- Saves predictions to `artifacts/predictions.json`

#### Command

```bash
python main.py --inference --model_path artifacts/best_model.pth
```

### `--eval`: Evaluate Model Performance

Evaluates saved predictions using mAP, precision, recall, and per-class AP.

#### Requirements

- `artifacts/predictions.json` must exist (generated from a previous inference run)
- Number of classes must match the model output (configured in `pipelines.py`)

#### What it does

- Loads predictions from the saved JSON file
- Computes the following metrics:
  - Overall **Precision**, **Recall**, **F1 Score**
  - **mAP** (mean Average Precision) across multiple IoU thresholds
  - **Class-wise Average Precision (AP)**

#### Command

```bash
python main.py --eval
```
### `--visualize`: Visualize Predictions

Saves side-by-side visualization of predicted and GT bounding boxes.

#### Requirements

- `artifacts/predictions.json` must exist (generated from inference)
- Validation images must be available and paths set in `config.yaml`

#### What it does

- Overlays bounding boxes on each image:
  - Ground Truth (green boxes)
  - Predictions (red boxes)
- Saves visualizations to `artifacts/vis/` as `.jpg` images

#### Command

```bash
python main.py --visualize
```

## Features

- **Modular & reusable architecture**
- **Easy YAML-based configuration**
- **Logging** to both terminal and file (`artifacts/run.log`)
- **Visualization support** after inference
- **Supports both CPU & CUDA**

---

## TODOs & Improvements
- Add support for COCO-style evaluation metrics
- Add early stopping and LR scheduler


