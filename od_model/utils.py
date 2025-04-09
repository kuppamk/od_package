import json
import os
from typing import Any, List, Dict
import yaml
import logging

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_logger():
    # Set up basic logging config
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler("artifacts/run.log")  # Optional: Save to file
        ]
    )
    logger = logging.getLogger(__name__)
    return logger




def save_predictions_to_json(predictions: Any, save_path: str = "outputs") -> None:
    """Saves model predictions to a JSON file.

    Args:
        predictions (Any): A nested structure containing model predictions (e.g., dict, 
            list, torch.Tensor).
        save_path (str): Directory to save the predictions file. Defaults to "outputs".
    """
    os.makedirs(save_path, exist_ok=True)

    def convert(obj: Any) -> Any:
        """Recursively converts torch.Tensors to native Python types.

        Args:
            obj (Any): Input object that may contain torch.Tensors.

        Returns:
            Any: Object with tensors converted to lists.
        """
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        else:
            return obj

    clean_preds = convert(predictions)

    output_path = os.path.join(save_path, "predictions.json")
    with open(output_path, "w") as f:
        json.dump(clean_preds, f, indent=2)


def load_predictions_from_json(json_path: str) -> List[Dict[str, torch.Tensor]]:
    """Loads predictions from a JSON file and converts lists back to torch.Tensors.

    Args:
        json_path (str): Path to the JSON file containing saved predictions.

    Returns:
        List[Dict[str, torch.Tensor]]: List of dictionaries with tensorized predictions.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    for pred in data:
        pred["pred_boxes"] = torch.tensor(pred["pred_boxes"], dtype=torch.float32)
        pred["pred_labels"] = torch.tensor(pred["pred_labels"], dtype=torch.int64)
        pred["gt_boxes"] = torch.tensor(pred["gt_boxes"], dtype=torch.float32)
        pred["gt_labels"] = torch.tensor(pred["gt_labels"], dtype=torch.int64)
        pred["pred_scores"] = torch.tensor(pred["pred_scores"], dtype=torch.float32)

    return data

def visualize_sample(
    image_tensor: torch.Tensor,
    gt_boxes: torch.Tensor,
    pred_boxes: torch.Tensor,
    pred_labels: torch.Tensor,
    save_path: str
) -> None:
    """Draws and saves ground truth and predicted boxes on an image.

    Args:
        image_tensor (torch.Tensor): Image in tensor format [C, H, W].
        gt_boxes (torch.Tensor): Ground truth bounding boxes [N, 4].
        pred_boxes (torch.Tensor): Predicted bounding boxes [M, 4].
        pred_labels (torch.Tensor): Predicted class labels (not shown here).
        save_path (str): Path to save the visualized image.
    """
    img = (image_tensor * 255).byte().clone()
    img = draw_bounding_boxes(img, gt_boxes, colors="green",
                              labels=["GT"] * len(gt_boxes))
    img = draw_bounding_boxes(img, pred_boxes, colors="red",
                              labels=["PRED"] * len(pred_boxes))
    img = F.to_pil_image(img)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path)

def run_visualizer(
    valid_image_dir: str,
    predictions: list,
    num_samples: int = 10,
    output_path: str = "artifacts/viz"
) -> None:
    """Visualizes a subset of predictions by drawing bounding boxes.

    Args:
        valid_image_dir (str): Directory containing validation images.
        predictions (list): List of prediction dicts loaded from JSON.
        num_samples (int): Number of samples to visualize.
        output_path (str): Path to store the images with GT and predicted boxes.
    """
    os.makedirs(output_path, exist_ok=True)
    for i, pred in enumerate(predictions):
        img_path = os.path.join(valid_image_dir, pred["filename"])
        if os.path.exists(img_path):
            image = Image.open(img_path).convert("RGB")
            image_tensor = F.to_tensor(image)

            visualize_sample(
                image_tensor,
                gt_boxes=pred["gt_boxes"],
                pred_boxes=pred["pred_boxes"],
                pred_labels=pred["pred_labels"],
                save_path=f"{output_path}/{pred['filename']}"
            )

        if i + 1 >= num_samples:
            break