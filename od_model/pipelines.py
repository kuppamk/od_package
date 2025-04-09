from typing import Any, Dict
import logging

from od_model.bdd_dataset import BDDDetectionDataset, DataPipeline
from od_model.model import get_model
from od_model.train import TrainModel
from od_model.inference import InferenceModel
from od_model.evaluate import Evaluator
from od_model.utils import load_predictions_from_json, load_config
from od_model.utils import run_visualizer

import torch


def training_pipeline(cfg: Dict) -> int:
    """Runs the full training pipeline for the detection model.

    Args:
        cfg (Dict): Configuration dictionary containing dataset paths,
            hyperparameters, and model weights path.

    Returns:
        int: 0 if the pipeline completes successfully.
    """
    data_obj = DataPipeline(
        train_images_path=cfg["TRAIN_IMAGE_DIR"],
        train_labels_json_path=cfg["TRAIN_LABEL_JSON_PATH"],
        valid_images_path=cfg["VALID_IMAGE_DIR"],
        valid_labels_json_path=cfg["VALID_LABEL_JSON_PATH"],
        evaluation=False,
    )
    train_dataloader, valid_dataloader = data_obj.create_dataloaders()

    model = get_model(
        model_path=cfg["BASE_WEIGTHS"],
        num_classes=data_obj.num_classes,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_obj = TrainModel(model=model, device=device, lr=cfg["LR"])

    train_obj.train(
        train_dataloader=train_dataloader,
        num_epochs=cfg["NUM_EPOCHS"],
        val_dataloader=valid_dataloader,
        save_path="artifacts/best_model.pth",
    )
    return 0


def inference_pipeline(cfg: Dict, model_path: str) -> int:
    """Runs inference using the trained model.

    Args:
        cfg (Dict): Configuration dictionary containing dataset paths,
            hyperparameters, and model weights path.
        model_path (str): Path to the saved model weights.

    Returns:
        int: 0 if inference runs successfully.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_obj = DataPipeline(
        train_images_path=cfg["TRAIN_IMAGE_DIR"],
        train_labels_json_path=cfg["TRAIN_LABEL_JSON_PATH"],
        valid_images_path=cfg["VALID_IMAGE_DIR"],
        valid_labels_json_path=cfg["VALID_LABEL_JSON_PATH"],
        evaluation=True,
    )
    _, valid_dataloader = data_obj.create_dataloaders()
    model = get_model(
        model_path=cfg["BASE_WEIGTHS"],
        num_classes=data_obj.num_classes,
    )

    infer_obj = InferenceModel(
        model=model,
        weights_path=model_path,
        device=device,
    )

    infer_obj.run_inference(
        dataloader=valid_dataloader,
        score_thresh=cfg["SCORE_THRESH"],
        save_dir="artifacts",
    )
    return 0

def evaluation_pipeline(cfg: Dict, json_path: str, num_classes: int, logger: logging.Logger) -> None:
    """Evaluates predictions stored in a JSON file using mAP and class-wise AP.

    Args:
        cfg (Dict): Configuration dictionary containing dataset paths,
            hyperparameters, and model weights path.
        json_path (str): Path to the JSON file with predictions.
        num_classes (int): Number of classes in the dataset.
        logger (logging.Logger): Logger instance for logging the evaluation output.
    """
    predictions = load_predictions_from_json(json_path)
    eval_obj = Evaluator(
        predictions=predictions,
        num_classes=num_classes,
    )

    logger.info("Overall metrics at different IoU thresholds:")
    for i in range(1, 10):
        iou_thresh = i / 10
        result = eval_obj.compute_overall_metrics(iou_thresh=iou_thresh)
        logger.info(f"IoU {iou_thresh:.1f}: Precision: {result[0]:.4f}, "
                    f"Recall: {result[1]:.4f}, F1: {result[2]:.4f}")

    logger.info("\n Class-wise AP at different IoU thresholds:")
    for i in range(1, 10):
        iou_thresh = i / 10
        ap_per_class, mAP = eval_obj.compute_classwise_ap(iou_thresh=iou_thresh)
        logger.info(f"IoU {iou_thresh:.1f}: mAP: {mAP:.4f}, AP per class: {ap_per_class}")


def visualize_pipeline(
    cfg: Dict,
    json_path: str,
    num_samples: int = 100
) -> None:
    """Loads predictions and visualizes them with bounding boxes.

    Args:
        cfg (Dict): Configuration dictionary containing dataset paths,
            hyperparameters, and model weights path.
        json_path (str): Path to saved predictions JSON.
        valid_image_dir (str): Directory of validation images.
        num_samples (int): Number of images to visualize.
    """
    predictions = load_predictions_from_json(json_path)
    run_visualizer(
        predictions=predictions,
        valid_image_dir=cfg["VALID_IMAGE_DIR"],
        num_samples=num_samples,
        output_path="artifacts/viz"
    )
