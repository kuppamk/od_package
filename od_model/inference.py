import os
import torch
from tqdm import tqdm

from od_model.utils import save_predictions_to_json


class InferenceModel:
    """Performs inference on a dataset using a trained object detection model."""

    def __init__(self, model: torch.nn.Module, weights_path: str, device: torch.device):
        """Initializes the inference model with weights and device.

        Args:
            model (torch.nn.Module): The detection model architecture.
            weights_path (str): Path to the trained model weights (.pth file).
            device (torch.device): The device to run inference on (CPU or CUDA).
        """
        self.model = model
        self.model.load_state_dict(torch.load(weights_path))
        self.device = device

    def run_inference(
        self,
        dataloader,
        score_thresh: float = 0.4,
        save_dir: str = None
    ) -> list:
        """Runs inference on the dataset and saves predictions as JSON.

        Args:
            dataloader (DataLoader): DataLoader providing the input data.
            score_thresh (float): Confidence threshold for filtering predictions.
            save_dir (str): Directory to save the prediction results in JSON format.

        Returns:
            list: A list of prediction dictionaries per image.
        """
        self.model.to(self.device)
        self.model.eval()

        os.makedirs(save_dir, exist_ok=True)
        predictions = []

        for i, (images, targets) in enumerate(tqdm(dataloader)):
            images = [img.to(self.device) for img in images]

            with torch.no_grad():
                outputs = self.model(images)

            for image, output, target in zip(images, outputs, targets):
                keep = output["scores"] > score_thresh
                pred_boxes = output["boxes"][keep].cpu()
                pred_labels = output["labels"][keep].cpu()
                pred_scores = output["scores"][keep].cpu()

                predictions.append({
                    "filename": target["image_id"],
                    "pred_boxes": pred_boxes,
                    "pred_scores": pred_scores,
                    "gt_boxes": target["boxes"],
                    "pred_labels": pred_labels,
                    "gt_labels": target["labels"]
                })

            if i == 400:
                break

        save_predictions_to_json(predictions, save_path=save_dir)
        return predictions
