import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_model(model_path: bool, num_classes: int) -> torch.nn.Module:
    """Initializes a Faster R-CNN model with a custom classification head.

    Args:
        model_path (bool): If True, loads ImageNet pre-trained weights.
        num_classes (int): Number of output classes (including background).

    Returns:
        torch.nn.Module: The customized Faster R-CNN model.
    """
    model = fasterrcnn_resnet50_fpn(pretrained=model_path)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
