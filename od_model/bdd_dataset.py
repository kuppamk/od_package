import json
import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Define transformation
transform_list = transforms.Compose([
    transforms.ToTensor()
])


class BDDDetectionDataset(Dataset):
    """Custom Dataset for BDD100K object detection."""

    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        category_map: dict = None,
        transforms=None,
        evaluation: bool = False
    ):
        """Initializes the dataset.

        Args:
            image_dir (str): Path to image directory.
            annotation_file (str): Path to annotation JSON.
            category_map (dict, optional): Mapping of category to class index.
            transforms (callable, optional): Transformations to apply to the image.
            evaluation (bool): Whether evaluation mode is enabled.
        """
        self.image_dir = image_dir
        self.transforms = transforms
        self.evaluation = evaluation

        with open(annotation_file) as f:
            self.annotations = json.load(f)

        self.category_map = category_map or self._generate_category_map()
        self.image_annotations = self._organize_annotations()

    def _generate_category_map(self) -> dict:
        """Generates category-to-index mapping, skipping non-object classes.

        Returns:
            dict: Mapping of category names to indices.
        """
        categories = set()
        for item in self.annotations:
            for label in item.get("labels", []):
                categories.add(label["category"])

        categories = sorted(list(categories))
        for cat in ["lane", "drivable area"]:
            if cat in categories:
                categories.remove(cat)

        return {cat: idx + 1 for idx, cat in enumerate(categories)}  # 0 = background

    def _organize_annotations(self) -> list:
        """Organizes and filters annotations per image.

        Returns:
            list: List of dictionaries containing valid image annotations.
        """
        image_annots = []

        for item in self.annotations:
            filename = item["name"]
            labels = item.get("labels", [])

            boxes = []
            labels_idx = []

            for label in labels:
                if "box2d" not in label:
                    continue

                box = label["box2d"]
                x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]

                if x2 <= x1 or y2 <= y1:
                    continue

                boxes.append([x1, y1, x2, y2])
                labels_idx.append(self.category_map[label["category"]])

            if len(boxes) == 0:
                continue

            image_annots.append({
                "filename": filename,
                "boxes": boxes,
                "labels": labels_idx
            })

        return image_annots

    def __getitem__(self, idx: int):
        """Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (transformed image, target dict with boxes and labels)
        """
        data = self.image_annotations[idx]
        img_path = os.path.join(self.image_dir, data["filename"])
        img = Image.open(img_path).convert("RGB")

        boxes = torch.tensor(data["boxes"], dtype=torch.float32)
        labels = torch.tensor(data["labels"], dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels
        }

        if self.evaluation:
            target["image_id"] = data["filename"]

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self) -> int:
        """Returns the size of the dataset."""
        return len(self.image_annotations)

    def get_category_map(self) -> dict:
        """Returns the category map."""
        return self.category_map


class DataPipeline:
    """Handles loading of training and validation datasets and dataloaders."""

    def __init__(
        self,
        train_images_path: str,
        train_labels_json_path: str,
        valid_images_path: str,
        valid_labels_json_path: str,
        evaluation: bool = False
    ):
        """Initializes training and validation datasets and builds category map.

        Args:
            train_images_path (str): Path to training images.
            train_labels_json_path (str): Path to training labels JSON.
            valid_images_path (str): Path to validation images.
            valid_labels_json_path (str): Path to validation labels JSON.
            evaluation (bool): Whether to enable evaluation mode.
        """
        self.train_dataset = BDDDetectionDataset(
            train_images_path,
            train_labels_json_path,
            category_map=None,
            transforms=transform_list,
            evaluation=evaluation
        )

        self.val_dataset = BDDDetectionDataset(
            valid_images_path,
            valid_labels_json_path,
            category_map=self.train_dataset.get_category_map(),
            transforms=transform_list,
            evaluation=evaluation
        )

        self.category_map = self.train_dataset.get_category_map()
        self.num_classes = len(self.category_map) + 1  # +1 for background

    def create_dataloaders(self):
        """Creates PyTorch dataloaders for training and validation.

        Returns:
            tuple: (train_dataloader, valid_dataloader)
        """
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=16,
            shuffle=True,
            collate_fn=lambda x: tuple(zip(*x))
        )

        valid_dataloader = DataLoader(
            self.val_dataset,
            batch_size=16,
            shuffle=True,
            collate_fn=lambda x: tuple(zip(*x))
        )

        return train_dataloader, valid_dataloader
