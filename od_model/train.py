import os
import torch
from torch.utils.data import DataLoader


class TrainModel:
    """
    Trainer class for PyTorch object detection models with training and validation.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device, lr: float = 0.005):
        """Initializes the training class.

        Args:
            model (torch.nn.Module): The object detection model to be trained.
            device (torch.device): The device to run the training on (CPU or CUDA).
            lr (float): Learning rate for the optimizer. Defaults to 0.005.
        """
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.optim = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=0.0005,
        )

    def train_on_epoch(self, train_dataloader: DataLoader) -> float:
        """Runs one epoch of training.

        Args:
            train_dataloader (DataLoader): DataLoader for training data.

        Returns:
            float: Average training loss over the epoch.
        """
        self.model.train()
        train_loss = 0.0

        for images, targets in train_dataloader:
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            self.optim.zero_grad()
            losses.backward()
            self.optim.step()

            train_loss += losses.item()

        avg_train_loss = train_loss / len(train_dataloader)
        return avg_train_loss

    def validate_after_epoch(self, valid_dataloader: DataLoader) -> float:
        """Runs validation after each epoch.

        Args:
            valid_dataloader (DataLoader): DataLoader for validation data.

        Returns:
            float: Average validation loss over the dataset.
        """
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, targets in valid_dataloader:
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()

        avg_val_loss = val_loss / len(valid_dataloader)
        return avg_val_loss

    def train(
        self,
        train_dataloader: DataLoader,
        num_epochs: int,
        val_dataloader: DataLoader,
        save_path: str,
    ) -> None:
        """Main training loop with validation and model checkpointing.

        Args:
            train_dataloader (DataLoader): Dataloader for training data.
            num_epochs (int): Number of epochs to train for.
            val_dataloader (DataLoader): Dataloader for validation data.
            save_path (str): Path to save the best model weights.
        """
        best_val_loss = float("inf")

        for epoch in range(num_epochs):
            avg_train_loss = self.train_on_epoch(train_dataloader)
            avg_val_loss = self.validate_after_epoch(val_dataloader)

            print(
                f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f}"
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(self.model.state_dict(), save_path)
                print(
                    f"Saved best model at epoch {epoch + 1} with val loss "
                    f"{best_val_loss:.4f}"
                )
