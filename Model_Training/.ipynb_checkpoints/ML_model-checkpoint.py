import pytorch_lightning as pl
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import json
from typing import Dict, List, Union, Optional
from metrics import SegmentationMetrics  # Ensure this is correctly imported

class ResNetSegmentationPL(pl.LightningModule):
    def __init__(self, input_channels=3, num_classes=4, lr=0.001, input_types=['image']):
        """
        ResNet-based segmentation model using PyTorch Lightning.

        Args:
            input_channels: Number of input channels (e.g., 3 for RGB).
            num_classes: Number of segmentation classes.
            lr: Learning rate.
            input_types: List of input feature types (e.g., ['image', 'crop_mask']).
        """
        super(ResNetSegmentationPL, self).__init__()
        self.lr = lr
        self.input_types = input_types
        self.num_classes = num_classes

        # Load pretrained ResNet
        self.resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        # Modify first conv layer to accept different input channels
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove the fully connected layer
        self.resnet.fc = nn.Identity()

        # Segmentation head (Upsampling to match original input size)
        self.segmentation_head = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, num_classes, kernel_size=1),  # Output segmentation mask
            nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)  # Final upsampling
        )

        self.loss_fn = nn.CrossEntropyLoss()
        self.metrics = SegmentationMetrics(num_classes)  # Initialize metrics

        # Store aggregated epoch-wise metrics
        self.train_metrics = {}
        self.val_metrics = {}

        # Temporary storage for batch-wise results
        self.epoch_train_metrics = []
        self.epoch_val_metrics = []

    def forward(self, x):
        x = self.resnet(x)
        x = x.unsqueeze(-1).unsqueeze(-1)  # Reshape for segmentation head
        x = self.segmentation_head(x)
        return x  # Output: (batch_size, num_classes, 224, 224)

    def training_step(self, batch, batch_idx):
        """Defines a single training step with per-class metric computation."""

        images = torch.cat([
            batch[key].to(self.device).float() if batch[key].dim() == 4 
            else batch[key].unsqueeze(1).to(self.device).float()
            for key in self.input_types
        ], dim=1)

        masks = batch['true_mask'].to(self.device)
        outputs = self(images)
        loss = self.loss_fn(outputs, masks)

        y_true_np = masks.cpu().numpy()
        y_pred_np = torch.softmax(outputs, dim=1).detach().cpu().numpy()
        y_pred_np = np.clip(y_pred_np, 1e-7, 1.0)  
        y_pred = np.argmax(y_pred_np, axis=1)

        # Update metrics for this batch
        self.metrics.update(y_true_np, y_pred_np)
        results = self.metrics.compute()

        # Convert NaN to 0 to prevent issues
        iou_micro = np.nan_to_num(results["iou"]["micro"], nan=0.0)
        precision_micro = np.nan_to_num(results["precision"]["micro"], nan=0.0)
        recall_micro = np.nan_to_num(results["recall"]["micro"], nan=0.0)
        f1_micro = np.nan_to_num(results["f1"]["micro"], nan=0.0)

        iou_per_class = np.nan_to_num(results["iou"]["per_class"], nan=0.0)
        precision_per_class = np.nan_to_num(results["precision"]["per_class"], nan=0.0)
        recall_per_class = np.nan_to_num(results["recall"]["per_class"], nan=0.0)
        f1_per_class = np.nan_to_num(results["f1"]["per_class"], nan=0.0)

        # Store per-batch metrics for later aggregation
        if not hasattr(self, 'batch_metrics'):
            self.batch_metrics = {
                "loss": [],
                "iou_per_class": [],
                "iou_micro": [],
                "precision_per_class": [],
                "precision_micro": [],
                "recall_per_class": [],
                "recall_micro": [],
                "f1_per_class": [],
                "f1_micro": []
            }

        self.batch_metrics["loss"].append(loss.item())
        self.batch_metrics["iou_per_class"].append(iou_per_class)
        self.batch_metrics["iou_micro"].append(iou_micro)
        self.batch_metrics["precision_per_class"].append(precision_per_class)
        self.batch_metrics["precision_micro"].append(precision_micro)
        self.batch_metrics["recall_per_class"].append(recall_per_class)
        self.batch_metrics["recall_micro"].append(recall_micro)
        self.batch_metrics["f1_per_class"].append(f1_per_class)
        self.batch_metrics["f1_micro"].append(f1_micro)

        # Log metrics per batch
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_iou", iou_micro, prog_bar=True)
        self.log("train_precision", precision_micro, prog_bar=True)
        self.log("train_recall", recall_micro, prog_bar=True)
        self.log("train_f1", f1_micro, prog_bar=True)

        for class_idx in range(self.num_classes):
            self.log(f"train_iou_class_{class_idx}", iou_per_class[class_idx], prog_bar=False)
            self.log(f"train_precision_class_{class_idx}", precision_per_class[class_idx], prog_bar=False)
            self.log(f"train_recall_class_{class_idx}", recall_per_class[class_idx], prog_bar=False)
            self.log(f"train_f1_class_{class_idx}", f1_per_class[class_idx], prog_bar=False)

        return loss


    def on_train_epoch_end(self):
        """Aggregate per-batch training metrics and reset for next epoch."""

        if not hasattr(self, 'batch_metrics'):
            print("Warning: No training metrics recorded for this epoch.")
            return

        # Compute mean across all batches
        self.train_metrics = {
            "loss": np.mean(self.batch_metrics["loss"]),
            "iou_per_class": np.mean(self.batch_metrics["iou_per_class"], axis=0).tolist(),
            "iou_micro": np.mean(self.batch_metrics["iou_micro"]),
            "precision_per_class": np.mean(self.batch_metrics["precision_per_class"], axis=0).tolist(),
            "precision_micro": np.mean(self.batch_metrics["precision_micro"]),
            "recall_per_class": np.mean(self.batch_metrics["recall_per_class"], axis=0).tolist(),
            "recall_micro": np.mean(self.batch_metrics["recall_micro"]),
            "f1_per_class": np.mean(self.batch_metrics["f1_per_class"], axis=0).tolist(),
            "f1_micro": np.mean(self.batch_metrics["f1_micro"])
        }

        # Reset batch-level storage
        self.batch_metrics = {
            "loss": [],
            "iou_per_class": [],
            "iou_micro": [],
            "precision_per_class": [],
            "precision_micro": [],
            "recall_per_class": [],
            "recall_micro": [],
            "f1_per_class": [],
            "f1_micro": []
        }

        # Reset metric computations to avoid accumulation across epochs
        self.metrics.reset()


    def validation_step(self, batch, batch_idx):
        """Defines a single validation step."""
        images = torch.cat([
            batch[key].to(self.device).float() if batch[key].dim() == 4 
            else batch[key].unsqueeze(1).to(self.device).float()
            for key in self.input_types
        ], dim=1)

        masks = batch['true_mask'].to(self.device)

        outputs = self(images)
        loss = self.loss_fn(outputs, masks)

        # Convert to numpy for metric computation
        y_true_np = masks.cpu().numpy()
        y_pred_np = torch.softmax(outputs, dim=1).detach().cpu().numpy()

        self.metrics.update(y_true_np, y_pred_np)  # Update metrics

        self.epoch_val_metrics.append(loss.item())  # Store per-batch loss

        self.log("val_loss", loss, prog_bar=True)

    def on_validation_epoch_end(self):
        """Compute and log per-class metrics at the end of validation."""
        results = self.metrics.compute()

        # Store aggregated per-class metrics for validation epoch
        self.val_metrics = {
            "loss": np.mean(self.epoch_val_metrics),
            "iou_per_class": results["iou"]["per_class"].tolist(),
            "iou_micro": results["iou"]["micro"],
            "precision_per_class": results["precision"]["per_class"].tolist(),
            "precision_micro": results["precision"]["micro"],
            "recall_per_class": results["recall"]["per_class"].tolist(),
            "recall_micro": results["recall"]["micro"],
            "f1_per_class": results["f1"]["per_class"].tolist(),
            "f1_micro": results["f1"]["micro"]
        }

        # Log micro metrics
        self.log("val_iou", results['iou']['micro'], prog_bar=True)
        self.log("val_precision", results['precision']['micro'], prog_bar=True)
        self.log("val_recall", results['recall']['micro'], prog_bar=True)
        self.log("val_f1", results['f1']['micro'], prog_bar=True)

        # Reset for next epoch
        self.metrics.reset()
        self.epoch_val_metrics = []

    def configure_optimizers(self):
        """Defines optimizer and learning rate scheduler."""
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def save_metrics(self, filename="training_results.json"):
        """Save per-class training and validation metrics to a JSON file."""
        results = {
            "train": self.train_metrics,
            "validation": self.val_metrics
        }
        with open(filename, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Metrics saved to {filename}")
