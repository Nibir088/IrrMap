import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import yaml
import torch.optim as optim
import torch.nn.functional as F
from dataset import ImageMaskDataset
from ML_model import ResNetSegmentation


loss_fn = nn.CrossEntropyLoss()

# Define optimizer function
def get_optimizer(model, lr=0.001):
    return optim.Adam(model.parameters(), lr=lr)

# Define IoU metric
def iou_score(preds, labels, num_classes):
    """
    Compute IoU (Intersection over Union) for multi-class segmentation.

    Args:
        preds: Model predictions (logits), shape: (batch, num_classes, H, W)
        labels: Ground truth, shape: (batch, H, W) with class indices
        num_classes: Number of classes

    Returns:
        Mean IoU score across all classes
    """
    preds = torch.argmax(preds, dim=1)  # Convert logits to class indices

    iou_per_class = []
    for cls in range(num_classes):
        pred_mask = (preds == cls)
        true_mask = (labels == cls)

        intersection = (pred_mask & true_mask).float().sum()
        union = (pred_mask | true_mask).float().sum()

        iou = (intersection + 1e-10) / (union + 1e-10)  # Avoid division by zero
        iou_per_class.append(iou)

    return sum(iou_per_class) / num_classes  # Mean IoU across all classes


# Training function
def train(model, train_loader, test_loader, num_epochs=10, device='cuda', input_types=['rgb'], num_classes = 4):
    """
    Train the model using a specific input configuration.

    Args:
        model: The ResNet-based segmentation model.
        train_loader: DataLoader for training.
        test_loader: DataLoader for testing.
        num_epochs: Number of training epochs.
        device: 'cuda' or 'cpu'.
        input_types: List of input feature types (e.g., ['rgb', 'crop_mask', 'ndvi']).
    """
    model.to(device)
    optimizer = get_optimizer(model)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            # Concatenate selected input types dynamically
            images = torch.cat([batch[key].to(device) for key in input_types], dim=1)
            masks = batch['true_mask'].to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Evaluate model after each epoch
        model.eval()
        with torch.no_grad():
            iou_scores = []
            for batch in test_loader:
                images = torch.cat([batch[key].to(device) for key in input_types], dim=1)
                masks = batch['true_mask'].to(device)

                outputs = model(images)
                iou_scores.append(iou_score(outputs, masks, num_classes).item())

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}, IoU: {sum(iou_scores)/len(iou_scores):.4f}")

# Initialize and train model
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a ResNet-based segmentation model")
    parser.add_argument('--data_folder', type=str, default='LandSat/IrrMap_combined.yaml', help="Path to the YAML file containing dataset paths")
    parser.add_argument('--input_types', nargs='+', default=['image'], help="List of input features (e.g., 'rgb', 'ndvi', 'crop_mask', 'land_mask')")
    parser.add_argument('--num_epochs', type=int, default=1, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to train on (cpu or cuda)")
    parser.add_argument('--source', type=str, default="landsat")

    args = parser.parse_args()
    
    with open(args.data_folder, "r") as file:
        data_path = yaml.safe_load(file)

    # Define train/test image and mask paths
    train_image_paths = data_path['train']['patches']
    train_mask_paths = data_path['train']['masks']
    test_image_paths = data_path['test']['patches']
    test_mask_paths = data_path['test']['masks']

    # Define dataset parameters
    image_size = (224, 224)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create datasets
    train_dataset = ImageMaskDataset(
        image_paths=train_image_paths,
        mask_paths=train_mask_paths,
        states=[],
        image_size=image_size,
        transform=False,
        gamma_value=1.3,
        is_binary=False,
        agri_indices=['ndvi'],
        source = args.source
    )

    test_dataset = ImageMaskDataset(
        image_paths=test_image_paths,
        mask_paths=test_mask_paths,
        states=[],
        image_size=image_size,
        transform=False,
        gamma_value=1.3,
        is_binary=False,
        agri_indices=['ndvi'],
        source = args.source
    )

    # Update batch size if provided
    batch_size = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Train model with specified input types
    model = ResNetSegmentation(input_channels=len(args.input_types) + 2, num_classes=4)
    train(model, train_loader, test_loader, num_epochs=args.num_epochs, device=args.device, input_types=args.input_types)
    
    # Save trained model with parameters
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
        'input_channels': len(args.input_types) + 2,  # Number of input channels used
        'input_types': args.input_types,  # List of input features
        'optimizer_state_dict': optimizer.state_dict(),
        'num_epochs': args.num_epochs,
        'learning_rate': args.lr
    }, f"{source}_input_types.pth")

    print("Model saved successfully with parameters!")

