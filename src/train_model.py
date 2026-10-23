
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"

# Training settings
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.0001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print("Device:", DEVICE)
    print("Loading chest X-ray datasets...")

    # Convert images to tensors and resize them for ResNet18
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    train_dir = DATA_DIR / "train"
    val_dir = DATA_DIR / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            "Training or validation folders are missing. "
            "Run src/prepare_data.py first."
        )

    train_dataset = datasets.ImageFolder(
        root=str(train_dir),
        transform=transform
    )
    val_dataset = datasets.ImageFolder(
        root=str(val_dir),
        transform=transform
    )

    print("Class names:", train_dataset.classes)
    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # Load ResNet18 with pretrained ImageNet weights
    print("Loading ResNet18...")
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    # Replace the final layer with two output classes
    model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
    model = model.to(DEVICE)

    # Compensate for the imbalance between Normal and TB images
    class_counts = torch.bincount(
        torch.tensor(train_dataset.targets),
        minlength=len(train_dataset.classes)
    ).float()

    class_weights = class_counts.sum() / (
        len(class_counts) * class_counts.clamp(min=1)
    )
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(DEVICE)
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_val_accuracy = 0.0

    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                predictions = outputs.argmax(dim=1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

        train_accuracy = 100 * train_correct / max(train_total, 1)
        val_accuracy = 100 * val_correct / max(val_total, 1)

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(f"Train loss: {train_loss / max(train_total, 1):.4f}")
        print(f"Train accuracy: {train_accuracy:.2f}%")
        print(f"Validation loss: {val_loss / max(val_total, 1):.4f}")
        print(f"Validation accuracy: {val_accuracy:.2f}%")

        # Save the best model based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": train_dataset.classes,
                    "validation_accuracy": val_accuracy,
                },
                MODEL_DIR / "best_model.pth"
            )
            print("Best model saved.")

    print("\nTraining complete!")
    print("Best validation accuracy:", f"{best_val_accuracy:.2f}%")
    print("Model location:", MODEL_DIR / "best_model.pth")
    print(
        "\nThis is a baseline centralized model. "
        "Federated learning and privacy protection come later."
    )


if __name__ == "__main__":
    main()
