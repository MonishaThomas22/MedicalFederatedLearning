
from pathlib import Path
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights

# -----------------------------------------
# 1. PATHS AND SETTINGS
# -----------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed_lung_3class"
MODEL_DIR = PROJECT_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 0.001

DEVICE = torch.device("cpu")

print("Device:", DEVICE)
print("Loading lung X-ray dataset...")

# -----------------------------------------
# 2. IMAGE TRANSFORMATIONS
# -----------------------------------------

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------------------
# 3. LOAD DATASETS
# -----------------------------------------

train_dataset = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    DATA_DIR / "val",
    transform=val_transform
)

class_names = train_dataset.classes
NUM_CLASSES = len(class_names)

if NUM_CLASSES != 3:
    raise ValueError(
        f"Expected 3 classes, found {class_names}"
    )

if val_dataset.classes != class_names:
    raise ValueError("Training and validation classes do not match.")

print("Classes:", class_names)
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

# -----------------------------------------
# 4. LOAD PRETRAINED RESNET18
# -----------------------------------------

print("\nLoading ResNet18...")

try:
    model = models.resnet18(
        weights=ResNet18_Weights.DEFAULT
    )
    pretrained = True
    print("Pretrained ResNet18 loaded.")

except Exception as error:
    print("Pretrained weights unavailable:", error)
    print("Using ResNet18 without pretrained weights.")

    model = models.resnet18(weights=None)
    pretrained = False

# Freeze feature extraction layers
for parameter in model.parameters():
    parameter.requires_grad = False

# Replace final classification layer
input_features = model.fc.in_features

model.fc = nn.Linear(
    input_features,
    NUM_CLASSES
)

model = model.to(DEVICE)

# -----------------------------------------
# 5. LOSS AND OPTIMIZER
# -----------------------------------------

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=LEARNING_RATE
)

# -----------------------------------------
# 6. TRAINING
# -----------------------------------------

best_val_accuracy = 0.0

print("\n========== TRAINING STARTED ==========")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_index, (images, labels) in enumerate(train_loader):

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        total += labels.size(0)
        correct += (predictions == labels).sum().item()

        if (batch_index + 1) % 50 == 0:
            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Batch {batch_index + 1}/{len(train_loader)}"
            )

    train_loss = running_loss / total
    train_accuracy = 100 * correct / total

    # -------------------------------------
    # 7. VALIDATION
    # -------------------------------------

    model.eval()

    val_correct = 0
    val_total = 0
    val_loss_total = 0.0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss_total += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            val_total += labels.size(0)
            val_correct += (
                predictions == labels
            ).sum().item()

    val_loss = val_loss_total / val_total
    val_accuracy = 100 * val_correct / val_total

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print(f"Training loss: {train_loss:.4f}")
    print(f"Training accuracy: {train_accuracy:.2f}%")
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_accuracy:.2f}%")

    # -------------------------------------
    # 8. SAVE BEST MODEL
    # -------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "num_classes": NUM_CLASSES,
            "architecture": "ResNet18",
            "pretrained": pretrained,
            "best_validation_accuracy": best_val_accuracy
        }

        torch.save(
            checkpoint,
            MODEL_DIR / "lung_3class_best.pth"
        )

        print("Best model checkpoint saved.")

# -----------------------------------------
# 9. SAVE TRAINING INFORMATION
# -----------------------------------------

metadata = {
    "architecture": "ResNet18",
    "classes": class_names,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "pretrained": pretrained,
    "best_validation_accuracy": best_val_accuracy,
    "training_images": len(train_dataset),
    "validation_images": len(val_dataset),
    "device": str(DEVICE)
}

with open(
    MODEL_DIR / "lung_3class_metadata.json",
    "w"
) as file:
    json.dump(metadata, file, indent=4)

print("\n========== TRAINING COMPLETE ==========")
print("Best validation accuracy:", best_val_accuracy)
print("Saved model:", MODEL_DIR / "lung_3class_best.pth")
