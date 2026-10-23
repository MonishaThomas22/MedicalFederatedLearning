
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"

DEVICE = torch.device("cpu")

NUM_HOSPITALS = 3
BATCH_SIZE = 16
LOCAL_EPOCHS = 3
LEARNING_RATE = 0.0001
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load the training dataset
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

train_dataset = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=transform,
)

class_names = train_dataset.classes
num_classes = len(class_names)

print("Device:", DEVICE)
print("Classes:", class_names)
print("Total training images:", len(train_dataset))
print("Simulated hospitals:", NUM_HOSPITALS)


# --------------------------------------------------
# 3. Divide training images among hospitals
# --------------------------------------------------

indices = list(range(len(train_dataset)))
random.shuffle(indices)

hospital_indices = [
    indices[i::NUM_HOSPITALS]
    for i in range(NUM_HOSPITALS)
]


# --------------------------------------------------
# 4. Create a fresh model for each hospital
# --------------------------------------------------

def create_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes,
    )
    return model.to(DEVICE)


# --------------------------------------------------
# 5. Train each hospital independently
# --------------------------------------------------

for hospital_id, local_indices in enumerate(
    hospital_indices, start=1
):
    print(f"\n========== HOSPITAL {hospital_id} ==========")

    local_dataset = Subset(
        train_dataset,
        local_indices,
    )

    local_loader = DataLoader(
        local_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    # Each hospital starts with a new, randomly initialized model.
    # No centralized or other hospital model is loaded here.
    local_model = create_model()

    optimizer = optim.Adam(
        local_model.parameters(),
        lr=LEARNING_RATE,
    )

    criterion = nn.CrossEntropyLoss()

    print("Local training images:", len(local_dataset))
    print("Training this hospital's model independently...")

    for epoch in range(1, LOCAL_EPOCHS + 1):
        local_model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in local_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = local_model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        epoch_loss = total_loss / total
        epoch_accuracy = 100.0 * correct / total

        print(
            f"Epoch {epoch}/{LOCAL_EPOCHS} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Training accuracy: {epoch_accuracy:.2f}%"
        )

    # Save this hospital's model separately.
    model_path = (
        MODEL_DIR / f"hospital_{hospital_id}_model.pth"
    )

    torch.save(
        {
            "model_state_dict": local_model.state_dict(),
            "class_names": class_names,
            "hospital_id": hospital_id,
            "training_images": len(local_dataset),
            "local_epochs": LOCAL_EPOCHS,
        },
        model_path,
    )

    print("Saved local model to:", model_path)

    del local_model
    del optimizer


# --------------------------------------------------
# 6. Training summary
# --------------------------------------------------

print("\n========================================")
print("ALL HOSPITALS FINISHED LOCAL TRAINING")
print("========================================")

for hospital_id in range(1, NUM_HOSPITALS + 1):
    model_path = (
        MODEL_DIR / f"hospital_{hospital_id}_model.pth"
    )
    print(f"Hospital {hospital_id}: {model_path}")

print("\nThese are three separate local model checkpoints.")
print("FedAvg aggregation will be performed in the next stage.")
print("Note: Hospital clients are simulated using one public dataset.")
