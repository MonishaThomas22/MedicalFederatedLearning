
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent

TEST_DIR = PROJECT_DIR / "data" / "processed_lung_3class" / "test"

MODEL_PATH = (
    PROJECT_DIR / "models" / "lung_hospital_federated_model.pth"
)

DEVICE = torch.device("cpu")
BATCH_SIZE = 32

# Test transformations
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

print("Loading test dataset...")

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

class_names = test_dataset.classes

print("Classes:", class_names)
print("Test images:", len(test_dataset))

# Load federated model
print("\nLoading federated global model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    checkpoint["num_classes"]
)

model.load_state_dict(checkpoint["model_state_dict"])

model.to(DEVICE)
model.eval()

print("Federated model loaded successfully.")

# Evaluate
all_predictions = []
all_labels = []

print("\nEvaluating federated model...")

with torch.no_grad():
    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.tolist())

# Results
accuracy = accuracy_score(all_labels, all_predictions)

print("\n========== FEDERATED TEST RESULTS ==========")

print(f"Test accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        all_labels,
        all_predictions
    )
)

print("\nFederated aggregation:", checkpoint["aggregation"])
print("Communication rounds:", checkpoint["communication_rounds"])
print("Hospital sample counts:", checkpoint["client_sample_counts"])

print("\n========== EVALUATION COMPLETE ==========")

