
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


# ==========================================
# 1. CONFIGURATION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data" / "processed_brain_mri"
MODEL_FILE = (
    PROJECT_DIR
    / "models"
    / "brain_mri_4class_federated_model.pth"
)

DEVICE = torch.device("cpu")
BATCH_SIZE = 32


# ==========================================
# 2. IMAGE PREPROCESSING
# ==========================================

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# 3. LOAD TEST DATASET
# ==========================================

test_dataset = datasets.ImageFolder(
    DATA_DIR / "testing",
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("\nBRAIN MRI MODEL EVALUATION")
print("=" * 55)

print("Device:", DEVICE)
print("Testing images:", len(test_dataset))
print("Classes:", test_dataset.classes)


# ==========================================
# 4. LOAD SAVED FEDERATED MODEL
# ==========================================

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE,
    weights_only=False
)

class_names = checkpoint["class_names"]

if test_dataset.classes != class_names:
    raise ValueError(
        "Test dataset classes do not match model classes."
    )

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    len(class_names)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)
model.eval()

print("\nFederated ResNet18 model loaded successfully.")
print("Aggregation:", checkpoint["aggregation"])
print("Clients:", checkpoint["num_clients"])


# ==========================================
# 5. MAKE TEST PREDICTIONS
# ==========================================

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )


# ==========================================
# 6. CALCULATE METRICS
# ==========================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

print("\nTEST ACCURACY")
print("=" * 55)

print(f"Accuracy: {accuracy * 100:.2f}%")


# ==========================================
# 7. CLASSIFICATION REPORT
# ==========================================

print("\nCLASSIFICATION REPORT")
print("=" * 55)

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)


# ==========================================
# 8. CONFUSION MATRIX
# ==========================================

matrix = confusion_matrix(
    all_labels,
    all_predictions
)

print("\nCONFUSION MATRIX")
print("=" * 55)

print("Class order:", class_names)

print(matrix)


# ==========================================
# 9. FINAL SUMMARY
# ==========================================

print("\n" + "=" * 55)

print("BRAIN MRI EVALUATION COMPLETED")

print("=" * 55)

print("Architecture: ResNet18")
print("Aggregation: FedAvg")
print("Clients: 3")
print("Communication rounds:", checkpoint["communication_rounds"])
print("Test images:", len(test_dataset))
print(f"Test accuracy: {accuracy * 100:.2f}%")

print("\nThis is an educational research prototype.")
print("It is not a clinically validated diagnostic system.")
