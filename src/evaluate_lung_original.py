
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent

TEST_DIR = PROJECT_DIR / "data" / "processed_lung_3class" / "test"

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "privacy_backup"
    / "lung_hospital_federated_model.pth"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():

    print("\n========== ORIGINAL NON-DP LUNG MODEL EVALUATION ==========")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    test_dataset = datasets.ImageFolder(
        str(TEST_DIR),
        transform=transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint.get("global_model_state_dict"),
    )

    if state_dict is None:
        raise KeyError(
            f"No model weights found. Keys: {list(checkpoint.keys())}"
        )

    class_names = checkpoint.get(
        "class_names",
        test_dataset.classes,
    )

    if list(class_names) != test_dataset.classes:
        raise ValueError("Model classes do not match test dataset.")

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        len(class_names),
    )

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    actual = []
    predicted = []

    with torch.no_grad():
        for images, labels in test_loader:

            outputs = model(images.to(DEVICE))

            predictions = outputs.argmax(dim=1)

            actual.extend(labels.tolist())
            predicted.extend(predictions.cpu().tolist())

    print("\nTest images:", len(test_dataset))
    print("Classes:", class_names)

    print("\nAccuracy:", round(
        accuracy_score(actual, predicted) * 100, 2
    ), "%")

    print("Macro precision:", round(
        precision_score(actual, predicted, average="macro",
                        zero_division=0) * 100, 2
    ), "%")

    print("Macro recall:", round(
        recall_score(actual, predicted, average="macro",
                     zero_division=0) * 100, 2
    ), "%")

    print("Macro F1:", round(
        f1_score(actual, predicted, average="macro",
                 zero_division=0) * 100, 2
    ), "%")

    print("\nConfusion matrix:")
    print(confusion_matrix(
        actual,
        predicted,
        labels=list(range(len(class_names))),
    ))

    print("\nClassification report:")
    print(classification_report(
        actual,
        predicted,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    ))


if __name__ == "__main__":
    main()
