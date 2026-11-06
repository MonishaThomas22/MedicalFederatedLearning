
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
    / "lung_hospital_federated_model_dp.pth"
)

IMAGE_SIZE = 224
BATCH_SIZE = 32

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def main():

    print("\n========== DP FEDERATED LUNG MODEL EVALUATION ==========")

    print("Device:", DEVICE)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    if not TEST_DIR.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_DIR}"
        )

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    test_dataset = datasets.ImageFolder(
        root=str(TEST_DIR),
        transform=transform,
    )

    if len(test_dataset) == 0:
        raise ValueError("No test images found.")

    print("\nTest classes:", test_dataset.classes)
    print("Total test images:", len(test_dataset))

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # This is our own locally generated checkpoint.
    # weights_only=False is needed because its metadata
    # contains NumPy scalar values.
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif "global_model_state_dict" in checkpoint:
        state_dict = checkpoint["global_model_state_dict"]

    else:
        raise KeyError(
            "Model weights not found. Available keys: "
            + str(list(checkpoint.keys()))
        )

    class_names = checkpoint.get(
        "class_names",
        test_dataset.classes,
    )

    if list(class_names) != test_dataset.classes:
        raise ValueError(
            f"Checkpoint classes {class_names} do not match "
            f"test classes {test_dataset.classes}"
        )

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        len(class_names),
    )

    model.load_state_dict(state_dict)

    model = model.to(DEVICE)
    model.eval()

    all_labels = []
    all_predictions = []

    print("\nEvaluating the saved DP federated model...")

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            all_labels.extend(labels.tolist())

            all_predictions.extend(
                predictions.cpu().tolist()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    print("\n========== TEST RESULTS ==========")

    print(f"Test accuracy:   {accuracy * 100:.2f}%")
    print(f"Macro precision: {precision * 100:.2f}%")
    print(f"Macro recall:    {recall * 100:.2f}%")
    print(f"Macro F1-score:  {f1 * 100:.2f}%")

    print("\nConfusion Matrix")
    print("Rows = Actual; Columns = Predicted")
    print("Class order:", class_names)

    print(
        confusion_matrix(
            all_labels,
            all_predictions,
            labels=list(range(len(class_names))),
        )
    )

    print("\nDetailed Classification Report")

    print(
        classification_report(
            all_labels,
            all_predictions,
            labels=list(range(len(class_names))),
            target_names=class_names,
            zero_division=0,
        )
    )

    print("\n========== PRIVACY INFORMATION ==========")

    print(
        "Differential Privacy:",
        checkpoint.get("differential_privacy", False),
    )

    print(
        "Privacy mechanism:",
        checkpoint.get("privacy_mechanism", "Not available"),
    )

    print(
        "Privacy accountant:",
        checkpoint.get("privacy_accountant", "Not available"),
    )

    print(
        "Epsilon per hospital:",
        checkpoint.get("epsilon_per_hospital", "Not available"),
    )

    print(
        "Secure aggregation:",
        checkpoint.get("secure_aggregation", False),
    )

    print(
        "Secure aggregation mode:",
        checkpoint.get("secure_aggregation_mode", "Not available"),
    )

    print(
        "Production secure aggregation:",
        checkpoint.get("production_secure_aggregation", False),
    )

    print("\nEvaluated checkpoint:", MODEL_PATH)

    print("\nDP federated model evaluation completed!")


if __name__ == "__main__":
    main()
