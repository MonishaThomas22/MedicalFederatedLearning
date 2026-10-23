
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
TEST_DIR = PROJECT_DIR / "data" / "processed" / "test"

# Evaluate the federated model instead of the centralized model
MODEL_PATH = PROJECT_DIR / "models" / "federated_model.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print("Device:", DEVICE)
    print("Evaluating the FEDERATED model")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Federated model not found: {MODEL_PATH}"
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

    print("Class names:", test_dataset.classes)
    print("Test images:", len(test_dataset))

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    # Support the checkpoint format used by the federated script
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "global_model_state_dict" in checkpoint:
        state_dict = checkpoint["global_model_state_dict"]
    else:
        raise KeyError(
            "Could not find model weights in the checkpoint. "
            f"Available keys: {list(checkpoint.keys())}"
        )

    class_names = checkpoint.get(
        "class_names",
        test_dataset.classes,
    )

    if class_names != test_dataset.classes:
        raise ValueError(
            f"Model classes {class_names} do not match "
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

    print("\nEvaluating saved federated model on test images...")

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            all_labels.extend(labels.tolist())
            all_predictions.extend(predictions.cpu().tolist())

    accuracy = accuracy_score(all_labels, all_predictions)
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

    print("\n========== FEDERATED MODEL TEST RESULTS ==========")
    print(f"Test accuracy:   {accuracy * 100:.2f}%")
    print(f"Macro precision: {precision * 100:.2f}%")
    print(f"Macro recall:    {recall * 100:.2f}%")
    print(f"Macro F1-score:  {f1 * 100:.2f}%")

    print("\nConfusion matrix:")
    print("Rows = actual class; columns = predicted class")
    print("Class order:", class_names)
    print(confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(len(class_names))),
    ))

    print("\nDetailed classification report:")
    print(classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    ))

    print("\nFederated model evaluation complete.")
    print("Evaluated checkpoint:", MODEL_PATH)


if __name__ == "__main__":
    main()
