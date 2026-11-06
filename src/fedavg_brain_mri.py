
from pathlib import Path
import copy
import random

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data" / "processed_brain_mri"
MODEL_DIR = PROJECT_DIR / "models"

OUTPUT_FILE = MODEL_DIR / "brain_mri_4class_federated_model.pth"

DEVICE = torch.device("cpu")

NUM_CLIENTS = 3
COMMUNICATION_ROUNDS = 3
LOCAL_EPOCHS = 1

BATCH_SIZE = 32
LEARNING_RATE = 0.001

SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. IMAGE TRANSFORMATIONS
# ==========================================

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

evaluation_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# 3. LOAD THREE CLIENT DATASETS
# ==========================================

client_loaders = []
client_sample_counts = []

class_names = None

print("\nBRAIN MRI FEDERATED LEARNING")
print("=" * 55)

print("Device:", DEVICE)
print("Number of clients:", NUM_CLIENTS)
print("Communication rounds:", COMMUNICATION_ROUNDS)


for client_id in range(1, NUM_CLIENTS + 1):

    client_path = DATA_DIR / f"client_{client_id}"

    dataset = datasets.ImageFolder(
        client_path,
        transform=train_transform
    )

    if class_names is None:
        class_names = dataset.classes

    elif dataset.classes != class_names:
        raise ValueError(
            f"Class mismatch in client {client_id}"
        )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    client_loaders.append(loader)
    client_sample_counts.append(len(dataset))

    print(
        f"Client {client_id}: "
        f"{len(dataset)} training images"
    )


num_classes = len(class_names)

if num_classes != 4:
    raise ValueError(
        f"Expected 4 classes, but found {num_classes}"
    )

print("\nClasses:", class_names)
print("Total client training images:", sum(client_sample_counts))


# ==========================================
# 4. LOAD VALIDATION DATA
# ==========================================

validation_dataset = datasets.ImageFolder(
    DATA_DIR / "validation",
    transform=evaluation_transform
)

if validation_dataset.classes != class_names:
    raise ValueError("Validation class order does not match clients.")

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Validation images:", len(validation_dataset))


# ==========================================
# 5. CREATE SHARED PRETRAINED RESNET18
# ==========================================

def create_initial_model():

    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )

    # Freeze pretrained feature extractor.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Replace classifier with four MRI classes.
    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model.to(DEVICE)


global_model = create_initial_model()

print("\nImageNet-pretrained ResNet18 created.")
print("Only the final classification layer is trainable.")


# ==========================================
# 6. LOCAL TRAINING AT EACH CLIENT
# ==========================================

def train_local_model(model, loader, client_id, round_id):

    model.train()

    # Keep frozen BatchNorm layers unchanged.
    for module in model.modules():

        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    optimizer = optim.Adam(
        model.fc.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_images = 0

    for epoch in range(LOCAL_EPOCHS):

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size

            predictions = outputs.argmax(dim=1)

            total_correct += (
                predictions == labels
            ).sum().item()

            total_images += batch_size

    average_loss = total_loss / total_images

    accuracy = (
        100.0 * total_correct / total_images
    )

    print(
        f"Client {client_id} | "
        f"Round {round_id} | "
        f"Loss: {average_loss:.4f} | "
        f"Accuracy: {accuracy:.2f}%"
    )

    return copy.deepcopy(model.state_dict())


# ==========================================
# 7. FEDAVG AGGREGATION
# ==========================================

def aggregate_models(client_states, sample_counts):

    total_samples = sum(sample_counts)

    global_state = {}

    for key in client_states[0]:

        tensors = [
            state[key] for state in client_states
        ]

        if torch.is_floating_point(tensors[0]):

            weighted_average = torch.zeros_like(
                tensors[0]
            )

            for tensor, count in zip(
                tensors,
                sample_counts
            ):

                weighted_average += (
                    tensor * (count / total_samples)
                )

            global_state[key] = weighted_average

        else:

            # Preserve integer model buffers.
            global_state[key] = tensors[0].clone()

    return global_state


# ==========================================
# 8. VALIDATION
# ==========================================

def evaluate_validation(model):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    accuracy = 100.0 * correct / total

    return accuracy


# ==========================================
# 9. FEDERATED TRAINING
# ==========================================

for round_id in range(1, COMMUNICATION_ROUNDS + 1):

    print("\n" + "=" * 55)

    print(
        f"BRAIN MRI FEDERATED ROUND {round_id}"
    )

    print("=" * 55)

    client_states = []

    for client_id, loader in enumerate(
        client_loaders,
        start=1
    ):

        # Each client receives the latest global model.
        local_model = models.resnet18(weights=None)

        local_model.fc = nn.Linear(
            local_model.fc.in_features,
            num_classes
        )

        local_model.load_state_dict(
            global_model.state_dict()
        )

        local_model.to(DEVICE)

        # Freeze feature extractor.
        for parameter in local_model.parameters():
            parameter.requires_grad = False

        for parameter in local_model.fc.parameters():
            parameter.requires_grad = True

        local_state = train_local_model(
            local_model,
            loader,
            client_id,
            round_id
        )

        client_states.append(local_state)

        del local_model

    # Central server performs weighted FedAvg.
    aggregated_state = aggregate_models(
        client_states,
        client_sample_counts
    )

    global_model.load_state_dict(
        aggregated_state
    )

    print(
        f"\nFedAvg aggregation completed for round {round_id}."
    )

    validation_accuracy = evaluate_validation(
        global_model
    )

    print(
        f"Global validation accuracy: "
        f"{validation_accuracy:.2f}%"
    )


# ==========================================
# 10. SAVE FINAL GLOBAL MODEL
# ==========================================

torch.save(
    {
        "model_state_dict": global_model.state_dict(),

        "class_names": class_names,

        "num_classes": num_classes,

        "architecture": "ResNet18",

        "num_clients": NUM_CLIENTS,

        "client_sample_counts": client_sample_counts,

        "communication_rounds": COMMUNICATION_ROUNDS,

        "local_epochs": LOCAL_EPOCHS,

        "aggregation": "FedAvg",

        "initialization": "ImageNet pretrained weights",

        "training_mode": (
            "Frozen feature extractor; "
            "trainable classifier"
        ),

        "data_distribution": (
            "Balanced class-wise IID simulation"
        ),

        "differential_privacy": False,

        "secure_aggregation": False
    },

    OUTPUT_FILE
)


# ==========================================
# 11. TRAINING SUMMARY
# ==========================================

print("\n" + "=" * 55)

print("BRAIN MRI FEDERATED TRAINING COMPLETED!")

print("=" * 55)

print("Classes:", class_names)

print("Clients:", NUM_CLIENTS)

print("Communication rounds:", COMMUNICATION_ROUNDS)

print("Client sample counts:", client_sample_counts)

print("Aggregation: FedAvg")

print("Final validation accuracy:",
      f"{evaluate_validation(global_model):.2f}%")

print("Saved global model:", OUTPUT_FILE)

print("\nThis is a simulated three-client FedAvg experiment.")

print("Differential privacy: Not enabled.")

print("Secure aggregation: Not enabled.")

print("This model is an educational research prototype.")
