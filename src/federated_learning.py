
from pathlib import Path
import copy
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
MODEL_PATH = PROJECT_DIR / "models" / "best_model.pth"

DEVICE = torch.device("cpu")
NUM_CLIENTS = 3
FEDERATED_ROUNDS = 3
LOCAL_EPOCHS = 1
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)


# --------------------------------------------------
# 2. Load training and validation datasets
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

val_dataset = datasets.ImageFolder(
    DATA_DIR / "val",
    transform=transform,
)

class_names = train_dataset.classes
num_classes = len(class_names)

print("Device:", DEVICE)
print("Classes:", class_names)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print(f"Simulated hospital clients: {NUM_CLIENTS}")


# --------------------------------------------------
# 3. Divide training images among three clients
# --------------------------------------------------

indices = list(range(len(train_dataset)))
random.shuffle(indices)

client_indices = [
    indices[i::NUM_CLIENTS]
    for i in range(NUM_CLIENTS)
]

client_loaders = []

for client_id, client_data in enumerate(client_indices, start=1):
    subset = Subset(train_dataset, client_data)

    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    client_loaders.append(loader)

    print(
        f"Hospital {client_id}: "
        f"{len(subset)} local training images"
    )


# --------------------------------------------------
# 4. Create the ResNet18 model
# --------------------------------------------------

def create_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model.to(DEVICE)


global_model = create_model()

# Start federated training from the saved baseline model.
checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True,
)

global_model.load_state_dict(checkpoint["model_state_dict"])

print("\nLoaded the saved baseline model.")
print("Starting federated training...")


# --------------------------------------------------
# 5. Train one hospital's local model
# --------------------------------------------------

def train_local_model(model, loader):
    model.train()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    criterion = nn.CrossEntropyLoss()

    for _ in range(LOCAL_EPOCHS):
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

    return model.state_dict()


# --------------------------------------------------
# 6. FedAvg: combine local model parameters
# --------------------------------------------------

def fedavg(client_states, client_sizes):
    total_images = sum(client_sizes)
    averaged_state = {}

    for key in client_states[0]:
        if torch.is_floating_point(client_states[0][key]):
            averaged_state[key] = torch.zeros_like(
                client_states[0][key]
            )

            for state, size in zip(client_states, client_sizes):
                averaged_state[key] += (
                    state[key] * (size / total_images)
                )
        else:
            # Copy non-floating state values from the first client.
            averaged_state[key] = client_states[0][key].clone()

    return averaged_state


# --------------------------------------------------
# 7. Evaluate the global model
# --------------------------------------------------

def evaluate_model(model, loader):
    model.eval()

    correct = 0
    total = 0
    total_loss = 0.0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total
    average_loss = total_loss / total

    return average_loss, accuracy


# --------------------------------------------------
# 8. Run federated learning rounds
# --------------------------------------------------

for round_number in range(1, FEDERATED_ROUNDS + 1):
    print(f"\n========== FEDERATED ROUND {round_number} ==========")

    client_states = []
    client_sizes = []

    global_state = global_model.state_dict()

    for client_id, loader in enumerate(client_loaders, start=1):
        print(f"Training Hospital {client_id}...")

        # Each client begins from the current global model.
        local_model = create_model()
        local_model.load_state_dict(global_state)

        local_state = train_local_model(local_model, loader)

        client_states.append({
            key: value.detach().cpu().clone()
            for key, value in local_state.items()
        })

        client_sizes.append(len(loader.dataset))

        del local_model

    # The server combines model parameters, weighted by client size.
    aggregated_state = fedavg(client_states, client_sizes)

    global_model.load_state_dict(aggregated_state)

    val_loss, val_accuracy = evaluate_model(
        global_model,
        DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        ),
    )

    print(f"Global validation loss: {val_loss:.4f}")
    print(f"Global validation accuracy: {val_accuracy:.2f}%")


# --------------------------------------------------
# 9. Save the federated model separately
# --------------------------------------------------

federated_model_path = PROJECT_DIR / "models" / "federated_model.pth"
federated_model_path.parent.mkdir(parents=True, exist_ok=True)

torch.save(
    {
        "model_state_dict": global_model.state_dict(),
        "class_names": class_names,
        "federated_rounds": FEDERATED_ROUNDS,
        "num_clients": NUM_CLIENTS,
    },
    federated_model_path,
)

print("\nFederated learning complete!")
print("Federated model saved to:", federated_model_path)
print("\nNote: This is a simulation of three hospital clients.")
print("Differential privacy and secure aggregation are not implemented yet.")
