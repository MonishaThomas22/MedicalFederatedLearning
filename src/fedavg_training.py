
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_FILE = MODEL_DIR / "federated_model.pth"

DEVICE = torch.device("cpu")

NUM_HOSPITALS = 3
COMMUNICATION_ROUNDS = 3
LOCAL_EPOCHS = 1
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. LOAD TRAINING DATA
# ==================================================

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
print("Training images:", len(train_dataset))
print("Simulated hospitals:", NUM_HOSPITALS)
print("Communication rounds:", COMMUNICATION_ROUNDS)


# ==================================================
# 3. SPLIT TRAINING DATA BETWEEN HOSPITALS
# ==================================================

indices = list(range(len(train_dataset)))
random.shuffle(indices)

hospital_indices = [
    indices[i::NUM_HOSPITALS]
    for i in range(NUM_HOSPITALS)
]

hospital_loaders = []
hospital_sample_counts = []

for hospital_id, local_indices in enumerate(
    hospital_indices, start=1
):
    local_dataset = Subset(train_dataset, local_indices)

    loader = DataLoader(
        local_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    hospital_loaders.append(loader)
    hospital_sample_counts.append(len(local_dataset))

    print(
        f"Hospital {hospital_id}: "
        f"{len(local_dataset)} training images"
    )


# ==================================================
# 4. CREATE ONE SHARED GLOBAL MODEL
# ==================================================

def create_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes,
    )
    return model.to(DEVICE)


global_model = create_model()

print("\nInitial global model created.")


# ==================================================
# 5. LOCAL TRAINING FUNCTION
# ==================================================

def train_local_model(model, loader, hospital_id, round_id):
    model.train()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
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
    accuracy = 100.0 * total_correct / total_images

    print(
        f"Hospital {hospital_id} | "
        f"Round {round_id} | "
        f"Loss: {average_loss:.4f} | "
        f"Training accuracy: {accuracy:.2f}%"
    )

    return model.state_dict()


# ==================================================
# 6. FEDAVG AGGREGATION
# ==================================================

def aggregate_models(client_states, sample_counts):
    total_samples = sum(sample_counts)
    global_state = {}

    for key in client_states[0]:
        tensors = [
            state[key] for state in client_states
        ]

        if torch.is_floating_point(tensors[0]):
            weighted_average = torch.zeros_like(tensors[0])

            for tensor, count in zip(tensors, sample_counts):
                weighted_average += (
                    tensor * (count / total_samples)
                )

            global_state[key] = weighted_average
        else:
            # Preserve integer buffers safely.
            global_state[key] = tensors[0].clone()

    return global_state


# ==================================================
# 7. COMMUNICATION ROUNDS
# ==================================================

for round_id in range(1, COMMUNICATION_ROUNDS + 1):
    print("\n" + "=" * 55)
    print(f"FEDERATED LEARNING ROUND {round_id}")
    print("=" * 55)

    client_states = []

    # Every hospital starts this round with the
    # same current global model parameters.
    for hospital_id, loader in enumerate(
        hospital_loaders, start=1
    ):
        local_model = create_model()

        local_model.load_state_dict(
            global_model.state_dict()
        )

        local_state = train_local_model(
            local_model,
            loader,
            hospital_id,
            round_id,
        )

        client_states.append(local_state)
        del local_model

    # Server combines the local model updates.
    aggregated_state = aggregate_models(
        client_states,
        hospital_sample_counts,
    )

    global_model.load_state_dict(aggregated_state)

    print(f"\nRound {round_id} aggregation completed.")


# ==================================================
# 8. SAVE FINAL GLOBAL MODEL
# ==================================================

torch.save(
    {
        "model_state_dict": global_model.state_dict(),
        "class_names": class_names,
        "num_classes": num_classes,
        "num_clients": NUM_HOSPITALS,
        "client_sample_counts": hospital_sample_counts,
        "communication_rounds": COMMUNICATION_ROUNDS,
        "local_epochs": LOCAL_EPOCHS,
        "aggregation": "FedAvg",
        "initialized_from_centralized_model": False,
    },
    OUTPUT_FILE,
)

print("\n" + "=" * 55)
print("FEDERATED TRAINING COMPLETED!")
print("=" * 55)
print("Communication rounds:", COMMUNICATION_ROUNDS)
print("Hospital sample counts:", hospital_sample_counts)
print("Aggregation method: FedAvg")
print("Saved global model to:", OUTPUT_FILE)
print("\nNext: evaluate the new global model.")
