
from pathlib import Path
import copy
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

from privacy_helpers import (
    ClientPrivacyAccountant,
    get_original_model_state,
    secure_aggregate_weighted_deltas,
)


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data" / "processed_lung_3class"
MODEL_DIR = PROJECT_DIR / "models"

# Save to a NEW file. Do not overwrite the original model.
OUTPUT_FILE = MODEL_DIR / "lung_hospital_federated_model_dp.pth"

DEVICE = torch.device("cpu")

NUM_HOSPITALS = 3
COMMUNICATION_ROUNDS = 3
LOCAL_EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 0.001
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LOAD TRAINING DATA
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

train_dataset = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=train_transform
)

class_names = train_dataset.classes
num_classes = len(class_names)

print("Device:", DEVICE)
print("Classes:", class_names)
print("Training images:", len(train_dataset))
print("Number of simulated hospitals:", NUM_HOSPITALS)
print("Communication rounds:", COMMUNICATION_ROUNDS)


# ==========================================
# 3. DIVIDE DATA BETWEEN HOSPITALS
# ==========================================

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
        num_workers=0
    )

    hospital_loaders.append(loader)
    hospital_sample_counts.append(len(local_dataset))

    print(
        f"Hospital {hospital_id}: "
        f"{len(local_dataset)} training images"
    )


# ==========================================
# 4. CREATE GLOBAL RESNET18 MODEL
# ==========================================

def create_initial_model():

    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )

    # Freeze pretrained feature extractor.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Train only the final classifier.
    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model.to(DEVICE)


global_model = create_initial_model()

print("\nImageNet-pretrained ResNet18 created.")
print("Feature extractor: Frozen")
print("Final classifier: Trainable")


# ==========================================
# 5. CREATE ONE PRIVACY ACCOUNTANT
#    FOR EACH HOSPITAL
# ==========================================

privacy_accounts = [
    ClientPrivacyAccountant()
    for _ in range(NUM_HOSPITALS)
]

epsilon_history = [
    [] for _ in range(NUM_HOSPITALS)
]


# ==========================================
# 6. LOCAL DIFFERENTIALLY PRIVATE TRAINING
# ==========================================

def train_local_model(
    model,
    loader,
    hospital_id,
    round_id,
    privacy_account
):

    model.train()

    # Keep frozen BatchNorm layers in evaluation mode.
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    optimizer = optim.Adam(
        model.fc.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.CrossEntropyLoss()

    # Convert local training into DP-SGD.
    private_model, private_optimizer, private_loader = (
        privacy_account.make_private(
            model,
            optimizer,
            loader
        )
    )

    private_model.train()

    # Frozen feature extractor BatchNorm remains in eval mode.
    for module in private_model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    total_loss = 0.0
    total_correct = 0
    total_images = 0

    for epoch in range(LOCAL_EPOCHS):

        for images, labels in private_loader:

            # Poisson sampling can produce empty batches.
            if labels.numel() == 0:
                continue

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            private_optimizer.zero_grad()

            outputs = private_model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            private_optimizer.step()

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size

            predictions = outputs.argmax(dim=1)

            total_correct += (
                predictions == labels
            ).sum().item()

            total_images += batch_size

    if total_images > 0:

        average_loss = total_loss / total_images
        accuracy = 100.0 * total_correct / total_images

        print(
            f"Hospital {hospital_id} | "
            f"Round {round_id} | "
            f"Loss: {average_loss:.4f} | "
            f"Accuracy: {accuracy:.2f}%"
        )

    else:
        print(
            f"Hospital {hospital_id} | "
            f"Round {round_id} | No non-empty batches"
        )

    # Record the privacy budget for this hospital.
    epsilon = privacy_account.get_epsilon()

    epsilon_history[hospital_id - 1].append(epsilon)

    print(
        f"Hospital {hospital_id} privacy budget: "
        f"epsilon = {epsilon:.4f}"
    )

    # Return the underlying ResNet18 state dictionary.
    return copy.deepcopy(
        get_original_model_state(private_model)
    )


# ==========================================
# 7. FEDERATED TRAINING WITH SECURE AGGREGATION
# ==========================================

for round_id in range(1, COMMUNICATION_ROUNDS + 1):

    print("\n" + "=" * 60)
    print(f"FEDERATED LEARNING ROUND {round_id}")
    print("=" * 60)

    client_states = []

    for hospital_id, loader in enumerate(
        hospital_loaders, start=1
    ):

        print(f"\nTraining Hospital {hospital_id}...")

        # Each hospital receives the latest global model.
        local_model = models.resnet18(weights=None)

        local_model.fc = nn.Linear(
            local_model.fc.in_features,
            num_classes
        )

        local_model.load_state_dict(
            global_model.state_dict()
        )

        local_model.to(DEVICE)

        # Freeze the feature extractor.
        for parameter in local_model.parameters():
            parameter.requires_grad = False

        for parameter in local_model.fc.parameters():
            parameter.requires_grad = True

        local_state = train_local_model(
            local_model,
            loader,
            hospital_id,
            round_id,
            privacy_accounts[hospital_id - 1]
        )

        client_states.append(local_state)

        del local_model

    # Simulated pairwise-mask secure aggregation.
    print("\nPerforming simulated secure aggregation...")

    aggregated_state = secure_aggregate_weighted_deltas(
        global_model.state_dict(),
        client_states,
        hospital_sample_counts
    )

    # Update the shared global model.
    global_model.load_state_dict(aggregated_state)

    print(
        f"Round {round_id}: "
        "Secure aggregation and FedAvg completed."
    )


# ==========================================
# 8. COLLECT FINAL PRIVACY INFORMATION
# ==========================================

final_epsilon = [
    account.get_epsilon()
    for account in privacy_accounts
]

print("\nFinal privacy budgets:")

for hospital_id, epsilon in enumerate(
    final_epsilon, start=1
):
    print(
        f"Hospital {hospital_id}: "
        f"epsilon = {epsilon:.4f}"
    )


# ==========================================
# 9. SAVE THE NEW GLOBAL MODEL
# ==========================================

checkpoint = {
    "model_state_dict": global_model.state_dict(),
    "class_names": class_names,
    "num_classes": num_classes,
    "architecture": "ResNet18",

    "num_clients": NUM_HOSPITALS,
    "client_sample_counts": hospital_sample_counts,

    "communication_rounds": COMMUNICATION_ROUNDS,
    "local_epochs": LOCAL_EPOCHS,

    "aggregation": "Weighted FedAvg",
    "initialization": "ImageNet pretrained weights",

    "training_mode": (
        "Frozen feature extractor; "
        "trainable classifier"
    ),

    "data_distribution": "Random IID simulation",

    "differential_privacy": True,
    "privacy_mechanism": "DP-SGD using Opacus",
    "privacy_accountant": "PRV",
    "noise_multiplier": 1.0,
    "max_grad_norm": 1.0,
    "delta": 1e-5,
    "epsilon_per_hospital": final_epsilon,
    "epsilon_history": epsilon_history,

    "privacy_scope": (
        "Trainable classifier gradients; "
        "feature extractor frozen"
    ),

    "secure_aggregation": True,
    "secure_aggregation_mode": "Simulated pairwise masking",
    "production_secure_aggregation": False,

    "secure_rng": False,
    "research_prototype": True,
}

torch.save(checkpoint, OUTPUT_FILE)


# ==========================================
# 10. TRAINING SUMMARY
# ==========================================

print("\n" + "=" * 60)
print("PRIVACY-ENABLED FEDERATED TRAINING COMPLETED")
print("=" * 60)

print("Classes:", class_names)
print("Hospitals:", NUM_HOSPITALS)
print("Communication rounds:", COMMUNICATION_ROUNDS)
print("Hospital sample counts:", hospital_sample_counts)
print("Aggregation: Weighted FedAvg")
print("Differential Privacy: Enabled")
print("Secure Aggregation: Simulated pairwise masking")
print("Production secure aggregation: Not implemented")
print("Secure random number generation: Disabled")
print("Saved global model:", OUTPUT_FILE)

print(
    "\nResearch prototype only. "
    "This is not a production-secure medical system "
    "or a clinical diagnostic tool."
)
