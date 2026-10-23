
from pathlib import Path
import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / "models"

CLIENT_FILES = [
    MODELS_DIR / "hospital_1_model.pth",
    MODELS_DIR / "hospital_2_model.pth",
    MODELS_DIR / "hospital_3_model.pth",
]

OUTPUT_FILE = MODELS_DIR / "federated_model.pth"

# Each hospital trained on 702 images.
# FedAvg uses each hospital's number of training samples as its weight.
CLIENT_SAMPLE_COUNTS = [702, 702, 702]


def load_checkpoint(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "Run src/hospital_training.py first."
        )

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise ValueError(f"Invalid model checkpoint: {path}")

    return checkpoint, state_dict


def fedavg(state_dicts, sample_counts):
    total_samples = sum(sample_counts)
    averaged_state = {}

    first_state = state_dicts[0]

    # Check that every hospital model has the same architecture.
    for key in first_state:
        for client_state in state_dicts[1:]:
            if key not in client_state:
                raise ValueError(f"Model parameter missing: {key}")

            if first_state[key].shape != client_state[key].shape:
                raise ValueError(f"Model parameter shape mismatch: {key}")

    # Weighted average of each model parameter.
    for key in first_state:
        tensors = [state[key] for state in state_dicts]

        if torch.is_floating_point(tensors[0]) or torch.is_complex(tensors[0]):
            averaged = torch.zeros_like(tensors[0])

            for tensor, count in zip(tensors, sample_counts):
                averaged += tensor * (count / total_samples)

            averaged_state[key] = averaged
        else:
            # Keep the largest counter value for integer buffers.
            averaged_state[key] = torch.stack(tensors).max(dim=0).values

    return averaged_state


def main():
    print("=" * 55)
    print("FEDERATED LEARNING - FEDAVG AGGREGATION")
    print("=" * 55)

    state_dicts = []
    checkpoint_metadata = []

    for index, path in enumerate(CLIENT_FILES, start=1):
        print(f"\nLoading Hospital {index}: {path.name}")

        checkpoint, state_dict = load_checkpoint(path)
        state_dicts.append(state_dict)

        if isinstance(checkpoint, dict):
            checkpoint_metadata.append(checkpoint)
        else:
            checkpoint_metadata.append({})

        print(f"Hospital {index} checkpoint loaded successfully.")

    print("\nCombining the three hospital models using FedAvg...")

    global_state = fedavg(state_dicts, CLIENT_SAMPLE_COUNTS)

    # Save in the format expected by the project's evaluator.
    first_metadata = checkpoint_metadata[0]
    class_names = first_metadata.get(
        "class_names", ["Normal", "Tuberculosis"]
    )

    global_checkpoint = {
        "model_state_dict": global_state,
        "class_names": class_names,
        "num_classes": len(class_names),
        "num_clients": len(CLIENT_FILES),
        "client_sample_counts": CLIENT_SAMPLE_COUNTS,
        "aggregation": "FedAvg",
        "initialized_from_centralized_model": False,
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(global_checkpoint, OUTPUT_FILE)

    print("\n" + "=" * 55)
    print("FEDAVG AGGREGATION COMPLETED!")
    print("=" * 55)
    print(f"Number of hospitals: {len(CLIENT_FILES)}")
    print(f"Images per hospital: {CLIENT_SAMPLE_COUNTS}")
    print(f"Total samples represented: {sum(CLIENT_SAMPLE_COUNTS)}")
    print(f"Aggregation method: FedAvg")
    print(f"Global model saved to:\n{OUTPUT_FILE}")
    print("\nNext, we will evaluate the new global model.")


if __name__ == "__main__":
    main()
