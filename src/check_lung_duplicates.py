
from pathlib import Path
from collections import defaultdict
import hashlib

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"

folders = {
    "Normal": RAW_DIR / "Normal",
    "Tuberculosis": RAW_DIR / "Tuberculosis",
    "Pneumonia": RAW_DIR / "pneumonia_dataset" / "chest_xray" / "chest_xray",
}

hash_groups = defaultdict(list)

for label, folder in folders.items():
    for path in folder.rglob("*"):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        digest = hashlib.sha256(path.read_bytes()).hexdigest()

        if label == "Pneumonia":
            split = path.parent.parent.name
        else:
            split = "raw"

        hash_groups[digest].append({
            "class": label,
            "split": split,
            "filename": path.name
        })

duplicates = [
    group for group in hash_groups.values()
    if len(group) > 1
]

print("Total duplicate groups:", len(duplicates))

for number, group in enumerate(duplicates, 1):
    print(f"\nDuplicate group {number}:")

    for item in group:
        print(
            f"Class: {item['class']} | "
            f"Split: {item['split']} | "
            f"File: {item['filename']}"
        )
