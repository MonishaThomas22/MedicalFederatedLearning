
from pathlib import Path
from collections import defaultdict
import hashlib
import random
import shutil

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed_lung_3class"

PNEUMONIA_DIR = (
    RAW_DIR / "pneumonia_dataset" / "chest_xray" / "chest_xray"
)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# --------------------------------------------------
# 2. CHECK SOURCE FOLDERS
# --------------------------------------------------

sources = {
    "Normal": [
        RAW_DIR / "Normal",
        PNEUMONIA_DIR / "train" / "NORMAL",
        PNEUMONIA_DIR / "test" / "NORMAL",
        PNEUMONIA_DIR / "val" / "NORMAL",
    ],
    "Tuberculosis": [
        RAW_DIR / "Tuberculosis",
    ],
    "Pneumonia": [
        PNEUMONIA_DIR / "train" / "PNEUMONIA",
        PNEUMONIA_DIR / "test" / "PNEUMONIA",
        PNEUMONIA_DIR / "val" / "PNEUMONIA",
    ],
}

for label, folders in sources.items():
    for folder in folders:
        if not folder.exists():
            raise FileNotFoundError(
                f"Missing source folder: {folder}"
            )

# Never overwrite an existing prepared dataset.
if any(OUTPUT_DIR.iterdir()) if OUTPUT_DIR.exists() else False:
    raise FileExistsError(
        f"Output folder is not empty: {OUTPUT_DIR}\n"
        "Please do not delete anything. Check the folder first."
    )

# --------------------------------------------------
# 3. IDENTIFY PATIENT GROUPS
# --------------------------------------------------

def get_group_id(path, label, source_folder):
    filename = path.stem

    if label == "Pneumonia":
        # person123_bacteria_... and person123_virus_...
        if filename.startswith("person"):
            return filename.split("_")[0]

    if label == "Normal" and "chest_xray" in str(source_folder):
        # Group NORMAL2-IM-0587 images by IM number.
        parts = filename.split("-")

        if len(parts) >= 3 and parts[0] == "NORMAL2":
            return f"{parts[1]}-{parts[2]}"

        if len(parts) >= 2 and parts[0] == "IM":
            return f"IM-{parts[1]}"

    # The source does not provide a reliable patient ID.
    # Keep the image in its own group.
    return f"{label}_{filename}_{hashlib.sha256(str(path).encode()).hexdigest()[:10]}"


# --------------------------------------------------
# 4. COLLECT IMAGES AND REMOVE EXACT DUPLICATES
# --------------------------------------------------

class_images = defaultdict(list)
seen_hashes = defaultdict(set)

total_found = 0
duplicates_removed = 0

for label, folders in sources.items():
    for folder in folders:
        for path in folder.rglob("*"):

            if not path.is_file():
                continue

            if path.suffix.lower() not in VALID_EXTENSIONS:
                continue

            total_found += 1

            digest = hashlib.sha256(path.read_bytes()).hexdigest()

            # Remove exact duplicates within each class.
            if digest in seen_hashes[label]:
                duplicates_removed += 1
                continue

            seen_hashes[label].add(digest)

            group_id = get_group_id(path, label, folder)

            class_images[label].append({
                "path": path,
                "group": group_id,
                "hash": digest,
            })

# --------------------------------------------------
# 5. SPLIT BY GROUP
# --------------------------------------------------

def split_groups(records):
    groups = defaultdict(list)

    for record in records:
        groups[record["group"]].append(record)

    group_ids = list(groups.keys())
    random.shuffle(group_ids)

    total_groups = len(group_ids)

    train_end = int(total_groups * 0.70)
    val_end = int(total_groups * 0.85)

    train_groups = set(group_ids[:train_end])
    val_groups = set(group_ids[train_end:val_end])
    test_groups = set(group_ids[val_end:])

    result = {
        "train": [],
        "val": [],
        "test": [],
    }

    for group_id, images in groups.items():
        if group_id in train_groups:
            result["train"].extend(images)
        elif group_id in val_groups:
            result["val"].extend(images)
        else:
            result["test"].extend(images)

    return result


# --------------------------------------------------
# 6. CREATE OUTPUT FOLDERS AND COPY IMAGES
# --------------------------------------------------

for split in ["train", "val", "test"]:
    for label in ["Normal", "Tuberculosis", "Pneumonia"]:
        (OUTPUT_DIR / split / label).mkdir(
            parents=True,
            exist_ok=True
        )

summary = {}

for label, records in class_images.items():

    split_data = split_groups(records)

    summary[label] = {}

    for split, images in split_data.items():

        destination = OUTPUT_DIR / split / label

        for number, item in enumerate(images, 1):
            source = item["path"]

            new_name = (
                f"{number:06d}_{source.stem}{source.suffix.lower()}"
            )

            shutil.copy2(source, destination / new_name)

        summary[label][split] = len(images)


# --------------------------------------------------
# 7. PRINT FINAL REPORT
# --------------------------------------------------

print("\n========== DATASET PREPARATION COMPLETE ==========")

print("Total source images found:", total_found)
print("Exact duplicate copies removed:", duplicates_removed)
print("Unique images copied:", sum(
    sum(counts.values()) for counts in summary.values()
))

for label, counts in summary.items():
    print(f"\n{label}")
    for split, count in counts.items():
        print(f"  {split}: {count}")

print("\nOutput folder:", OUTPUT_DIR)
print("\nOriginal raw datasets were not modified.")
