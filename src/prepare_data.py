
from pathlib import Path
from collections import defaultdict
import hashlib
import random
import shutil

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

SPLITS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}

SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def get_hash(image_path):
    digest = hashlib.sha256()

    with image_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_DIR}")

    # Group identical images together, even if their filenames differ.
    groups_by_class = defaultdict(lambda: defaultdict(list))

    for class_dir in sorted(RAW_DIR.iterdir()):
        if not class_dir.is_dir():
            continue

        for image_path in class_dir.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                groups_by_class[class_dir.name][get_hash(image_path)].append(image_path)

    if not groups_by_class:
        raise RuntimeError(f"No images found in {RAW_DIR}")

    # Prepare the split folders.
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)

    for split in SPLITS:
        (PROCESSED_DIR / split).mkdir(parents=True, exist_ok=True)

    random_generator = random.Random(SEED)

    for class_name, image_groups in sorted(groups_by_class.items()):
        groups = list(image_groups.values())
        random_generator.shuffle(groups)

        total_images = sum(len(group) for group in groups)

        train_target = round(total_images * SPLITS["train"])
        val_target = round(total_images * SPLITS["val"])

        assigned = {"train": 0, "val": 0, "test": 0}

        for group in groups:
            # Keep every identical copy in the same split.
            if assigned["train"] < train_target:
                split = "train"
            elif assigned["val"] < val_target:
                split = "val"
            else:
                split = "test"

            destination_dir = PROCESSED_DIR / split / class_name
            destination_dir.mkdir(parents=True, exist_ok=True)

            for image_path in group:
                destination = destination_dir / image_path.name

                # Avoid filename collisions within a destination folder.
                counter = 1
                while destination.exists():
                    destination = destination_dir / (
                        f"{image_path.stem}_{counter}{image_path.suffix}"
                    )
                    counter += 1

                shutil.copy2(image_path, destination)

            assigned[split] += len(group)

        print(f"\nClass: {class_name}")
        for split in SPLITS:
            print(f"  {split}: {assigned[split]} images")

    print("\nDataset preparation complete.")
    print("Identical files are kept together in one split.")
    print("Raw images were not changed.")


if __name__ == "__main__":
    main()
