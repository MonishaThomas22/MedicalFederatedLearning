
from pathlib import Path
from collections import Counter
import random
import shutil

# --------------------------------------------------
# PATHS AND SETTINGS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_DIR / "data" / "raw_brain_mri"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed_brain_mri"

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

NUM_CLIENTS = 3
VALIDATION_RATIO = 0.15
RANDOM_SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

random.seed(RANDOM_SEED)


def get_images(folder):
    """Return supported image files from a class folder."""
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def split_list(items, ratios):
    """Split a list according to the supplied ratios."""
    first = int(len(items) * ratios[0])
    second = int(len(items) * ratios[1])

    return (
        items[:first],
        items[first:first + second],
        items[first + second:]
    )


def copy_images(images, destination):
    """Copy images into a destination folder."""
    destination.mkdir(parents=True, exist_ok=True)

    for image_path in images:
        shutil.copy2(image_path, destination / image_path.name)


def main():
    print("\nBRAIN MRI DATA PREPARATION")
    print("=" * 50)

    # Check the original dataset.
    for split in ["Training", "Testing"]:
        for class_name in CLASSES:
            folder = RAW_DIR / split / class_name

            if not folder.is_dir():
                raise FileNotFoundError(
                    f"Missing dataset folder: {folder}"
                )

    # Prevent accidental overwriting of previous processed data.
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"\nOutput folder already exists:\n{OUTPUT_DIR}\n"
            "Please inspect it before running this script again."
        )

    print("\nOriginal dataset verified.")

    # Prepare output directories.
    client_counts = Counter()
    validation_counts = Counter()
    testing_counts = Counter()

    for class_name in CLASSES:
        print(f"\nProcessing class: {class_name}")

        images = get_images(RAW_DIR / "Training" / class_name)
        random.shuffle(images)

        # 15% validation, 85% client training pool.
        train_pool, validation_images, _ = split_list(
            images,
            (1 - VALIDATION_RATIO, VALIDATION_RATIO)
        )

        # Distribute training images across 3 clients.
        client_parts = [
            train_pool[i::NUM_CLIENTS]
            for i in range(NUM_CLIENTS)
        ]

        for client_number, client_images in enumerate(
            client_parts, start=1
        ):
            destination = (
                OUTPUT_DIR
                / f"client_{client_number}"
                / class_name
            )

            copy_images(client_images, destination)
            client_counts[(client_number, class_name)] = len(
                client_images
            )

        # Save validation images separately.
        copy_images(
            validation_images,
            OUTPUT_DIR / "validation" / class_name
        )

        validation_counts[class_name] = len(validation_images)

        # Copy original test images without using them for training.
        test_images = get_images(
            RAW_DIR / "Testing" / class_name
        )

        copy_images(
            test_images,
            OUTPUT_DIR / "testing" / class_name
        )

        testing_counts[class_name] = len(test_images)

    # Print summary.
    print("\n" + "=" * 50)
    print("DATA PREPARATION COMPLETED")
    print("=" * 50)

    print("\nCLIENT TRAINING COUNTS")

    for client_number in range(1, NUM_CLIENTS + 1):
        total = 0
        print(f"\nClient {client_number}:")

        for class_name in CLASSES:
            count = client_counts[(client_number, class_name)]
            total += count
            print(f"  {class_name}: {count}")

        print(f"  Total: {total}")

    print("\nVALIDATION COUNTS")
    for class_name in CLASSES:
        print(f"  {class_name}: {validation_counts[class_name]}")

    print("\nTESTING COUNTS")
    for class_name in CLASSES:
        print(f"  {class_name}: {testing_counts[class_name]}")

    print(f"\nProcessed dataset saved at:\n{OUTPUT_DIR}")
    print("\nOriginal MRI dataset was not modified.")


if __name__ == "__main__":
    main()
