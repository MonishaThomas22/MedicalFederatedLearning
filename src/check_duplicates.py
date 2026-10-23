
from pathlib import Path
from collections import defaultdict
import hashlib

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"

SPLITS = ["train", "val", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def get_hash(image_path):
    """Calculate a hash of the exact file contents."""
    digest = hashlib.sha256()

    with image_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    hashes_by_split = {
        split: defaultdict(list) for split in SPLITS
    }

    total_images = 0

    for split in SPLITS:
        split_dir = DATA_DIR / split

        if not split_dir.exists():
            print(f"ERROR: Missing folder: {split_dir}")
            return

        for image_path in split_dir.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                file_hash = get_hash(image_path)
                hashes_by_split[split][file_hash].append(image_path)
                total_images += 1

    print(f"Total images checked: {total_images}")

    duplicate_found = False

    # Check for identical image files across different splits.
    for i, split_a in enumerate(SPLITS):
        for split_b in SPLITS[i + 1:]:
            shared_hashes = (
                set(hashes_by_split[split_a])
                & set(hashes_by_split[split_b])
            )

            if shared_hashes:
                duplicate_found = True
                duplicate_count = sum(
                    min(
                        len(hashes_by_split[split_a][h]),
                        len(hashes_by_split[split_b][h]),
                    )
                    for h in shared_hashes
                )

                print(
                    f"\nWARNING: {duplicate_count} identical image(s) "
                    f"found across {split_a} and {split_b}."
                )

                for file_hash in list(shared_hashes)[:5]:
                    print("  Split A:", hashes_by_split[split_a][file_hash][0])
                    print("  Split B:", hashes_by_split[split_b][file_hash][0])

    if not duplicate_found:
        print("No exact duplicate files found across train, val, and test.")

    print(
        "\nNote: This checks exact file contents. "
        "It does not detect visually similar images saved differently."
    )
    print("Duplicate check complete. No images were changed or deleted.")


if __name__ == "__main__":
    main()
