
from pathlib import Path
from collections import defaultdict
from PIL import Image
import hashlib

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed_lung_3class"

splits = ["train", "val", "test"]
classes = ["Normal", "Tuberculosis", "Pneumonia"]
extensions = {".jpg", ".jpeg", ".png"}

hash_locations = defaultdict(list)
corrupted = []
total = 0

print("\n========== DATASET VERIFICATION ==========")

for split in splits:
    print(f"\n{split.upper()}")

    for label in classes:
        folder = DATA_DIR / split / label
        files = [
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        ]

        print(f"{label}: {len(files)} images")

        for path in files:
            total += 1

            try:
                with Image.open(path) as img:
                    img.verify()

                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                hash_locations[digest].append((split, label, path.name))

            except Exception as error:
                corrupted.append((path, str(error)))

print("\n========== FINAL CHECKS ==========")
print("Total images checked:", total)
print("Corrupted images:", len(corrupted))

cross_split_duplicates = []

for digest, locations in hash_locations.items():
    split_names = {item[0] for item in locations}

    if len(split_names) > 1:
        cross_split_duplicates.append(locations)

print("Exact duplicate groups across splits:",
      len(cross_split_duplicates))

if corrupted:
    print("\nCorrupted image examples:")
    for path, error in corrupted[:10]:
        print(path, error)

if cross_split_duplicates:
    print("\nCross-split duplicate examples:")
    for group in cross_split_duplicates[:10]:
        print(group)

if not corrupted and not cross_split_duplicates:
    print("\nALL BASIC IMAGE CHECKS PASSED")
