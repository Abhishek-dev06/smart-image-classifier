"""Split dataset/raw into dataset/train, dataset/validation, dataset/test."""

import random
import shutil

from PIL import Image

from utils import (RAW_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR, SEED,
                   TRAIN_RATIO, VAL_RATIO, VALID_EXTENSIONS, get_class_names)


def is_valid_image(path) -> bool:
    """Return True if the file can be opened as an image."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def list_images(class_dir):
    files = sorted(p for p in class_dir.iterdir()
                   if p.suffix.lower() in VALID_EXTENSIONS)
    good = [p for p in files if is_valid_image(p)]
    skipped = len(files) - len(good)
    if skipped:
        print(f"  Skipped {skipped} corrupted image(s) in {class_dir.name}")
    return good


def copy_files(files, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, destination / f.name)


def main():
    class_names = get_class_names(RAW_DIR)
    print(f"Found {len(class_names)} classes: {class_names}\n")

    # Validate every class before removing a previously usable split.
    images_by_class = {name: list_images(RAW_DIR / name) for name in class_names}
    for name, images in images_by_class.items():
        if len(images) < 10:
            raise ValueError(f"Class '{name}' has too few images ({len(images)}).")

    # Remove old splits so runs are always clean (raw data is untouched)
    for folder in (TRAIN_DIR, VAL_DIR, TEST_DIR):
        if folder.exists():
            shutil.rmtree(folder)

    rng = random.Random(SEED)
    totals = {"train": 0, "validation": 0, "test": 0}

    for name in class_names:
        images = images_by_class[name]

        rng.shuffle(images)
        n_train = int(len(images) * TRAIN_RATIO)
        n_val = int(len(images) * VAL_RATIO)

        splits = {
            "train": (images[:n_train], TRAIN_DIR),
            "validation": (images[n_train:n_train + n_val], VAL_DIR),
            "test": (images[n_train + n_val:], TEST_DIR),
        }
        for split_name, (files, folder) in splits.items():
            copy_files(files, folder / name)
            totals[split_name] += len(files)

        print(f"{name:<12} total={len(images):<5} "
              f"train={n_train:<5} val={n_val:<5} "
              f"test={len(images) - n_train - n_val}")

    print("\nDone.")
    print(f"Train: {totals['train']} | Validation: {totals['validation']} | "
          f"Test: {totals['test']}")


if __name__ == "__main__":
    main()
