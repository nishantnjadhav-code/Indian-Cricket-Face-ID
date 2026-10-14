from pathlib import Path
import hashlib
import random
import shutil

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "dataset" / "raw"
PROCESSED_DIR = ROOT / "dataset" / "processed"

SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MIN_IMAGES_PER_CLASS = 10


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def valid_image(path):
    try:
        with Image.open(path) as image:
            image.verify()

        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            image.convert("RGB")

        return True

    except Exception:
        return False


def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {RAW_DIR}"
        )

    # Support datasets with player folders directly inside raw,
    # as well as datasets containing a parent folder such as
    # dataset/raw/indian cricketer/<Player Name>/.
    class_dirs = sorted(
        path
        for path in RAW_DIR.rglob("*")
        if path.is_dir()
        and not path.name.startswith(".")
        and any(
            child.is_file()
            and child.suffix.lower() in IMAGE_EXTENSIONS
            for child in path.iterdir()
        )
    )

    # Keep only actual player folders, not parent directories
    # containing other player folders.
    class_dirs = [
        path for path in class_dirs
        if not any(
            other != path and other.parent == path
            for other in class_dirs
        )
    ]

    if not class_dirs:
        raise RuntimeError(
            "No player folders containing images were found. "
            "Check the dataset/raw directory."
        )

    random.seed(SEED)

    # Clear only the generated processed dataset.
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)

    for split in ("train", "val", "test"):
        (PROCESSED_DIR / split).mkdir(
            parents=True,
            exist_ok=True
        )

    # Prevent exact duplicate images from crossing classes or splits.
    seen_hashes = set()

    total_images = 0
    total_duplicates = 0
    total_invalid = 0
    skipped_classes = 0

    print("\nDataset inspection")
    print("=" * 50)

    for class_dir in class_dirs:
        candidates = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        class_images = []

        for path in candidates:
            if not valid_image(path):
                print(f"Invalid image skipped: {path}")
                total_invalid += 1
                continue

            file_hash = sha256_file(path)

            if file_hash in seen_hashes:
                total_duplicates += 1
                continue

            seen_hashes.add(file_hash)
            class_images.append(path)

        if len(class_images) < MIN_IMAGES_PER_CLASS:
            print(
                f"Skipping {class_dir.name}: "
                f"only {len(class_images)} valid unique images"
            )
            skipped_classes += 1
            continue

        # Shuffle images reproducibly for each player.
        random.shuffle(class_images)

        count = len(class_images)

        train_count = int(count * TRAIN_RATIO)
        val_count = int(count * VAL_RATIO)
        test_count = count - train_count - val_count

        if min(train_count, val_count, test_count) < 1:
            print(
                f"Skipping {class_dir.name}: "
                "insufficient images for all splits"
            )
            skipped_classes += 1
            continue

        train_end = train_count
        val_end = train_count + val_count

        split_data = {
            "train": class_images[:train_end],
            "val": class_images[train_end:val_end],
            "test": class_images[val_end:],
        }

        print(f"\n{class_dir.name}: {count} images")

        for split, images in split_data.items():
            destination = (
                PROCESSED_DIR / split / class_dir.name
            )
            destination.mkdir(
                parents=True,
                exist_ok=True
            )

            for index, source in enumerate(images):
                output_path = destination / f"{index:05d}.jpg"

                try:
                    with Image.open(source) as image:
                        image = ImageOps.exif_transpose(image)
                        image = image.convert("RGB")
                        image.save(
                            output_path,
                            format="JPEG",
                            quality=95
                        )

                except Exception as error:
                    print(
                        f"Could not process {source}: {error}"
                    )
                    total_invalid += 1

                    if output_path.exists():
                        output_path.unlink()

            actual_count = sum(
                1 for file in destination.iterdir()
                if file.is_file() and file.suffix.lower() == ".jpg"
            )

            print(f"  {split}: {actual_count}")

        total_images += count

    # Verify that the three splits contain identical class sets.
    split_classes = {}

    for split in ("train", "val", "test"):
        split_classes[split] = {
            path.name
            for path in (PROCESSED_DIR / split).iterdir()
            if path.is_dir()
            and any(path.iterdir())
        }

    if not split_classes["train"]:
        raise RuntimeError("No training classes were generated.")

    if not (
        split_classes["train"]
        == split_classes["val"]
        == split_classes["test"]
    ):
        raise RuntimeError(
            "Train, validation, and test class folders do not match. "
            "Inspect the dataset preparation output."
        )

    print("\nPreparation completed")
    print(f"Usable source images: {total_images}")
    print(f"Exact duplicate files skipped: {total_duplicates}")
    print(f"Invalid images skipped: {total_invalid}")
    print(f"Classes skipped: {skipped_classes}")
    print(f"Player classes: {len(split_classes['train'])}")
    print(f"Output: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()