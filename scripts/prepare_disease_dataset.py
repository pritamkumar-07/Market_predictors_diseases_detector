from pathlib import Path
import random
import shutil

ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "disease_dataset" / "raw_dataset"
OUTPUT = ROOT / "disease_dataset"

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

random.seed(SEED)


def main():
    if not SOURCE.exists():
        raise SystemExit(
            f"Dataset not found at:\n{SOURCE}\n\n"
            "Make sure the extracted dataset folder is named raw_dataset."
        )

    class_dirs = [
        p for p in SOURCE.iterdir()
        if p.is_dir()
    ]

    if not class_dirs:
        raise SystemExit("No disease class folders found inside raw_dataset.")

    print(f"Found {len(class_dirs)} classes.")

    for class_dir in sorted(class_dirs):
        images = [
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        random.shuffle(images)

        total = len(images)

        train_end = int(total * TRAIN_RATIO)
        val_end = train_end + int(total * VAL_RATIO)

        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]

        print(
            f"{class_dir.name}: "
            f"{len(train_images)} train, "
            f"{len(val_images)} validation, "
            f"{len(test_images)} test"
        )

        for split_name, split_images in [
            ("train", train_images),
            ("validation", val_images),
            ("test", test_images),
        ]:
            destination = OUTPUT / split_name / class_dir.name
            destination.mkdir(parents=True, exist_ok=True)

            for image_path in split_images:
                target = destination / image_path.name

                # Avoid filename collision
                if target.exists():
                    target = destination / f"{image_path.stem}_{random.randint(100000, 999999)}{image_path.suffix}"

                shutil.copy2(image_path, target)

    print("\nDataset preparation completed successfully!")
    print("Train      :", OUTPUT / "train")
    print("Validation :", OUTPUT / "validation")
    print("Test       :", OUTPUT / "test")


if __name__ == "__main__":
    main()