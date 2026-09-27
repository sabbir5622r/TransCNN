from pathlib import Path

from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_generators(dataset_root, image_size, batch_size, augmentation,
                      include_real_world=False):
    dataset_root = Path(dataset_root)
    train_datagen = ImageDataGenerator(rescale=1.0 / 255, **augmentation)
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255)

    generators = {}
    splits = ["train", "val", "test"]
    if include_real_world:
        splits.append("real_world")

    for split in splits:
        directory = dataset_root / split
        if not directory.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {directory}")

        datagen = train_datagen if split == "train" else eval_datagen
        generators[split] = datagen.flow_from_directory(
            str(directory),
            target_size=tuple(image_size[:2]),
            batch_size=batch_size,
            class_mode="binary",
            shuffle=split in ("train", "val"),
        )

        expected_classes = {"fake": 0, "real": 1}
        if generators[split].class_indices != expected_classes:
            raise ValueError(
                f"Expected class folders {expected_classes} in {directory}, "
                f"found {generators[split].class_indices}"
            )

    return generators


def create_evaluation_generator(directory, image_size, batch_size):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {directory}")
    generator = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
        str(directory), target_size=tuple(image_size[:2]),
        batch_size=batch_size, class_mode="binary", shuffle=False,
    )
    if generator.class_indices != {"fake": 0, "real": 1}:
        raise ValueError(f"Expected fake/real folders in {directory}")
    return generator
