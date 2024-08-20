import numpy as np
from .MyDataset import MyDataset
from torch.utils.data import DataLoader
from logger import logging
import pandas as pd

def get_dataloader(
        data_dir,
        csv_path,
        columns_to_use,
        class_mapping,
        total_images=None,
        split=(0.8, 0.1, 0.1),
        batch_size=32,
        train_transform=None,
        val_transform=None,
        seed=42,  # Set a default seed for reproducibility
        *args, **kwargs):
    """
    This function returns the train, val, and test dataloaders, using a specified mapping for class labels.
    
    :param data_dir: Directory where all images are stored.
    :param csv_path: Path to the CSV file containing data annotations.
    :param class_mapping: Dictionary mapping class labels to numeric values.
    :param total_images: Total number of images to use for the dataset, defaults to all images if not specified.
    :param split: Tuple indicating how to split data into train, validation, and test sets.
    :param batch_size: Number of samples in each batch.
    :param train_transform: Transformations to apply to training images.
    :param val_transform: Transformations to apply to validation and test images.
    :param seed: Random seed for reproducibility.
    """
    # Set seed for reproducibility
    np.random.seed(seed)

    # Load the entire dataset
    full_data = pd.read_csv(csv_path)

    # Optionally select a subset of the data
    if total_images is not None and total_images < len(full_data):
        full_data = full_data.sample(n=total_images, random_state=seed)

    # Shuffle and split indices into train, validation, and test sets
    indices = np.random.permutation(len(full_data))
    train_end = int(np.floor(split[0] * len(indices)))
    val_end = train_end + int(np.floor(split[1] * len(indices)))

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]


    # Create data subsets and reset index
    train_data = full_data.iloc[train_indices].reset_index(drop=True)
    val_data = full_data.iloc[val_indices].reset_index(drop=True)
    test_data = full_data.iloc[test_indices].reset_index(drop=True)

    # Instantiate datasets
    train_ds = MyDataset(data=train_data, img_dir=data_dir, class_mapping=class_mapping, columns=columns_to_use, transform=train_transform)
    val_ds = MyDataset(data=val_data, img_dir=data_dir, class_mapping=class_mapping, columns=columns_to_use, transform=val_transform)
    test_ds = MyDataset(data=test_data, img_dir=data_dir, class_mapping=class_mapping, columns=columns_to_use, transform=val_transform)

    # Set up the dataloaders
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, *args, **kwargs)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, *args, **kwargs)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, *args, **kwargs)

    # logging.info(f'Train samples={len(train_ds)}, Validation samples={len(val_ds)}, Test samples={len(test_ds)}')
    # Log class distributions
    train_distribution = train_ds.get_class_distribution()
    val_distribution = val_ds.get_class_distribution()
    test_distribution = test_ds.get_class_distribution()

    logging.info(f"Train Class Distribution: {train_distribution}")
    logging.info(f"Validation Class Distribution: {val_distribution}")
    logging.info(f"Test Class Distribution: {test_distribution}")

    return train_dl, val_dl, test_dl