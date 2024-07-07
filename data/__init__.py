import numpy as np
from .MyDataset import MyDataset, balanced_weights, calculate_weights
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from logger import logging








def get_dataloader(
        train_dir,
        val_dir,
        train_csv,
        val_csv,
        class_mapping,
        train_transform=None,
        val_transform=None,
        split=(0.6, 0.4),
        batch_size=32,
        *args, **kwargs):
    """
    This function returns the train, val, and test dataloaders, using specified mappings for class labels.

    :param train_dir: Directory where training images are stored.
    :param val_dir: Directory where validation images are stored.
    :param train_csv: Path to the CSV file for training data.
    :param val_csv: Path to the CSV file for validation data.
    :param class_mapping: Dictionary mapping class labels to numeric values.
    :param train_transform: Optional transformations to apply to training images.
    :param val_transform: Optional transformations to apply to validation images.
    :param split: Tuple indicating how to split validation data into val and test sets.
    :param batch_size: Number of samples in each batch.
    """
    # Create the datasets using the provided CSV files and class mapping

    

    train_ds = MyDataset(csv_file=train_csv, img_dir=train_dir, class_mapping=class_mapping, transform=train_transform)
    val_ds = MyDataset(csv_file=val_csv, img_dir=val_dir, class_mapping=class_mapping, transform=val_transform)

    # Assuming 'train_ds' is an instance of MyDataset
    # weights = balanced_weights([(data, label) for data, label in train_ds], len(class_mapping))
    # sampler = WeightedRandomSampler(weights, len(weights))

    sample_weights = calculate_weights(train_csv)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    

    # Now we want to split the val_ds into validation and test datasets
    lengths = np.array(split) * len(val_ds)
    lengths = lengths.astype(int)
    left = len(val_ds) - lengths.sum()
    # We need to add the difference due to float approximation to int
    lengths[-1] += left

    val_ds, test_ds = random_split(val_ds, lengths.tolist())
    logging.info(f'Train samples={len(train_ds)}, Validation samples={len(val_ds)}, Test samples={len(test_ds)}')

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False, sampler=sampler,  *args, **kwargs)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, *args, **kwargs)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, *args, **kwargs)

    return train_dl, val_dl, test_dl




# def get_dataloader(data_csv, 
#                    img_dir, 
#                    class_mapping, 
#                    train_transform,
#                    val_transform,
#                    split=(0.6, 0.2, 0.2), 
#                    batch_size=32, 
#                    *args, **kwargs):
#     """
#     This function returns the train, validation, and test dataloaders, using a single CSV for input and splitting
#     the dataset into specified portions.

#     :param data_csv: Path to the CSV file containing data.
#     :param img_dir: Directory where images are stored.
#     :param class_mapping: Dictionary mapping class labels to numeric values.
#     :param transform: Optional transformations to apply to images.
#     :param split: Tuple indicating how to split the data into train, val, and test sets.
#     :param batch_size: Number of samples in each batch.
#     """
#     # Read the dataset from CSV

    

#     full_dataset = MyDataset(csv_file=data_csv, img_dir=img_dir, class_mapping=class_mapping, transform=transform)

#     # Calculate split lengths
#     train_size, val_size, test_size = [int(x * len(full_dataset)) for x in split]
#     test_size = len(full_dataset) - train_size - val_size  # Adjust test size for any rounding off errors

#     # Split the dataset
#     train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
#     logging.info(f'Train samples={len(train_dataset)}, Validation samples={len(val_dataset)}, Test samples={len(test_dataset)}')

#     # Create dataloaders
#     train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, *args, **kwargs)
#     val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, *args, **kwargs)
#     test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, *args, **kwargs)

#     return train_dl, val_dl, test_dl
