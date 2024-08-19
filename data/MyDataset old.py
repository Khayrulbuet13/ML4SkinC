from torch.utils.data import Dataset
import os
import torch
from PIL import Image
import pandas as pd
import logging



# Define custom dataset

class MyDataset(Dataset):
    def __init__(self, csv_file, img_dir, class_mapping, columns=None, transform=None):
        """
        Initializes the dataset.
        :param csv_file: Path to the CSV file containing data.
        :param img_dir: Directory where images are stored.
        :param class_mapping: Dictionary mapping class names to numeric values.
        :param columns: List of column names to include as features. If None, all columns are included.
        :param transform: Optional transform to be applied on a sample.
        """
        self.data_frame = pd.read_csv(csv_file, low_memory=False)
        self.img_dir = img_dir
        self.transform = transform
        # Use specified columns if provided, otherwise use all columns starting from the third column
        if columns is not None:
            self.csv_data = self.data_frame[columns]
        # Ensure the image names include the file extension if it's missing
        self.data_frame['image_name'] = self.data_frame['isic_id'].apply(lambda x: f"{x}.jpg" if not x.lower().endswith('.jpg') else x)
        # Directly use the numeric targets from the dataset
        self.targets = self.data_frame['target'].astype(int)
        self.class_mapping = class_mapping

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = self.data_frame.loc[idx, 'image_name']
        img_path = os.path.join(self.img_dir, img_name)
        try:
            image = Image.open(img_path)
            if self.transform:
                image = self.transform(image)
        except FileNotFoundError:
            logging.error(f"Image not found: {img_path}")
            return None  # Consider how you handle missing images in your training loop

        target = self.targets[idx]
        return image, target

    



def calculate_weights(file_path):
    df = pd.read_csv(file_path)
    class_counts = df['target'].value_counts().to_dict()
    total_samples = len(df)
    weights = {cls: total_samples/count for cls, count in class_counts.items()}
    sample_weights = df['target'].map(weights)
    return sample_weights.tolist()


