from torch.utils.data import Dataset
import os
import torch
from PIL import Image
import pandas as pd
import logging



# Define custom dataset

class MyDataset(Dataset):
    def __init__(self, data, img_dir, class_mapping, columns=None, transform=None):
        """
        Initializes the dataset.
        :param csv_file: Path to the CSV file containing data.
        :param img_dir: Directory where images are stored.
        :param class_mapping: Dictionary mapping class names to numeric values.
        :param columns: List of column names to include as features. If None, all columns are included.
        :param transform: Optional transform to be applied on a sample.
        """

        if isinstance(data, str):
            self.data_frame = pd.read_csv(data, low_memory=False)
        elif isinstance(data, pd.DataFrame):
            self.data_frame = data
        else:
            raise ValueError("Data should be a filepath or a pandas DataFrame.")


        self.img_dir = img_dir
        self.transform = transform
        # Use specified columns if provided, otherwise use all columns starting from the third column
        if columns is not None:
            self.csv_data = self.data_frame[columns]
            self.number_of_csv_columns = len(self.csv_data.columns)
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


        csv_data_row = self.csv_data.iloc[idx]
        csv = torch.tensor(csv_data_row.values, dtype=torch.float)
        target = self.targets[idx]

        return (image, csv), target
    

        
    def get_class_distribution(self):
        """
        Returns a count of how many instances exist for each class in the dataset.
        """
        return self.data_frame['target'].value_counts().to_dict()