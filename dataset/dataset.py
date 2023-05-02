import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import pandas as pd
from PIL import Image
import numpy as np


class ImageCaptionDataset(Dataset):
    def __init__(self, images_dir="../data/train/images",
                 table_path="../data/train/data.csv",
                 transform=None,
                 shuffle=False):

        self.transform = transform
        self.root_dir = images_dir

        self.df = pd.read_csv(table_path)

        self.captions = self.df['caption']
        self.images = self.df['image']

        if shuffle:
            pmt = np.random.permutation(len(self.df))
            self.captions = self.captions.iloc[pmt].reset_index(drop=True)
            self.images = self.images.iloc[pmt].reset_index(drop=True)

        print(len(self.captions))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.captions[index]
        image = self.images[index]

        img = Image.open(os.path.join(self.root_dir,image)).convert("RGB")

        if (self.transform):
            img = self.transform(img)

        return img, caption


def load_data(img_size=64, root_dir="./data"):
    # transform = transforms.Compose(
    #         [
    #             transforms.ToTensor(),
    #         ]
    #     )

    data_transforms = [
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data into [0,1]
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)

    train_img_dir = os.path.join(root_dir, "train/images")
    train_table_dir = os.path.join(root_dir, "train/data.csv")
    test_img_dir = os.path.join(root_dir, "test/images")
    test_table_dir = os.path.join(root_dir, "test/data.csv")

    train_dataset = ImageCaptionDataset(train_img_dir, train_table_dir, transform=data_transform)
    test_dataset = ImageCaptionDataset(test_img_dir, test_table_dir, transform=data_transform)

    return train_dataset, test_dataset
