import torch
from torch.utils.data import Dataset
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
