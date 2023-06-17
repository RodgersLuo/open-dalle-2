import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import os
import pandas as pd
from PIL import Image
import numpy as np

import sys
sys.path.insert(0, 'nn_components')
from tokenizer import tokenize


class ImageCaptionDataset(Dataset):
    def __init__(self, images_dir="../data/train/images",
                 table_path="../data/train/data.csv",
                 transform=None,
                 shuffle=False,
                 clip=None,
                 context_length=None,
                 normalize_clip_embeddings=False,
                 return_image_only=False
                 ):

        self.transform = transform
        self.root_dir = images_dir
        self.return_image_only = return_image_only

        self.df = pd.read_csv(table_path)

        self.captions = self.df['caption']
        self.images = self.df['image']

        if shuffle:
            pmt = np.random.permutation(len(self.df))
            self.captions = self.captions.iloc[pmt].reset_index(drop=True)
            self.images = self.images.iloc[pmt].reset_index(drop=True)

        self.clip = clip
        if clip is not None:
            assert context_length is not None
            self.context_length = context_length
            self.normalize = normalize_clip_embeddings
            self.image_embeddings = [None] * len(self.images)
            self.text_embeddings = [None] * len(self.captions)
            self.text_encodings = [None] * len(self.captions)

        print(f"Dataset size: {len(self.captions)}")

    def image_only(self, return_image_only):
        self.return_image_only = return_image_only

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.captions[index]
        image = self.images[index]

        img = Image.open(os.path.join(self.root_dir,image)).convert("RGB")

        if (self.transform):
            img = self.transform(img)
        
        if self.return_image_only:
            return img

        if self.clip is not None:
            if self.image_embeddings[index] is None:
                self.image_embeddings[index] = self.clip.encode_image(img[None, ...], normalize=self.normalize).detach().squeeze()

                tokens = tokenize([caption], context_length=self.context_length)
                text_embedding, text_encoding = self.clip.encode_text(tokens, normalize=self.normalize, return_encodings=True)
                self.text_embeddings[index] = text_embedding.detach().squeeze()
                self.text_encodings[index] = text_encoding.detach().squeeze()

            clip_embeds = {
                "image_embedding": self.image_embeddings[index].clone(),
                "text_embedding": self.text_embeddings[index].clone(),
                "text_encoding": self.text_encodings[index].clone()
            }
            return img, caption, clip_embeds

        return img, caption


def load_data(img_size=32, root_dir="./data", clip=None, context_length=None, normalize_clip_embeddings=False):
    # transform = transforms.Compose(
    #         [
    #             transforms.ToTensor(),
    #         ]
    #     )

    data_transforms = [
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data into [0,1]
        # transforms.RandomRotation(degrees=180, fill=128),
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)

    train_img_dir = os.path.join(root_dir, "train/images")
    train_table_dir = os.path.join(root_dir, "train/data.csv")
    test_img_dir = os.path.join(root_dir, "test/images")
    test_table_dir = os.path.join(root_dir, "test/data.csv")

    train_dataset = ImageCaptionDataset(train_img_dir,
                                        train_table_dir,
                                        transform=data_transform,
                                        clip=clip,
                                        context_length=context_length,
                                        normalize_clip_embeddings=normalize_clip_embeddings)
    test_dataset = ImageCaptionDataset(test_img_dir,
                                       test_table_dir,
                                       transform=data_transform,
                                       clip=clip,
                                       context_length=context_length,
                                       normalize_clip_embeddings=normalize_clip_embeddings)

    # train_dataset= torchvision.datasets.CIFAR10(root=".", download=True, transform=data_transform)
    # test_dataset = None

    return train_dataset, test_dataset
