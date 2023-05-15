import torch
from torch import nn

from unet import UNet

class Decoder(nn.Module):
    def __init__(self,
                 clip,
                 unet
                 ) -> None:
        super().__init__()
        self.clip = clip
        self.unet = unet
