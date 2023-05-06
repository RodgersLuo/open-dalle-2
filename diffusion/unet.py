import torch
from torch import nn
import math


def non_linearity():
    return nn.SiLU()

def normalization(channels):
    return nn.BatchNorm2d(channels)


class Upsample(nn.Module):
    """
    An upsampling layer in UNet.

    :param channels: channels in the inputs and outputs.
    :param out_channels: number of output channels if different from input channels.
    """

    def __init__(self, channels, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.upsample = nn.ConvTranspose2d(self.channels, self.out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = self.upsample(x)
        return x
    

class Downsample(nn.Module):
    """
    A downsampling layer in UNet.

    :param channels: channels in the inputs and outputs.
    :param out_channels: number of output channels if different from input channels.
    """

    def __init__(self, channels, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.downsample = nn.Conv2d(self.channels, self.out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = self.downsample(x)
        return x
    

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, out_ch),
            non_linearity()
        )
        
        if up:
            self.conv1 = nn.Sequential(
                nn.Conv2d(2*in_ch, out_ch, 3, padding=1),
                non_linearity(),
                normalization(out_ch)
            )
            self.transform_h = Upsample(out_ch)
            self.transform_x = Upsample(out_ch)
            self.skip_connection = nn.Conv2d(2*in_ch, out_ch, 1)

        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                non_linearity(),
                normalization(out_ch)
            )
            self.transform_h = Downsample(out_ch)
            self.transform_x = Downsample(out_ch)
            self.skip_connection = nn.Conv2d(in_ch, out_ch, 1)

        self.conv2 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                non_linearity(),
                normalization(out_ch)
        )
        
    def forward(self, x, t):
        # First Conv
        h = self.conv1(x)
        # Time embedding
        time_emb = self.time_mlp(t)
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.conv2(h)
        # Down or Upsample
        h = self.transform_h(h)

        # Residual connection
        x = self.skip_connection(x)
        x = self.transform_x(x)
        return x + h


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class UNet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self):
        super().__init__()
        image_channels = 3
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 3 
        time_emb_dim = 32

        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )
        
        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([ResidualBlock(down_channels[i], down_channels[i+1], \
                                    time_emb_dim) \
                    for i in range(len(down_channels)-1)])
        # Upsample
        self.ups = nn.ModuleList([ResidualBlock(up_channels[i], up_channels[i+1], \
                                        time_emb_dim, up=True) \
                    for i in range(len(up_channels)-1)])
        
        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)           
            x = up(x, t)
        return self.output(x)