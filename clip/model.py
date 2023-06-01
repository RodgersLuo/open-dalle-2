from typing import Tuple

import numpy as np
import torch
from torch import nn

import sys
sys.path.insert(0, 'nn_components')
from transformer import Transformer


class ResBlock(nn.Module):

    def __init__(self, in_ch, out_ch, downsample_stride=1):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Conv2d(out_ch, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None

        if downsample_stride > 1:
            self.downsample = nn.AvgPool2d(downsample_stride)

        self.skip_connection = nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x: torch.Tensor):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.bn3(self.conv3(h))

        x = self.skip_connection(x)
        if self.downsample is not None:
            x = self.downsample(x)
            h = self.downsample(h)

        h = h + x
        h = self.relu3(h)
        return h


class ResNet(nn.Module):

    def __init__(self, layers_strides, output_dim, width):
        super().__init__()
        self.output_dim = output_dim

        self.stem = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # residual layers
        layers = []
        layers.append(ResBlock(width, width * 2, downsample_stride=layers_strides[0]))
        layers.append(ResBlock(width * 2, width * 4, downsample_stride=layers_strides[1]))
        if len(layers_strides) >= 3:
            layers.append(ResBlock(width * 4, width * 8, downsample_stride=layers_strides[2]))
        if len(layers_strides) >= 4:
            layers.append(ResBlock(width * 8, width * 12, downsample_stride=layers_strides[3]))
        self.layers = nn.Sequential(*layers)

        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.fc = nn.Linear(layers[-1].out_ch * 2 * 2, output_dim)
        
    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x = self.avgpool(x)
        x = self.fc(x.flatten(start_dim=1))
        return x


class CLIP(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Tuple[int, int, int, int],
                 vision_width: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 **kwargs
                ):
        super().__init__()

        self.context_length = context_length

        self.image_resolution = image_resolution

        self.resnet = ResNet(
            layers_strides=vision_layers,
            output_dim=embed_dim,
            width=vision_width
        )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            vocab_size=vocab_size,
            context_length=context_length,
            attn_mask=self.build_attention_mask()
        )

        self.text_projection = nn.Parameter(torch.randn(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def build_attention_mask(self):
        """
        Taken from https://github.com/openai/CLIP
        """
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.logit_scale.dtype

    def encode_image(self, image, normalize=False):
        image = self.resnet(image.type(self.dtype))
        if normalize:
            image = image / image.norm(dim=-1, keepdim=True)
        return image

    def encode_text(self, text, normalize=False):
        x = self.transformer(text)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        if normalize:
            x = x / x.norm(dim=-1, keepdim=True)
        return x

    def forward(self, image, text):
        """
        Modified from the pseudocode in the original paper: https://arxiv.org/pdf/2103.00020.pdf
        """
        # extract feature representations of each modality
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)

        # scaled pairwise cosine similarities [n, n]
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [batch_size, batch_size]
        return logits_per_image, logits_per_text


