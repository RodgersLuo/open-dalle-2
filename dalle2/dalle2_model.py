import torch
from torch import nn

import sys
from decoder import Decoder
from diffusion import Diffusion
from prior import DiffusionPriorNetwork

sys.path.insert(0, 'clip')
from model import CLIP
sys.path.insert(0, "nn_components")
from tokenizer import tokenize


class DALLE2(nn.Module):
    def __init__(self, clip: CLIP, prior: DiffusionPriorNetwork, decoder: Decoder) -> None:
        super().__init__()
        self.clip = clip
        self.prior = prior
        self.decoder = decoder
        self.val_mode(self.clip)
        self.val_mode(self.prior)
        self.val_mode(self.decoder)

    @torch.no_grad()
    def forward(self, image_dim, text, prior_diffusion: Diffusion, decoder_diffusion: Diffusion, cf_guidance_scale=None):
        text_tokens = tokenize(text, context_length=self.clip.context_length)
        text_tokens = text_tokens.to(device=self.device)
        text_embedding = self.clip.encode_text(text_tokens, normalize=True)
        image_embedding = self.prior.sample(prior_diffusion, text_embedding, text_encodings=text_tokens)

        return self.decoder.sample_one(image_dim,
                                        text_tokens,
                                        clip_emb=image_embedding,
                                        diffusion=decoder_diffusion,
                                        cf_guidance_scale=cf_guidance_scale)

    @property
    def device(self):
        return self.clip.positional_embedding.device

    @staticmethod
    @torch.no_grad()
    def val_mode(module):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False
