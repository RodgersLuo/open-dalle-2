import torch
from torch import nn

import sys
from decoder import Decoder
from diffusion import Diffusion
from prior import Prior

sys.path.insert(0, 'clip')
from model import CLIP
sys.path.insert(0, "nn_components")
from tokenizer import tokenize


class DALLE2(nn.Module):
    def __init__(self, clip: CLIP, prior: Prior, decoder: Decoder) -> None:
        super().__init__()
        self.clip = clip
        self.prior = prior
        self.decoder = decoder
        self.val_mode()

    @torch.no_grad()
    def forward(self, image_dim, text, cf_guidance_scale=None):
        text_tokens = tokenize(text, context_length=self.clip.context_length)
        text_tokens = text_tokens.to(device=self.device)
        text_embedding, text_encodings = self.clip.encode_text(text_tokens, normalize=True, return_encodings=True)
        image_embedding = self.prior.sample(text_embedding, text_encodings=text_encodings)

        return self.decoder.sample_one(image_dim,
                                        text_tokens,
                                        clip_emb=image_embedding,
                                        cf_guidance_scale=cf_guidance_scale)

    @property
    def device(self):
        return self.prior.learned_query.device

    @torch.no_grad()
    def val_mode(self):
        self.clip.eval()
        for param in self.clip.parameters():
            param.requires_grad = False

        self.prior.eval()
        for param in self.prior.parameters():
            param.requires_grad = False
