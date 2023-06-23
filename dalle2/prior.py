from einops import repeat
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn

from diffusion import Diffusion

import sys
sys.path.insert(0, "nn_components")
from transformer import Transformer


class Prior(nn.Module):
    """
    The Prior network.
    """
    def __init__(
        self,
        clip_emb_dim,
        T,
        diffusion: Diffusion,
        clip_token_dim,
        xf_layers,
        xf_heads,
        clip_context_len,
        **kwargs
    ):
        super().__init__()
        self.clip_dim = clip_emb_dim

        self.T = T
        self.diffusion = diffusion
        assert self.diffusion.T == self.T, "diffusion timesteps must be the same as the number of timesteps in the prior"

        self.clip_context_len = clip_context_len

        self.text_emb_layer = nn.Linear(clip_emb_dim, clip_emb_dim)

        self.text_encodings_layer = nn.Sequential(
            Rearrange("b c l -> b l c"),
            nn.Conv1d(clip_token_dim, 1, 1),
            nn.Tanh(),
            Rearrange("b 1 c -> b c"),
            nn.Linear(clip_context_len, clip_emb_dim),
            Rearrange("b d -> b 1 d")
        )

        self.time_emb_layer = nn.Sequential(
            nn.Embedding(T, clip_emb_dim),
            nn.Linear(clip_emb_dim, clip_emb_dim),
        )

        self.image_emb_layer = nn.Linear(clip_emb_dim, clip_emb_dim)

        self.final_token = nn.Parameter(torch.randn(clip_emb_dim))

        self.transformer = Transformer(width = clip_emb_dim,
                                       n_layers=xf_layers,
                                       heads=xf_heads,
                                       vocab_size=None,
                                       context_length=self.n_tokens,
                                       attn_mask=self.build_causal_mask(),
                                    )

    @property
    def n_tokens(self):
        """
        Return the pre-determined number of tokens in the context.
        """
        return self.n_image_embeds + self.n_text_embeds + self.n_time_embeds + self.n_text_encodings + 1

    def build_causal_mask(self):
        mask = torch.empty(self.n_tokens, self.n_tokens)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def forward(
        self,
        image_embedding,
        t,
        text_embedding,
        text_encodings,
    ):
        """
        Forward pass through the Prior network.

        Args:
            image_embedding: the noisy image embedding
            t: the timestep
            text_embedding: the text embedding
            text_encodings: the text encodings

        Returns:
            the predicted image embedding
        """
        assert (t < self.T).all(), "diffusion timesteps must be less than the total number of timesteps"
        assert(text_encodings.shape[1] == self.clip_context_len), f"text encodings must have a context length of {self.clip_context_len}"

        batch, _ = image_embedding.shape

        text_encodings = self.text_encodings_layer(text_encodings).unsqueeze(1)
        text_embedding = self.text_emb_layer(text_embedding).unsqueeze(1)
        time_embedding = self.time_emb_layer(t).unsqueeze(1)
        image_embedding = self.image_emb_layer(image_embedding).unsqueeze(1)

        final_token = repeat(self.final_token, "d -> b 1 d", b = batch)

        tokens = torch.cat((
            text_encodings,
            text_embedding,
            time_embedding,
            image_embedding,
            final_token
        ), dim = -2)

        tokens = self.transformer(tokens)

        # predict image embedding from the final token embedding
        return tokens[..., -1, :]

    @torch.no_grad()
    def sample(self, text_emb, text_encodings):
        """
        Sample an image embedding from the text embedding.

        Args:
            text_emb: the text embedding
            text_encodings: the text encodings

        Returns:
            the sampled image embedding
        """
        # Generate two image embeddings from the text embedding
        img_emb_noisy1 = torch.randn_like(text_emb)
        img_emb_noisy2 = torch.randn_like(text_emb)

        img_emb1 = self.sample_one(img_emb_noisy1, text_emb, text_encodings)
        img_emb2 = self.sample_one(img_emb_noisy2, text_emb, text_encodings)

        mask = self.dot_product(img_emb1, text_emb) > self.dot_product(img_emb2, text_emb)
        mask = repeat(mask, "b -> b d", d = self.clip_dim)
        return torch.where(mask, img_emb1, img_emb2)


    @torch.no_grad()
    def sample_one(self, img_emb_noisy, text_emb, text_encodings):
        """
        Calls the model to predict the noise in the image and returns
        the denoised image.
        Applies noise to this image, if we are not in the last step yet.

        Args:
            img_emb_noisy: the noisy image embedding
            text_emb: the text embedding
            text_encodings: the text encodings

        Returns:
            the image embedding at previous timestep
        """
        for t in range(self.T)[::-1]:
            ts = torch.full((len(text_emb),), t, dtype=torch.long, device=text_emb.device)
            img_emb_noisy = self(img_emb_noisy, ts, text_embed=text_emb, text_encodings=text_encodings)
            if t != 0:
                posterior_variance_t = self.diffusion.retireve_values(self.diffusion.posterior_variance, ts, text_emb.shape)
                img_emb_noisy += torch.sqrt(posterior_variance_t) * torch.randn_like(img_emb_noisy)
        return img_emb_noisy

    @staticmethod
    def dot_product(a, b):
        return torch.einsum("bd, bd->b", a, b)

