from einops import repeat
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn

from diffusion import Diffusion

import sys
sys.path.insert(0, "nn_components")
from transformer import Transformer


class Prior(nn.Module):
    def __init__(
        self,
        clip_emb_dim,
        T,
        clip_token_dim,
        xf_layers,
        xf_heads,
        clip_context_len,
        n_time_embeds = 1,
        n_image_embeds = 1,
        n_text_embeds = 1,
        n_text_encodings = 1,
        **kwargs
    ):
        super().__init__()
        self.clip_dim = clip_emb_dim

        self.T = T
        self.n_time_embeds = n_time_embeds
        self.n_image_embeds = n_image_embeds
        self.n_text_embeds = n_text_embeds
        self.n_text_encodings = n_text_encodings
        self.clip_context_len = clip_context_len

        self.to_text_emb = nn.Sequential(
            nn.Linear(clip_emb_dim, clip_emb_dim * n_text_embeds),
            Rearrange("b (n d) -> b n d", n = n_text_embeds)
        )

        self.to_text_encodings = nn.Sequential(
            Rearrange("b c l -> b l c"),
            nn.Conv1d(clip_token_dim, 1, 1),
            nn.Tanh(),
            Rearrange("b 1 c -> b c"),
            nn.Linear(clip_context_len, clip_emb_dim * n_text_encodings),
            Rearrange("b (n d) -> b n d", n = n_text_encodings)
        )

        self.to_time_emb = nn.Sequential(
            nn.Embedding(T, clip_emb_dim),
            nn.Linear(clip_emb_dim, clip_emb_dim * n_time_embeds),
            Rearrange("b (n d) -> b n d", n = n_time_embeds)
        )

        self.to_image_emb = nn.Sequential(
            nn.Linear(clip_emb_dim, clip_emb_dim * n_image_embeds),
            Rearrange("b (n d) -> b n d", n = n_image_embeds)
        )

        self.query = nn.Parameter(torch.randn(clip_emb_dim))

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
        image_embed,
        diffusion_timesteps,
        text_embed,
        text_encodings,
    ):
        assert (diffusion_timesteps < self.T).all(), "diffusion timesteps must be less than the total number of timesteps"
        assert(text_encodings.shape[1] == self.clip_context_len), f"text encodings must have a context length of {self.clip_context_len}"
        
        batch, _ = image_embed.shape

        text_embed = self.to_text_emb(text_embed)
        image_embed = self.to_image_emb(image_embed)
        text_encodings = self.to_text_encodings(text_encodings)
        time_embed = self.to_time_emb(diffusion_timesteps)

        queries = repeat(self.query, "d -> b 1 d", b = batch)

        tokens = torch.cat((
            text_encodings,
            text_embed,
            time_embed,
            image_embed,
            queries
        ), dim = -2)

        tokens = self.transformer(tokens)

        # predict image embedding from the final token embedding (which is the learned query) 
        return tokens[..., -1, :]

    @torch.no_grad()
    def sample(self, diffusion: Diffusion, text_emb, text_encodings):
        """
        Sample an image embedding from the text embedding.
        """
        # Generate two image embeddings from the text embedding
        img_emb_noisy1 = torch.randn_like(text_emb)
        img_emb_noisy2 = torch.randn_like(text_emb)

        img_emb1 = self.sample_one(diffusion, img_emb_noisy1, text_emb, text_encodings)
        img_emb2 = self.sample_one(diffusion, img_emb_noisy2, text_emb, text_encodings)

        mask = self.dot_product(img_emb1, text_emb) > self.dot_product(img_emb2, text_emb)
        mask = repeat(mask, "b -> b d", d = self.clip_dim)
        return torch.where(mask, img_emb1, img_emb2)


    @torch.no_grad()
    def sample_one(self, diffusion: Diffusion, img_emb_noisy, text_emb, text_encodings):
        """
        Calls the model to predict the noise in the image and returns
        the denoised image.
        Applies noise to this image, if we are not in the last step yet.
        """
        assert self.T == diffusion.T
        for t in range(self.T)[::-1]:
            ts = torch.full((len(text_emb),), t, dtype=torch.long, device=text_emb.device)
            img_emb_noisy = self(img_emb_noisy, ts, text_embed=text_emb, text_encodings=text_encodings)
            if t != 0:
                posterior_variance_t = diffusion.get_index_from_list(diffusion.posterior_variance, ts, text_emb.shape)
                img_emb_noisy += torch.sqrt(posterior_variance_t) * torch.randn_like(img_emb_noisy)
        return img_emb_noisy

    @staticmethod
    def dot_product(a, b):
        return torch.einsum("bd, bd->b", a, b)

