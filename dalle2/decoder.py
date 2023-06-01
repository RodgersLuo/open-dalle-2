import torch
from torch import nn
import math
from einops.layers.torch import Rearrange
from diffusion import Diffusion

import sys
sys.path.insert(0, 'nn_components')
from transformer import Transformer
from tokenizer import tokenize


def non_linearity():
    return nn.SiLU()

def normalization(channels):
    return nn.BatchNorm2d(channels)

def zero_params(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


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


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads,
        num_head_channels=-1,
        encoder_channels=None,
        use_checkpoint=True,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv1d(channels, channels*3, 1)
        self.attention = QKVAttention(self.num_heads)
        if encoder_channels is not None:
            self.encoder_kv = nn.Conv1d(encoder_channels, channels * 2, 1)
        self.proj_out = zero_params(nn.Conv1d(channels, channels, 1))

    # def forward(self, x, encoder_out=None):
    #     return checkpoint(self._forward, (x, encoder_out), self.parameters(), self.use_checkpoint)

    def forward(self, x, encoder_out=None):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        if encoder_out is None:
            h = self.attention(qkv)
        else:
            encoder_out = self.encoder_kv(encoder_out)
            h = self.attention(qkv, encoder_out)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


# class QKVAttentionLegacy(nn.Module):
#     """
#     A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
#     """

#     def __init__(self, n_heads):
#         super().__init__()
#         self.n_heads = n_heads

#     def forward(self, qkv):
#         """
#         Apply QKV attention.

#         :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
#         :return: an [N x (H * C) x T] tensor after attention.
#         """
#         bs, width, length = qkv.shape
#         assert width % (3 * self.n_heads) == 0
#         ch = width // (3 * self.n_heads)
#         q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
#         scale = 1 / math.sqrt(math.sqrt(ch))
#         weight = torch.einsum(
#             "bct,bcs->bts", q * scale, k * scale
#         )  # More stable with f16 than dividing afterwards
#         weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
#         a = torch.einsum("bts,bcs->bct", weight, v)
#         return a.reshape(bs, -1, length)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, encoder_kv=None):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        if encoder_kv is not None:
            assert encoder_kv.shape[1] == self.n_heads * ch * 2
            ek, ev = encoder_kv.reshape(bs * self.n_heads, ch * 2, -1).split(ch, dim=1)
            k = torch.cat([ek, k], dim=-1)
            v = torch.cat([ev, v], dim=-1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class UNetLayer(nn.Module):
    def __init__(self, *args) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(list(args))

    def forward(self, x, embedding, xf_out):
        for block in self.blocks:
            x = self.block_forward(block, x, embedding, xf_out)
        return x

    @staticmethod
    def block_forward(block, x, embedding, xf_out):
        if isinstance(block, ResidualBlock):
            return block(x, embedding)
        if isinstance(block, AttentionBlock):
            return block(x, xf_out)
        raise RuntimeError("The UNet block can only be a residual block or attention block.")


class UNet(nn.Module):
    def __init__(self,
                 down_channels,
                 time_emb_dim,
                 clip_emb_dim,
                 n_vocab,
                 context_length,
                 transformer_width,
                 transformer_layers,
                 transformer_heads,
                 qkv_heads,
                 n_clip_tokens=4,
                 image_channels=3,
                 out_dim=3,
                 **kwargs
                 ):
        super().__init__()
        self.image_channels = image_channels
        up_channels = down_channels[::-1]
        self.out_dim = out_dim
        self.clip_emb_dim = clip_emb_dim

        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                non_linearity()
        )

        # Project CLIP embedding to timestep embedding
        self.clip_emb_to_time_emb = nn.Sequential(
            nn.Linear(clip_emb_dim, time_emb_dim),
            non_linearity(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Project CLIP embedding to n text tokens
        self.clip_emb_to_tokens = nn.Sequential(
            nn.Linear(clip_emb_dim, context_length * n_clip_tokens),
            Rearrange('b (n d) -> b n d', n = n_clip_tokens)
        )

        # Transformer encoder
        self.transformer = Transformer(
                transformer_width,
                transformer_layers,
                transformer_heads,
                vocab_size=n_vocab,
                context_length=context_length,
                attn_mask=None
        )
        self.final_ln = nn.LayerNorm(transformer_width)
        self.token_embedding = nn.Embedding(n_vocab, transformer_width)
        self.positional_embedding = nn.Parameter(torch.rand(context_length, transformer_width, dtype=torch.float))
        self.transformer_proj = nn.Linear(transformer_width, time_emb_dim)

        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        downs = []
        for i in range(len(down_channels)-1):
            down = UNetLayer(
                ResidualBlock(down_channels[i], down_channels[i+1], time_emb_dim),
                AttentionBlock(down_channels[i+1], num_heads=qkv_heads, encoder_channels=transformer_width + n_clip_tokens)
            )
            downs.append(down)
        self.downs = nn.ModuleList(downs)

        # Upsample
        ups = []
        for i in range(len(up_channels)-1):
            up = UNetLayer(
                ResidualBlock(up_channels[i], up_channels[i+1], time_emb_dim, up=True),
                AttentionBlock(up_channels[i+1], num_heads=qkv_heads, encoder_channels=transformer_width + n_clip_tokens)
            )
            ups.append(up)
        self.ups = nn.ModuleList(ups)

        self.output = zero_params(nn.Conv2d(up_channels[-1], out_dim, 1))

    def embed_tokens(self, tokens):
        """
        The output of this transformer is used in two ways: first, the final token embedding is used
        in place of a class embedding in the ADM model; second, the last layer of token embeddings
        (a sequence of K feature vectors) is separately projected to the dimensionality of
        each attention layer throughout the ADM model, and then concatenated to the attention context at each layer.
        """
        assert tokens is not None

        # xf_in = self.token_embedding(tokens.long())
        # xf_in = xf_in + self.positional_embedding[None]
        # xf_out = self.transformer(xf_in)
        # xf_out = self.final_ln(xf_out)
        xf_out = self.transformer(tokens)
        xf_proj = self.transformer_proj(xf_out[:, -1])
        xf_out = xf_out.permute(0, 2, 1)  # NLC -> NCL
        return (xf_proj, xf_out)

    def forward(self, x, timestep, tokens, clip_emb):
        # Embed time
        embedding = self.time_mlp(timestep)

        # Embed CLIP embeddings in timestep embeddings
        assert clip_emb.shape[-1] == self.clip_emb_dim
        # embedding += self.clip_emb_to_time_emb(clip_emb)

        # Embed text tokens
        xf_proj, xf_out = self.embed_tokens(tokens)
        embedding = embedding + xf_proj.to(embedding)

        # Concat CLIP embeddings to text tokens
        xf_out = torch.cat((xf_out, self.clip_emb_to_tokens(clip_emb)), dim=-2)

        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, embedding, xf_out)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, embedding, xf_out)

        # Output conv
        return self.output(x)


class Decoder(nn.Module):
    def __init__(self,
                 unet,
                 num_timesteps,
                 clip=None
                 ) -> None:
        super().__init__()
        self.num_timesteps = num_timesteps
        self.unet = unet

    def forward(self, x, timestep, tokens, clip_emb):
        assert (timestep < self.num_timesteps).all(), f"timestep must be less than num_timesteps ({self.num_timesteps})"
        return self.unet(x, timestep, tokens, clip_emb)

    @torch.no_grad()
    def sample_one(self, image_dimensions, text_tokens, clip_emb, diffusion: Diffusion, cf_guidance_scale=None, stepsize=1):
        device = self.device
        image = torch.randn(1, *image_dimensions, device=device)

        for i in range(0, self.num_timesteps, stepsize)[::-1]:
            t = torch.full((1,), i, device=device, dtype=torch.long)
            image = self.sample_timestep(image, t, text_tokens, clip_emb, diffusion, cf_guidance_scale=cf_guidance_scale)
            image = torch.clamp(image, -1.0, 1.0)
        return image


    @torch.no_grad()
    def sample_timestep(self, x, t, tokens, clip_emb, diffusion: Diffusion, cf_guidance_scale=None):
        """
        Calls the model to predict the noise in the image and returns
        the denoised image.
        Applies noise to this image, if we are not in the last step yet.
        """
        assert len(x) == 1
        assert len(tokens) == 1
        assert len(clip_emb) == 1
        assert (t < self.num_timesteps).all(), f"timestep must be less than num_timesteps ({self.num_timesteps})"

        betas_t = diffusion.get_index_from_list(diffusion.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = diffusion.get_index_from_list(
            diffusion.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = diffusion.get_index_from_list(diffusion.sqrt_recip_alphas, t, x.shape)

        if cf_guidance_scale is None:
            # No classfier-free guidance
            noise = self(x, t, tokens=tokens, clip_emb=clip_emb)
        else:
            # With classifier-free guidance
            null_token = torch.zeros_like(tokens, dtype=tokens.dtype, device=tokens.device)
            null_clip_emb = torch.zeros_like(clip_emb, dtype=clip_emb.dtype, device=clip_emb.device)

            # The predicted noise with conditioning
            noise_label = self(x, t, tokens=tokens, clip_emb=clip_emb)

            # The predicted noise without conditioning
            noise = self(x, t, tokens=null_token, clip_emb=null_clip_emb)
            delta = cf_guidance_scale * (noise_label - noise)
            noise = noise + delta

        # Call model (current image - noise prediction)
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * noise / sqrt_one_minus_alphas_cumprod_t
        )
        posterior_variance_t = diffusion.get_index_from_list(diffusion.posterior_variance, t, x.shape)

        if t == 0:
            # The t's are offset from the t's in the paper
            return model_mean
        else:
            return model_mean + torch.sqrt(posterior_variance_t) * torch.randn_like(x)


    @property
    def device(self):
        return self.unet.positional_embedding.device
