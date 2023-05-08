import torch
import torch.nn.functional as F

class Diffusion:
    @staticmethod
    def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
        return torch.linspace(start, end, timesteps)

    def __init__(self, T=300) -> None:
        # Define beta schedule
        self.betas = self.linear_beta_schedule(timesteps=T)

        # Pre-calculate different terms for closed form
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) \
            / (1. - self.alphas_cumprod)

    def get_index_from_list(self, vals, t, x_shape):
        """
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def forward_diffusion_sample(self, x_0, t, device="cpu"):
        """
        Takes an image and a timestep as input and
        returns the noisy version of it
        """
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        # mean + variance
        return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
        + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)


@torch.no_grad()
def sample_timestep(x, t, tokens, model, diffusion, cf_guidance=False):
    """
    Calls the model to predict the noise in the image and returns
    the denoised image.
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = diffusion.get_index_from_list(diffusion.betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = diffusion.get_index_from_list(
        diffusion.sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = diffusion.get_index_from_list(diffusion.sqrt_recip_alphas, t, x.shape)

    # if cf_guidance:
    #     null_token = torch.zeros(tokens.shape[1])

    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t, tokens=tokens) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = diffusion.get_index_from_list(diffusion.posterior_variance, t, x.shape)

    if t == 0:
        # The t's are offset from the t's in the paper
        return model_mean
    else:
        return model_mean + torch.sqrt(posterior_variance_t) * torch.randn_like(x)
