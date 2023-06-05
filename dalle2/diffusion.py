import math
import torch
import torch.nn.functional as F

class Diffusion:
    @staticmethod
    def linear_beta_schedule(T, start=0.0001, end=0.01):
        return torch.linspace(start, end, T)

    @staticmethod
    def cosine_beta_schedule(T, max_beta=0.999):
        def cosine_cum_func(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        betas = []
        for i in range(T):
            t1 = i / T
            t2 = (i + 1) / T
            beta = 1 - cosine_cum_func(t2) / cosine_cum_func(t1)
            betas.append(min(beta, max_beta))
        return torch.Tensor(betas)

    def __init__(self, T, schedule="linear") -> None:
        self.T = T

        # Define beta schedule
        if schedule == "linear":
            self.betas = self.linear_beta_schedule(T=T)
        elif schedule == "cosine":
            self.betas = self.cosine_beta_schedule(T=T)

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
        Samples from the diffusion process at a specific time step t.
        :param x_0: the initial data
        :param t: the time step to sample from
        :param device: the device to use
        :return: the sampled data and the noise
        """
        noise = torch.randn_like(x_0).to(device)
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        # mean + variance
        return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
        + sqrt_one_minus_alphas_cumprod_t.to(device) * noise, noise

