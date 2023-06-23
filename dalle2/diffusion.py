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
        # bar alpha_t
        self.alphas_bar = torch.cumprod(self.alphas, axis=0)

        # bar alpha_{t-1}
        self.alphas_bar_prev = torch.ones_like(self.alphas_bar)
        self.alphas_bar_prev[1:] = self.alphas_bar[:-1]

        # sqrt(bar alpha_t)
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)

        # 1 / sqrt(alpha_t)
        self.sqrt_alphas_recip = torch.sqrt(1.0 / self.alphas)

        # sqrt(1 - bar alpha_t)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1. - self.alphas_bar)

        # tilde beta_t
        self.posterior_variance = self.betas * (1. - self.alphas_bar_prev) / (1. - self.alphas_bar)

    def retireve_values(self, values, i, x_shape):
        """
        Gets the value from a list of values at a specific index.
        :param values: the list of values
        :param i: the index
        :param x_shape: the shape of the input
        :return: the value at the index
        """
        batch, *_ = i.shape
        out = values.gather(-1, i.cpu())
        shape = [1 for _ in range(len(x_shape - 1))]
        return out.reshape(batch, *tuple(shape)).to(i.device)

    def forward_diffusion_sample(self, x_0, t, device="cpu"):
        """
        Samples from the diffusion process at a specific time step t.
        :param x_0: the initial data
        :param t: the time step to sample from
        :param device: the device to use
        :return: the sampled data and the noise
        """
        noise = torch.randn_like(x_0).to(device)
        mean = self.retireve_values(self.sqrt_alphas_bar, t, x_0.shape).to(device) * x_0.to(device)
        std = self.retireve_values(self.sqrt_one_minus_alphas_bar, t, x_0.shape).to(device) * noise
        return mean + std, noise

