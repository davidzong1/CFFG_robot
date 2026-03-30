import torch
from torch.distributions import Normal


def get_logprob_norm(sample, mu, sigma, safe_get_logprob=False):
    if safe_get_logprob:
        dist = Normal(loc=mu, scale=sigma)
        return dist.log_prob(sample)
    else:
        # logprob = log p(x|mu,sigma) = -log(sigma) - 0.5 * log(2 * pi) - 0.5 * ((x - mu) / sigma) ** 2
        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
        constant_term = -torch.log(sigma_safe) - 0.5 * torch.log(2 * torch.pi * torch.ones_like(sample))
        exponent_term = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)
        log_prob = constant_term + exponent_term
        log_prob = torch.where(mask, torch.zeros_like(log_prob), log_prob)
        return log_prob
