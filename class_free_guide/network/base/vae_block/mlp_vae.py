import torch
import torch.nn as nn
from typing import Optional
from class_free_guide.network.base.mlp import MLP


class Vae(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        output_dim: int,
        encoder_model: Optional[nn.Module] = None,
        decoder_model: Optional[nn.Module] = None,
        encoder_hidden_dim: Optional[list[int]] = [512, 256],
        decoder_hidden_dim: Optional[list[int]] = [256, 512],
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        if encoder_model is not None:
            self.encoder = encoder_model
        else:
            self.encoder = MLP(input_dim, 2 * latent_dim, encoder_hidden_dim, activation="swish")

        if decoder_model is not None:
            self.decoder = decoder_model
        else:
            self.decoder = MLP(latent_dim, output_dim, decoder_hidden_dim, activation="swish")

    def forward(self, x):
        """ban the forward method, since we only want to use the encode and decode methods"""
        raise NotImplementedError("Vae.forward() is disabled; call encoder/decoder directly.")

    def encode(self, x):
        mu_var = self.encoder(x)
        mu, log_var = torch.chunk(mu_var, 2, dim=-1)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, log_var

    def decode(self, z):
        return self.decoder(z)

    def cal_vae_loss(self, recon_x, x, mu, log_var):
        """
        calculate the VAE loss, which is the sum of the reconstruction loss and the KL divergence loss.
            - recon_loss: the mean squared error between the reconstructed output and the original input
            - kl_loss: the KL divergence between the latent distribution and the standard normal distribution
        Args:
            recon_x: the reconstructed output from the decoder
            x: the original input
            mu: the mean of the latent distribution
            log_var: the log variance of the latent distribution
        Returns:
            The total VAE loss, which is the sum of the reconstruction loss and the KL divergence loss.
        """
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + kl_loss
