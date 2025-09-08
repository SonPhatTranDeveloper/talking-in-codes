"""Feedforward autoencoder with Residual Vector Quantization.

This module defines a simple multilayer perceptron (MLP) based autoencoder
that inserts a Residual Vector Quantizer (RVQ) bottleneck. It provides small
building blocks used elsewhere in the project.
"""

import logging

import torch
import torch.nn as nn
from vector_quantize_pytorch import ResidualVQ

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeedForward(nn.Module):
    """Simple feedforward MLP block composed of Linear and ReLU layers.

    Args:
        input_dim: Size of the input feature dimension.
        hidden_dims: Sizes of the hidden layers, in order.
        output_dim: Size of the output feature dimension.

    Returns:
        None
    """

    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(hidden_dims)):
            self.layers.append(
                nn.Linear(input_dim if i == 0 else hidden_dims[i - 1], hidden_dims[i])
            )
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass through the MLP.

        Args:
            x: Input tensor of shape (..., input_dim).

        Returns:
            Tensor with shape (..., output_dim) after applying the MLP layers.
        """
        for layer in self.layers:
            x = layer(x)
        return x


class RQVAEAutoencoder(nn.Module):
    """Autoencoder with an RVQ bottleneck.

    The encoder maps inputs to a lower-dimensional representation, which is then
    quantized by a Residual Vector Quantizer. The decoder reconstructs the
    original input from the quantized representation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        num_quantizers: int,
        codebook_size: int,
    ) -> None:
        """Initialize the autoencoder.

        Args:
            input_dim: Size of the input feature dimension.
            hidden_dims: Sizes of the hidden layers used in encoder/decoder.
            output_dim: Size of the latent representation before quantization.
            num_quantizers: Number of residual quantizers in the RVQ stack.
            codebook_size: Number of entries per codebook in the RVQ.

        Returns:
            None
        """
        super().__init__()
        self.encoder = FeedForward(input_dim, hidden_dims, output_dim)
        self.quantizer = ResidualVQ(
            dim=output_dim,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            rotation_trick=True,
        )
        self.decoder = FeedForward(output_dim, hidden_dims[::-1], input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the autoencoder forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            - Q: Quantized tensor of shape (batch_size, output_dim).
            - I: Indices tensor of shape (batch_size, num_quantizers).
            - C: Commitment loss tensor of shape (batch_size, input_dim).
        """
        # Encode the input
        encoded = self.encoder(x)

        # Quantize the encoded input
        quantized, indices, commitment_loss = self.quantizer(encoded)
        # quantized, indices = self.quantizer(encoded)

        # Decode the quantized input
        decoded = self.decoder(quantized)

        # Return the quantized, indices, and commitment loss
        # return decoded, indices, commitment_loss
        return decoded, indices, commitment_loss

    def get_indices(self, x: torch.Tensor) -> torch.Tensor:
        """Get the indices of the quantized input.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Indices tensor of shape (batch_size, num_quantizers).
        """
        with torch.no_grad():
            encoded = self.encoder(x)
            _, indices, _ = self.quantizer(encoded)
        return indices
