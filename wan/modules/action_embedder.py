"""Discrete camera-action embedder (HunyuanActionEmbedder) — embeds the 81-class
discrete action labels into the timestep modulation signal. Extracted to be
self-contained for the Wan backbone.
"""
import torch
import torch.nn as nn


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """
    Create sinusoidal position embeddings for the given positions.

    Args:
        dim: Embedding dimension
        position: Position values, shape [N] or [batch_size, seq_len]

    Returns:
        Sinusoidal embeddings, shape [N, dim] or [batch_size, seq_len, dim]
    """
    orig_shape = position.shape
    position_flat = position.reshape(-1).type(torch.float64)
    freqs = torch.pow(10000, -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(dim // 2))
    sinusoid = torch.outer(position_flat, freqs)
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=-1)
    x = x.reshape(*orig_shape, dim)
    return x.to(position.dtype)


class HunyuanActionEmbedder(nn.Module):
    """
    Action embedder for HunyuanVideo camera control.
    
    Embeds discrete action labels into continuous vectors that can be added
    to the timestep modulation signal. Uses a two-layer MLP with SiLU activation.
    
    The action space has 81 classes (9 translation * 9 rotation actions).
    """
    
    def __init__(self, hidden_size: int, freq_dim: int = 256, num_actions: int = 81):
        """
        Args:
            hidden_size: Output embedding dimension (should match model's hidden size)
            freq_dim: Dimension for sinusoidal position encoding
            num_actions: Number of discrete action classes (default: 81 = 9*9)
        """
        super().__init__()
        self.freq_dim = freq_dim
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        
        # MLP: freq_dim -> hidden_size -> hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Initialize the output layer to zero for stable training
        nn.init.zeros_(self.mlp[2].weight)
        if self.mlp[2].bias is not None:
            nn.init.zeros_(self.mlp[2].bias)
    
    def forward(self, action_labels: torch.Tensor) -> torch.Tensor:
        """
        Embed action labels.
        
        Args:
            action_labels: Integer action labels, shape [batch_size] or [batch_size, seq_len]
            
        Returns:
            Action embeddings, shape [batch_size, hidden_size] or [batch_size, seq_len, hidden_size]
        """
        # Convert action labels to float and create sinusoidal embeddings
        action_emb = sinusoidal_embedding_1d(self.freq_dim, action_labels.float())
        
        # Pass through MLP
        return self.mlp(action_emb.to(self.mlp[0].weight.dtype))
