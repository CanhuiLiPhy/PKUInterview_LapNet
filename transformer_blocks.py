"""
Transformer blocks for PyTorch NNVMC.

This module implements transformer architectures similar to lapnet's transformer_blocks.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Sequence

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, 
                 d_model: int,
                 n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # QKV projection
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
        nn.init.zeros_(self.o_proj.bias)
    
    def scaled_dot_product_attention(self, 
                                   q: torch.Tensor, 
                                   k: torch.Tensor, 
                                   v: torch.Tensor,
                                   mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scaled dot-product attention."""
        d_k = q.size(-1)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            # Ensure mask has correct dimensions [batch, n_heads, n_electrons, n_electrons]
            # or [batch, n_electrons, n_electrons] or [n_electrons, n_electrons]
            if mask.dim() == 2:
                # [n_electrons, n_electrons] -> [batch, n_heads, n_electrons, n_electrons]
                mask = mask.unsqueeze(0).unsqueeze(0).expand(scores.shape[0], scores.shape[1], -1, -1)
            elif mask.dim() == 3:
                # [batch, n_electrons, n_electrons] -> [batch, n_heads, n_electrons, n_electrons]
                mask = mask.unsqueeze(1).expand(-1, scores.shape[1], -1, -1)
            # If mask already has 4 dimensions, use as is
            
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        values = torch.matmul(attention, v)
        
        return values, attention
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of multi-head attention.
        
        Args:
            x: Input tensor [batch_size, n_electrons, d_model]
            mask: Attention mask [batch_size, n_electrons, n_electrons]
            
        Returns:
            (output, attention): Output tensor and attention weights
        """
        batch_size, n_electrons, d_model = x.shape
        
        # Compute QKV
        qkv = self.qkv_proj(x)  # [batch, n_electrons, 3*d_model]
        qkv = qkv.reshape(batch_size, n_electrons, self.n_heads, 3 * self.d_k)
        qkv = qkv.transpose(1, 2)  # [batch, n_heads, n_electrons, 3*d_k]
        
        # Split into Q, K, V
        q, k, v = qkv.chunk(3, dim=-1)  # Each: [batch, n_heads, n_electrons, d_k]
        
        # Scaled dot-product attention
        values, attention = self.scaled_dot_product_attention(q, k, v, mask)
        
        # Concatenate heads
        values = values.transpose(1, 2).contiguous()  # [batch, n_electrons, n_heads, d_k]
        values = values.reshape(batch_size, n_electrons, d_model)
        
        # Final projection
        output = self.o_proj(values)
        
        return output, attention

class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross attention for asymmetric interactions."""
    
    def __init__(self, 
                 d_model: int,
                 n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # QK projection for source, V projection for target
        self.qk_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.qk_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
        nn.init.zeros_(self.o_proj.bias)
    
    def forward(self, 
                h_source: torch.Tensor, 
                h_target: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of cross attention.
        
        Args:
            h_source: Source features [batch_size, n_electrons, d_model]
            h_target: Target features [batch_size, n_electrons, d_model]
            mask: Attention mask
            
        Returns:
            Output tensor [batch_size, n_electrons, d_model]
        """
        batch_size, n_electrons, d_model = h_source.shape
        
        # Compute Q, K from source
        qk = self.qk_proj(h_source)  # [batch, n_electrons, 2*d_model]
        qk = qk.reshape(batch_size, n_electrons, self.n_heads, 2 * self.d_k)
        qk = qk.transpose(1, 2)  # [batch, n_heads, n_electrons, 2*d_k]
        q, k = qk.chunk(2, dim=-1)  # Each: [batch, n_heads, n_electrons, d_k]
        
        # Compute V from target
        v = self.v_proj(h_target)  # [batch, n_electrons, d_model]
        v = v.reshape(batch_size, n_electrons, self.n_heads, self.d_k)
        v = v.transpose(1, 2)  # [batch, n_heads, n_electrons, d_k]
        
        # Scaled dot-product attention
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        
        values = torch.matmul(attention, v)
        
        # Concatenate heads
        values = values.transpose(1, 2).contiguous()  # [batch, n_electrons, n_heads, d_k]
        values = values.reshape(batch_size, n_electrons, d_model)
        
        # Final projection
        output = self.o_proj(values)
        
        return output

class LayerNorm(nn.Module):
    """Layer normalization."""
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        
        self.scale = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization."""
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * normalized + self.bias

class LayerNormBlock(nn.Module):
    """Optional layer normalization block."""
    
    def __init__(self, d_model: int, use_layernorm: bool = True):
        super().__init__()
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.norm = LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer norm if enabled, otherwise return input."""
        if self.use_layernorm:
            return self.norm(x)
        else:
            return x

class Dense(nn.Module):
    """Dense linear layer."""
    
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 use_bias: bool = True,
                 activation: Optional[str] = None):
        super().__init__()
        
        self.linear = nn.Linear(in_features, out_features, bias=use_bias)
        self.activation = activation
        
        # Initialize weights
        nn.init.xavier_uniform_(self.linear.weight)
        if use_bias:
            nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.linear(x)
        
        if self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'gelu':
            x = F.gelu(x)
        elif self.activation == 'tanh':
            x = torch.tanh(x)
        
        return x

class TransformerLayer(nn.Module):
    """Single transformer layer with self-attention."""
    
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 d_ff: int,
                 use_layernorm: bool = True):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, n_heads)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            Dense(d_model, d_ff, activation='tanh'),
            Dense(d_ff, d_model)
        )
        
        # Layer normalization
        self.norm1 = LayerNormBlock(d_model, use_layernorm)
        self.norm2 = LayerNormBlock(d_model, use_layernorm)
    
    def forward(self, 
                x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with residual connections."""
        # Self-attention with residual connection
        attn_out, _ = self.attention(self.norm1(x), mask)
        x = x + attn_out
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        
        return x

class CrossAttentionLayer(nn.Module):
    """Cross attention layer for asymmetric interactions."""
    
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 use_layernorm: bool = True):
        super().__init__()
        
        self.attention = MultiHeadCrossAttention(d_model, n_heads)
        
        # Layer normalization
        self.norm1 = LayerNormBlock(d_model, use_layernorm)
        self.norm2 = LayerNormBlock(d_model, use_layernorm)
        self.norm3 = LayerNormBlock(d_model, use_layernorm)
    
    def forward(self, 
                h_source: torch.Tensor, 
                h_target: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass returning only the attention output (no residual)."""
        # Cross attention - just return the attention output
        attn_out = self.attention(self.norm1(h_source), self.norm2(h_target), mask)
        
        return attn_out  # Let the caller handle residual connections

def create_attention_mask(n_electrons: int, 
                         same_spin_mask: bool = False,
                         spins: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Create attention masks for transformer layers.
    
    Args:
        n_electrons: Number of electrons
        same_spin_mask: If True, mask different-spin interactions
        spins: Spin configuration [n_electrons] with +1/-1 values
        
    Returns:
        Attention mask [n_electrons, n_electrons]
    """
    mask = torch.ones(n_electrons, n_electrons)
    
    if same_spin_mask and spins is not None:
        # Mask interactions between different spins
        spin_matrix = spins.unsqueeze(0) * spins.unsqueeze(1)  # +1 for same spin, -1 for different
        mask = (spin_matrix > 0).float()
    
    return mask 