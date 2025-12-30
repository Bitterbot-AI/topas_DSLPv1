import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for better relative position encoding."""

    def __init__(self, dim, max_seq_len=512, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # Precompute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Precompute rotary embeddings
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0).unsqueeze(0))  # [1, 1, seq_len, dim]
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0).unsqueeze(0))

    def _rotate_half(self, x):
        """Rotate half the hidden dims of the input."""
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q, k, seq_len):
        """Apply rotary embeddings to queries and keys."""
        if seq_len > self.cos_cached.size(2):
            self._build_cache(seq_len)

        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]

        # Apply rotation
        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)

        return q_rot, k_rot


class RoPEMultiheadAttention(nn.Module):
    """Multi-head attention with Rotary Position Embeddings."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionalEmbedding(self.head_dim)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x, key_padding_mask=None):
        """Self-attention with RoPE."""
        B, L, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        q, k = self.rope(q, k, L)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Apply padding mask if provided
        if key_padding_mask is not None:
            # key_padding_mask: [B, L], True = ignore
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Apply attention to values
        out = torch.matmul(attn_probs, v)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        out = self.out_proj(out)

        return out


class LogicCoreLayer(nn.Module):
    """One layer of the Logic Core Transformer (with self-attn and cross-attn to canvas)."""

    def __init__(self, d_model, n_heads, ff_dim, dropout, use_rope=False):
        super().__init__()
        self.use_rope = use_rope
        if use_rope:
            self.self_att = RoPEMultiheadAttention(d_model, n_heads, dropout=dropout)
        else:
            self.self_att = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_att = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, ff_dim)
        self.linear2 = nn.Linear(ff_dim, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        # Gating parameter for cross-attention output (learned scalar, start small for stability)
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, canvas_context, pad_mask=None, stream_dropout=0.0):
        """
        x: Tensor of shape (batch, L_len, d_model) representing logic tokens.
        canvas_context: Tensor of shape (batch, C_len, d_model) for canvas tokens.
        pad_mask: Boolean mask of shape (batch, L_len) with True for dummy logic tokens to ignore.
        stream_dropout: probability to drop the cross-stream information.
        """
        # Self-attention on logic tokens (allowing logic tokens to interact among themselves)
        x_norm = self.norm1(x)
        # key_padding_mask causes True positions to be ignored (no attention to those positions)
        if self.use_rope:
            self_att_out = self.self_att(x_norm, key_padding_mask=pad_mask)
        else:
            self_att_out, _ = self.self_att(x_norm, x_norm, x_norm, key_padding_mask=pad_mask)
        x = x + self.dropout(self_att_out)  # residual

        # Cross-attention: Logic attending to Canvas context
        x_norm2 = self.norm2(x)
        # No padding mask on canvas_context (canvas tokens are all valid)
        cross_out, _ = self.cross_att(x_norm2, canvas_context, canvas_context)

        # Apply stream dropout: possibly drop cross-stream info for some samples
        if self.training and stream_dropout > 0:
            # Random mask for each sample in the batch: 1 means keep, 0 means drop cross attention
            keep_probs = torch.rand(x.shape[0], device=x.device)
            keep_mask = (keep_probs >= stream_dropout).float().view(-1, 1, 1)
            cross_out = cross_out * keep_mask  # drop entire cross-att output for some samples

        # Gated residual connection from cross-attention
        x = x + self.dropout(self.gate * cross_out)

        # Feed-forward network
        x_norm3 = self.norm3(x)
        ff_out = F.relu(self.linear1(x_norm3))
        ff_out = self.linear2(ff_out)
        x = x + self.dropout(ff_out)
        return x


class LogicCore(nn.Module):
    """Logic Core: a Transformer that processes logic tokens (one per demo) with cross-attention to canvas."""

    def __init__(self, d_model=432, n_heads=8, ff_dim=1728, n_layers=2, dropout=0.1, use_rope=False):
        super().__init__()
        self.layers = nn.ModuleList([
            LogicCoreLayer(d_model, n_heads, ff_dim, dropout, use_rope=use_rope) for _ in range(n_layers)
        ])

    def forward(self, logic_tokens, canvas_tokens, pad_mask=None, stream_dropout=0.0):
        """
        logic_tokens: (batch, L_len, d_model)
        canvas_tokens: (batch, C_len, d_model)
        pad_mask: (batch, L_len) bool mask for dummy logic tokens (True = ignore).
        """
        x = logic_tokens
        for layer in self.layers:
            x = layer(x, canvas_tokens, pad_mask=pad_mask, stream_dropout=stream_dropout)
        return x
