import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization (AdaLN).
    Replaces GroupNorm to be robust to zero-padding in variable grid sizes.
    """
    def __init__(self, num_channels, emb_dim):
        super().__init__()
        self.num_channels = num_channels

        # Standard LayerNorm (affine=False because we predict scale/shift)
        self.norm = nn.LayerNorm(num_channels, elementwise_affine=False)

        # Projection layer: Logic Embedding -> (Scale + Shift)
        self.emb_proj = nn.Linear(emb_dim, num_channels * 2)

        # Initialize to identity
        nn.init.zeros_(self.emb_proj.weight)
        nn.init.zeros_(self.emb_proj.bias)

    def forward(self, x, emb):
        """
        x: [B, C, H, W]
        emb: [B, Emb_Dim]
        """
        # 1. Permute to [B, H, W, C] for LayerNorm
        x_perm = x.permute(0, 2, 3, 1)

        # 2. Apply Standard Norm
        x_norm = self.norm(x_perm)

        # 3. Predict Scale and Shift
        emb_out = self.emb_proj(emb)  # [B, 2*C]

        # Reshape for broadcasting over H, W: [B, 1, 1, 2*C]
        emb_out = emb_out.unsqueeze(1).unsqueeze(1)
        scale, shift = emb_out.chunk(2, dim=-1)  # Each is [B, 1, 1, C]

        # 4. Apply Affine (Modulate)
        out = x_norm * (1 + scale) + shift

        # 5. Permute back to [B, C, H, W]
        return out.permute(0, 3, 1, 2)


class CanvasCoreLayer(nn.Module):
    """
    One layer of the Canvas Core with LOCAL spatial convolutions (NCA-style)
    and cross-attention to logic tokens.

    Uses AdaptiveGroupNorm to condition the CNN on the puzzle embedding,
    allowing task-specific feature modulation.
    """

    def __init__(self, d_model, n_heads, ff_dim, dropout, img_size):
        super().__init__()
        self.img_size = img_size
        self.d_model = d_model

        # Conv Block 1
        self.conv1 = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)
        self.adagn1 = AdaptiveLayerNorm(d_model, emb_dim=d_model)
        self.act1 = nn.ReLU()

        # Conv Block 2
        self.conv2 = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)
        self.adagn2 = AdaptiveLayerNorm(d_model, emb_dim=d_model)

        # Ignition Initialization - prevent dead grid
        nn.init.constant_(self.conv1.bias, 0.1)
        nn.init.constant_(self.conv2.bias, 0.1)

        # Cross-attention: Canvas attending to Logic context (global reasoning)
        self.cross_att = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        # Feed-forward network
        self.linear1 = nn.Linear(d_model, ff_dim)
        self.linear2 = nn.Linear(ff_dim, d_model)

        # Layer norms (these are per-sample, safe for any batch size)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        # Gating parameter for cross-attention output (start small for stability)
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, logic_context, puzzle_emb, valid_mask=None, logic_pad_mask=None, stream_dropout=0.0):
        """
        x: Tensor (batch, H*W, d_model) for canvas tokens.
        logic_context: Tensor (batch, L_len, d_model) for logic tokens.
        puzzle_emb: Tensor (batch, d_model) for task conditioning.
        valid_mask: [B, H*W, 1] - Binary mask (1=valid, 0=pad). CRITICAL for bias bleeding fix.
        logic_pad_mask: Boolean mask (batch, L_len) for dummy logic tokens.
        """
        B, C_len, D = x.shape
        H = W = self.img_size

        # Prepare mask for grid operations [B, 1, H, W]
        if valid_mask is not None:
            mask_grid = valid_mask.transpose(1, 2).reshape(B, 1, H, W)
        else:
            mask_grid = None

        # === 1. Conditioned Local Spatial Mixing ===
        x_norm1 = self.norm1(x)

        # FIX: Mask bias bleeding immediately after norm
        if valid_mask is not None:
            x_norm1 = x_norm1 * valid_mask

        # Reshape tokens to grid: [B, H*W, D] -> [B, D, H, W]
        x_grid = x_norm1.transpose(1, 2).reshape(B, D, H, W)

        # HARDENING: Mask conv INPUT to prevent boundary bleeding
        if mask_grid is not None:
            x_grid = x_grid * mask_grid

        # Conv Block 1 with AdaLN conditioning
        h = self.conv1(x_grid)
        h = self.adagn1(h, puzzle_emb)  # Conditioned on Task!
        h = self.act1(h)
        if mask_grid is not None:
            h = h * mask_grid  # Mask activations

        # Conv Block 2 with AdaLN conditioning
        # HARDENING: Mask between conv layers too
        if mask_grid is not None:
            h = h * mask_grid
        h = self.conv2(h)
        h = self.adagn2(h, puzzle_emb)  # Conditioned on Task!
        if mask_grid is not None:
            h = h * mask_grid  # Mask output of block 2

        # Reshape back to tokens: [B, D, H, W] -> [B, H*W, D]
        local_conv_out = h.reshape(B, D, C_len).transpose(1, 2)

        # Residual connection
        x = x + self.dropout(local_conv_out)

        # === 2. Cross-Attention: Canvas attending to Logic context ===
        x_norm2 = self.norm2(x)
        if valid_mask is not None:
            x_norm2 = x_norm2 * valid_mask  # Mask before attention query

        cross_out, _ = self.cross_att(
            x_norm2, logic_context, logic_context,
            key_padding_mask=logic_pad_mask
        )

        # Stream dropout: optionally drop logic influence for some samples
        if self.training and stream_dropout > 0:
            keep_probs = torch.rand(B, device=x.device)
            keep_mask = (keep_probs >= stream_dropout).float().view(-1, 1, 1)
            cross_out = cross_out * keep_mask

        # Gated residual connection
        x = x + self.dropout(self.gate * cross_out)

        # === 3. Feed-forward ===
        x_norm3 = self.norm3(x)
        if valid_mask is not None:
            x_norm3 = x_norm3 * valid_mask  # Mask before FFN

        ff_out = F.relu(self.linear1(x_norm3))
        ff_out = self.linear2(ff_out)

        # Final residual addition
        x = x + self.dropout(ff_out)

        # Final cleanup mask
        if valid_mask is not None:
            x = x * valid_mask

        return x


class CanvasCore(nn.Module):
    """
    Canvas Core: Updates canvas tokens with LOCAL 2D convolutions (NCA-style)
    and cross-attention to logic tokens.

    Uses AdaptiveGroupNorm (AdaGN) to condition the CNN on puzzle embeddings,
    allowing the Logic Core to "drive" the CNN with task-specific physics rules.

    Now with MASKED processing to prevent hallucination outside grid boundaries.
    """

    def __init__(self, d_model=432, n_heads=8, ff_dim=1728, n_layers=2, dropout=0.1, img_size=30):
        super().__init__()
        self.img_size = img_size
        self.layers = nn.ModuleList([
            CanvasCoreLayer(d_model, n_heads, ff_dim, dropout, img_size)
            for _ in range(n_layers)
        ])

    def forward(self, canvas_tokens, logic_tokens, puzzle_emb, logic_pad_mask=None, stream_dropout=0.0, valid_mask=None):
        """
        canvas_tokens: (batch, H*W, d_model)
        logic_tokens: (batch, L_len, d_model)
        puzzle_emb: (batch, d_model) - Task embedding for AdaLN conditioning
        logic_pad_mask: (batch, L_len) boolean mask for dummy logic tokens.
        valid_mask: (batch, H*W, 1) - Binary mask (1 for valid grid, 0 for pad)
                   Passed INTO each layer to prevent bias bleeding.
        """
        x = canvas_tokens
        for layer in self.layers:
            # Pass valid_mask INTO the layer for internal masking
            x = layer(x, logic_tokens, puzzle_emb, valid_mask=valid_mask,
                     logic_pad_mask=logic_pad_mask, stream_dropout=stream_dropout)

        return x
