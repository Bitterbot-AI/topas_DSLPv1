"""
TOPAS-DSPL: Recursive Dual-Stream Architecture


Merges:
1. DSPL Dual-Stream: Explicit Canvas (Spatial/CNN) vs Logic (Abstract/Transformer) separation.
2. TOPAS Recursion: Iterative refinement loops (H/L cycles) with distinct state persistence.

Key Feature: Dynamic AdaLN Conditioning
The Logic Core "programs" the Canvas Core at every step by modulating its CNN weights
via Adaptive Layer Norm (AdaLN), derived from the current Logic state.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# Import components from existing codebase
from logic_core import LogicCore
from canvas_core import CanvasCore
from sparse_embedding import CastedSparseEmbedding


# =============================================================================
# Configuration & Constants
# =============================================================================

PAD_ID = 0
EOS_ID = 1
# V3 uses 11 classes (0-9 colors + 10 PAD)
NUM_COLORS = 11

@dataclass
class DualStateCarry:
    """
    Persisted state for the recursive loop.
    z_C: Canvas State [B, H*W, D] (Spatial)
    z_L: Logic State  [B, Seq_Len, D] (Abstract)
    """
    z_C: torch.Tensor
    z_L: torch.Tensor
    steps: torch.Tensor
    halted: torch.Tensor

    # Cache static data to avoid passing it around
    input_grid: Optional[torch.Tensor] = None
    task_ids: Optional[torch.Tensor] = None


# =============================================================================
# Encoders (Adapted from DSPL for Recursion)
# =============================================================================

class LayerNorm2d(nn.Module):
    """LayerNorm for 2D feature maps (B, C, H, W)."""
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class SpatialEncoder(nn.Module):
    """Encodes 2D grid inputs into the initial Canvas State (z_C)."""
    def __init__(self, num_colors=11, d_model=384, img_size=30):
        super().__init__()
        self.d_model = d_model
        self.img_size = img_size
        # Input: Colors + Coordinate channels
        in_channels = num_colors + 2

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            LayerNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            LayerNorm2d(128),
            nn.ReLU(),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(128, d_model, kernel_size=1),
            LayerNorm2d(d_model)
        )

        # Positional Embeddings
        self.row_embed = nn.Parameter(torch.randn(img_size, d_model) * 0.02)
        self.col_embed = nn.Parameter(torch.randn(img_size, d_model) * 0.02)

    def forward(self, grid_onehot, valid_mask=None):
        """
        Args:
            grid_onehot: [B, C, H, W] one-hot encoded grid
            valid_mask: [B, 1, H, W] binary mask (1=valid, 0=pad)
        Returns:
            z_C: [B, H*W, D] canvas tokens
        """
        B, C, H, W = grid_onehot.shape
        device = grid_onehot.device

        # Relative Coords
        y_coords = torch.linspace(-1, 1, H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
        x_coords = torch.linspace(-1, 1, W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)

        x = torch.cat([grid_onehot, y_coords, x_coords], dim=1)
        x = self.stem(x)
        x = self.proj(x)  # [B, D, H, W]

        # Add Positional Embeddings
        x = x.permute(0, 2, 3, 1)  # [B, H, W, D]
        pos = self.row_embed[:H].unsqueeze(1) + self.col_embed[:W].unsqueeze(0)
        x = x + pos.unsqueeze(0)

        # Flatten to tokens [B, H*W, D]
        z_C = x.reshape(B, H * W, self.d_model)

        if valid_mask is not None:
            # Mask invalid pixels (padding)
            mask_flat = valid_mask.reshape(B, H * W, 1)
            z_C = z_C * mask_flat

        return z_C


class DemoEncoder(nn.Module):
    """Encodes Input/Output pairs into Logic Tokens."""
    def __init__(self, num_colors=11, d_model=384, img_size=30):
        super().__init__()
        self.spatial = SpatialEncoder(num_colors, d_model, img_size)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, inp, out):
        """
        Args:
            inp: [B, C, H, W] input grid
            out: [B, C, H, W] output grid
        Returns:
            [B, D] fused demo embedding
        """
        z_in = self.spatial(inp)
        z_out = self.spatial(out)
        # Global pool spatial tokens to get 1 vector per grid
        v_in = z_in.mean(dim=1)
        v_out = z_out.mean(dim=1)

        return self.fusion(torch.cat([v_in, v_out], dim=-1))


# =============================================================================
# Main Merged Model
# =============================================================================

class TOPASDSPLModel(nn.Module):
    """
    Recursive Dual-Stream Model.

    Streams:
    1. z_C (Canvas): Handled by CanvasCore (CNN/NCA). Represents the "Mental Image".
    2. z_L (Logic): Handled by LogicCore (Transformer). Represents the "Algorithm".

    Recursion:
    - H_cycles (Outer): Global re-thinking.
    - L_cycles (Inner): Step-by-step execution.

    Key Innovation: Dynamic AdaLN Conditioning
    At every recursive step, we pool the Logic State (program tokens) to create
    an "instruction vector" that conditions the Canvas Core's CNN via AdaLN.
    This allows the Logic Core to dynamically reprogram the physics of the Canvas Core.
    """

    def __init__(
        self,
        d_model=480,
        n_heads=8,
        expansion=4.0,
        dropout=0.1,
        H_cycles=3,
        L_cycles=4,
        L_layers=2,

        # Data Specs
        img_size=30,
        num_colors=11,  # 0-9 + PAD
        max_demos=3,

        # Puzzle Embedding (TRM Turbo Memory)
        puzzle_emb_ndim=0,
        num_tasks=100000,
        batch_size=32,  # Required for CastedSparseEmbedding

        # Halting
        halt_max_steps=12,
        halt_exploration_prob=0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        self.img_size = img_size
        self.num_colors = num_colors
        self.puzzle_emb_ndim = puzzle_emb_ndim
        self.halt_max_steps = halt_max_steps
        self.halt_exploration_prob = halt_exploration_prob

        ff_dim = int(d_model * expansion)

        # --- 1. Embeddings & Encoders ---
        self.spatial_encoder = SpatialEncoder(num_colors, d_model, img_size)
        self.demo_encoder = DemoEncoder(num_colors, d_model, img_size)

        # Learnable "Program Tokens" (Working memory for the Logic stream)
        self.num_program_tokens = 8
        self.program_token_init = nn.Parameter(torch.randn(1, self.num_program_tokens, d_model) * 0.02)

        # Grid size embedding (HARDENING: explicit grid size info for variable grids)
        # Encodes H and W separately, max 30
        self.grid_size_embed = nn.Embedding(img_size + 1, d_model // 2)
        self.grid_size_proj = nn.Linear(d_model, d_model)

        # Puzzle Embedding (TRM Turbo Memory - SignSGD with high LR)
        # Uses CastedSparseEmbedding for efficient sparse updates
        if puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                num_embeddings=num_tasks,
                embedding_dim=puzzle_emb_ndim,
                batch_size=batch_size,
                init_std=0.01,
                cast_to=torch.float32
            )
            # Project puzzle embedding to model dimension if needed
            if puzzle_emb_ndim != d_model:
                self.puzzle_proj = nn.Linear(puzzle_emb_ndim, d_model)
        else:
            self.puzzle_emb = None

        # --- 2. The Cores (Processors) ---

        # Logic Core: Transformer. Thinks about the task.
        # Inputs: [PuzzleEmb, ProgramTokens, DemoTokens]
        # Cross-attends to Canvas to "see" the grid
        self.logic_core = LogicCore(
            d_model=d_model,
            n_heads=n_heads,
            ff_dim=ff_dim,
            n_layers=L_layers,
            dropout=dropout,
            use_rope=True
        )

        # Canvas Core: CNN/NCA. Manipulates the grid.
        # Inputs: z_C (Grid). Conditioned on z_L via AdaLN.
        self.canvas_core = CanvasCore(
            d_model=d_model,
            n_heads=n_heads,
            ff_dim=ff_dim,
            n_layers=L_layers,
            dropout=dropout,
            img_size=img_size
        )

        # --- 3. Output Heads ---

        # Predict grid from Canvas State
        self.to_pixels = nn.Linear(d_model, num_colors)

        # Halting (Q-Learning) from Logic State
        # Predicts: [Q_halt, Q_continue]
        self.q_head = nn.Linear(d_model, 2)
        nn.init.constant_(self.q_head.bias, -2.0)  # Bias towards continuing

        # Auxiliary heads
        self.object_count_head = nn.Linear(d_model, 21)
        self.centroid_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2),
            nn.Sigmoid()
        )

    def _init_state(
        self,
        train_in: torch.Tensor,
        train_out: torch.Tensor,
        test_in: torch.Tensor,
        task_ids: Optional[torch.Tensor] = None,
        demo_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Initialize z_C and z_L states.

        z_C: Starts from test input grid (spatial context)
        z_L: Starts from demos + puzzle embedding (abstract context)
        """
        B = test_in.shape[0]
        device = test_in.device

        # --- Initialize z_C (Canvas) ---
        # Start with the test input grid
        pad_channel = self.num_colors - 1  # PAD is last channel (index 10)
        # Create mask: 1 where NOT pad, 0 where pad
        valid_mask_2d = 1.0 - test_in[:, pad_channel:pad_channel + 1, :, :]
        z_C = self.spatial_encoder(test_in, valid_mask=valid_mask_2d)  # [B, H*W, D]

        # --- Initialize z_L (Logic) ---
        logic_parts = []

        # 1. Puzzle Embedding (Prior / Long-term memory)
        if self.puzzle_emb is not None and task_ids is not None:
            p_emb = self.puzzle_emb(task_ids)
            if hasattr(self, 'puzzle_proj'):
                p_emb = self.puzzle_proj(p_emb)
            logic_parts.append(p_emb.unsqueeze(1))  # [B, 1, D]

        # 2. Grid Size Token (HARDENING: explicit grid dimensions)
        # Compute actual grid size from valid_mask
        valid_mask_2d_squeezed = valid_mask_2d.squeeze(1)  # [B, H, W]
        # Get height: count rows with any valid pixel
        h_valid = valid_mask_2d_squeezed.any(dim=2).sum(dim=1).long()  # [B]
        # Get width: count cols with any valid pixel
        w_valid = valid_mask_2d_squeezed.any(dim=1).sum(dim=1).long()  # [B]
        # Embed H and W, concat and project
        h_emb = self.grid_size_embed(h_valid)  # [B, D//2]
        w_emb = self.grid_size_embed(w_valid)  # [B, D//2]
        size_emb = self.grid_size_proj(torch.cat([h_emb, w_emb], dim=-1))  # [B, D]
        logic_parts.append(size_emb.unsqueeze(1))  # [B, 1, D]

        # 3. Program Tokens (Working Memory)
        prog = self.program_token_init.expand(B, -1, -1)
        logic_parts.append(prog)

        # 3. Demos (Few-Shot Context)
        # train_in: [B, n_demos, C, H, W]
        demos = []
        n_demos = train_in.shape[1]
        for i in range(n_demos):
            d_vec = self.demo_encoder(train_in[:, i], train_out[:, i])  # [B, D]
            demos.append(d_vec)
        demos = torch.stack(demos, dim=1)  # [B, n_demos, D]
        logic_parts.append(demos)

        z_L = torch.cat(logic_parts, dim=1)  # [B, SeqLen, D]

        return z_C, z_L, valid_mask_2d

    def forward(
        self,
        train_in: torch.Tensor,
        train_out: torch.Tensor,
        test_in: torch.Tensor,
        demo_mask: Optional[torch.Tensor] = None,
        task_ids: Optional[torch.Tensor] = None,
        return_intermediate: bool = False,
        min_steps: int = 0,
        object_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, None], Tuple]:
        """
        Forward pass through the dual-stream recursive architecture.

        Args:
            train_in: [B, n_demos, C, H, W] training input grids
            train_out: [B, n_demos, C, H, W] training output grids
            test_in: [B, C, H, W] test input grid
            demo_mask: [B, n_demos] mask for valid demos (True = invalid)
            task_ids: [B] task identifiers for puzzle embedding
            return_intermediate: Whether to return intermediate outputs

        Returns:
            final_pixels: [B, C, H, W] predicted output grid
            q_logits: [B, 2] Q-values for halting
            None: placeholder for compatibility
        """
        B = test_in.shape[0]
        device = test_in.device
        H, W = test_in.shape[2], test_in.shape[3]

        # 1. Initialization
        z_C, z_L, valid_mask_2d = self._init_state(
            train_in, train_out, test_in, task_ids, demo_mask
        )

        # Flatten valid mask for CanvasCore
        # [B, 1, H, W] -> [B, H*W, 1]
        valid_mask_flat = valid_mask_2d.reshape(B, 1, -1).permute(0, 2, 1)

        # 2. Recursive Loop
        # Structure: H cycles -> L cycles
        # Logic runs, then Canvas runs (conditioned on Logic)

        outputs = []
        q_logits_list = []
        halting_probs = []
        logic_states_hist = []

        # Pad mask for LogicCore (mask out padding demos)
        # Structure of z_L: [Puzzle(1)? | GridSize(1) | Program(8) | Demos(n_demos)]
        logic_pad_mask = None
        if demo_mask is not None:
            # Prefix length = total - demos (automatically correct regardless of structure)
            prefix_len = z_L.shape[1] - demo_mask.shape[1]
            prefix_mask = torch.zeros(B, prefix_len, dtype=torch.bool, device=device)
            logic_pad_mask = torch.cat([prefix_mask, demo_mask], dim=1)

        # Total cycles for ponder cost
        total_cycles = self.H_cycles * self.L_cycles

        for h in range(self.H_cycles):
            for l in range(self.L_cycles):

                # A. Logic Update (Global Reasoning)
                # Logic attends to Canvas to "see" current grid state
                z_L = self.logic_core(
                    logic_tokens=z_L,
                    canvas_tokens=z_C,
                    pad_mask=logic_pad_mask
                )

                # B. Conditioning Vector Generation
                # Pool z_L program tokens to create the "Instruction" for Canvas AdaLN
                # Program tokens are the working memory that encodes the algorithm
                # Structure: [Puzzle(1)? | GridSize(1) | Program(8) | Demos(n)]
                prog_start = (1 if self.puzzle_emb is not None else 0) + 1  # +1 for grid size token
                prog_end = prog_start + self.num_program_tokens
                instruction_vec = z_L[:, prog_start:prog_end].mean(dim=1)  # [B, D]

                # C. Canvas Update (Spatial Execution)
                # Canvas receives instruction via AdaLN inside CanvasCore
                z_C = self.canvas_core(
                    canvas_tokens=z_C,
                    logic_tokens=z_L,
                    puzzle_emb=instruction_vec,  # <--- Dynamic Conditioning!
                    logic_pad_mask=logic_pad_mask,
                    valid_mask=valid_mask_flat
                )

                # D. Outputs & Halting

                # Halting check (Q-Learning style) based on instruction vector
                q_vals = self.q_head(instruction_vec)  # [B, 2]
                q_logits_list.append(q_vals)

                # Convert to grid output
                pixels = self.to_pixels(z_C)  # [B, H*W, C]
                pixels = pixels.reshape(B, H, W, self.num_colors)
                pixels = pixels.permute(0, 3, 1, 2)  # [B, C, H, W]
                outputs.append(pixels)

                # Store logic state for consistency loss
                logic_states_hist.append(instruction_vec)

                # Calculate halt probability for logging
                q_halt = q_vals[:, 0]
                halt_prob = torch.sigmoid(q_halt)
                halting_probs.append(halt_prob)

        # 3. Final Predictions
        final_pixels = outputs[-1]
        final_q = q_logits_list[-1]

        # HARDENING: Force PAD class in padding regions
        # valid_mask_2d is [B, 1, H, W], 1=valid, 0=padding
        # Set padding pixels to have high PAD logit (index num_colors-1) and low others
        pad_mask = (1.0 - valid_mask_2d)  # [B, 1, H, W], 1=padding, 0=valid
        pad_channel = self.num_colors - 1  # PAD is last channel (10)

        # Create PAD logits: high for PAD channel, very negative for others
        pad_logits = torch.full_like(final_pixels, -100.0)  # [B, C, H, W]
        pad_logits[:, pad_channel, :, :] = 100.0  # High logit for PAD

        # Blend: keep original for valid pixels, use PAD logits for padding
        final_pixels = final_pixels * valid_mask_2d + pad_logits * pad_mask

        # Also fix intermediate outputs
        for i in range(len(outputs)):
            outputs[i] = outputs[i] * valid_mask_2d + pad_logits * pad_mask

        # Aux heads on final logic state
        final_logic = logic_states_hist[-1]
        obj_counts = self.object_count_head(final_logic)
        centroids = self.centroid_head(final_logic)

        # Ponder cost (for adaptive compute tracking)
        ponder_cost = torch.tensor(float(total_cycles), device=device)

        if return_intermediate:
            # Return order must match train.py expectations for TOPAS models:
            # (final_logits, inter_logits_list, halting_list, logic_states, ponder_cost, q_logits, ...)
            return (
                final_pixels,       # [0] final logits
                outputs,            # [1] intermediate outputs list
                halting_probs,      # [2] halting probabilities list
                logic_states_hist,  # [3] logic states for consistency loss
                ponder_cost,        # [4] ponder cost
                final_q,            # [5] q_logits (Q-values for halting) - REQUIRED for TOPAS
                obj_counts,         # [6] object count predictions
                centroids           # [7] centroid predictions
            )

        return final_pixels, final_q, None

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # --- Compatibility wrapper for existing train loop ---
    def forward_v3(self, *args, **kwargs) -> Union[Tuple[torch.Tensor, torch.Tensor, None], Tuple]:
        """Maps old call signature to new forward."""
        return self.forward(*args, **kwargs)


# =============================================================================
# Factory Function
# =============================================================================

def create_topas_model(size: str = 'small') -> TOPASDSPLModel:
    """
    Factory function for different model sizes.

    Args:
        size: One of 'tiny', 'small', 'base', 'large'

    Returns:
        Configured TOPASDSPLModel instance
    """
    configs = {
        # ~3.5M params
        'tiny': dict(
            d_model=256,
            n_heads=8,
            expansion=3.0,
            L_layers=1,
            H_cycles=2,
            L_cycles=3
        ),
        # ~15.6M params (default - good balance of speed/capacity)
        'small': dict(
            d_model=384,
            n_heads=8,
            expansion=4.0,
            L_layers=2,
            H_cycles=3,
            L_cycles=4
        ),
        # ~24M params
        'base': dict(
            d_model=480,
            n_heads=8,
            expansion=4.0,
            L_layers=2,
            H_cycles=3,
            L_cycles=4
        ),
        # ~43M params
        'large': dict(
            d_model=640,
            n_heads=8,
            expansion=4.0,
            L_layers=2,
            H_cycles=3,
            L_cycles=4
        ),
    }
    return TOPASDSPLModel(**configs[size])
