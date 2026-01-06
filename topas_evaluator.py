"""
TOPAS Evaluator: Test-Time Training + Augmentation for ARC-AGI

Features:
1. TTT (Test-Time Training): Optimizes program tokens on demos via Leave-One-Out CV
2. TTA (Test-Time Augmentation):
   - TRM-Style (num_aug > 100): Random sampling of Translation + Dihedral + Color
   - Systematic (num_aug <= 100): Full D8 dihedral + optional color permutations
3. Majority Voting: Robust ensemble prediction with top-k candidates (count-first)

Compatible with TOPASDSPLModel from topas_dslp_model.py
"""

import copy
import itertools
import json
import math
import os
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple, Any, Generator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

# Visualization imports (optional - graceful fallback if not available)
try:
    from visualization import (
        plot_grid, create_error_map, plot_error_map, ARC_CMAP, ERROR_CMAP
    )
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    VIZ_AVAILABLE = True
except ImportError:
    VIZ_AVAILABLE = False

# Constants
ARC_GRID_SIZE = 30
NUM_COLORS = 11  # 0-9 + PAD (index 10)
PAD_CLASS = 10

# Crop function - find bounding box of non-PAD content
def _crop_to_content(grid: np.ndarray) -> np.ndarray:
    """
    Crop grid to content region, assuming content starts at (0,0).

    This matches how training data is structured:
    - Content is placed at top-left (0,0)
    - Find first row/column that is ALL PAD to determine boundary

    For TOPAS: valid tokens are 0-9 (colors), PAD is 10.
    """
    grid = grid.reshape(30, 30)

    # Find content dimensions by scanning from top-left
    # Same approach as puzzle_dataset.py _seq_to_grid
    content_mask = (grid >= 0) & (grid <= 9)  # Non-PAD pixels

    # Find height: first row that has no content
    H = 30
    for h in range(30):
        if not content_mask[h, :].any():
            H = h
            break

    # Find width: first column that has no content
    W = 30
    for w in range(30):
        if not content_mask[:, w].any():
            W = w
            break

    if H == 0 or W == 0:
        # No valid content
        return np.array([[0]], dtype=np.uint8)

    return grid[:H, :W].astype(np.uint8)


class TOPASEvaluator:
    """
    Evaluator with Test-Time Training and Test-Time Augmentation.

    TTT: Before solving each puzzle, optimizes program tokens using
         Leave-One-Out Cross-Validation on demonstrations.

    TTA: Applies consistent augmentations to demos AND test input,
         then majority votes across all predictions.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        verbose: bool = False
    ):
        self.model = model
        self.device = device
        self.verbose = verbose
        self.model.eval()

        # Cache original program tokens
        self._original_program_tokens = model.program_token_init.data.clone()

    def solve_puzzle(
        self,
        puzzle_data: Dict,
        ttt_steps: int = 10,
        ttt_lr: float = 0.05,
        num_aug: int = 8,
        enable_ttt: bool = True,
        enable_color_perm: bool = False,
        num_color_perms: int = 10,
        top_k: int = 3
    ) -> List[np.ndarray]:
        """
        Solve a single puzzle with TTT + TTA.

        If num_aug > 100: Uses TRM-style random sampling (Translation + Dihedral + Color)
        If num_aug <= 100: Uses systematic D8 + color permutations

        Args:
            puzzle_data: dict with 'train' (list of {'input': grid, 'output': grid})
                        and 'test' (list of {'input': grid})
            ttt_steps: Number of gradient steps for TTT optimization
            ttt_lr: Learning rate for TTT
            num_aug: Number of augmentations (>100 triggers TRM random sampling)
            enable_ttt: Whether to run TTT before inference
            enable_color_perm: Whether to include color permutations in TTA
            num_color_perms: Number of random color permutations to try
            top_k: Return top-k candidates by vote count

        Returns:
            List of top-k predicted grids as numpy arrays (cropped)
        """
        # 1. Prepare Data
        train_pairs = puzzle_data['train']
        test_input = puzzle_data['test'][0]['input']

        # Convert to tensors
        demos_in, demos_out = self._pack_demos(train_pairs)
        test_in = self._pack_grid(test_input)

        # Get actual grid size (before padding)
        test_h, test_w = len(test_input), len(test_input[0])

        # 2. Test-Time Training
        if enable_ttt and ttt_steps > 0 and len(train_pairs) >= 2:
            if self.verbose:
                print(f"  > Running TTT for {ttt_steps} steps...")
            self._run_ttt(demos_in, demos_out, steps=ttt_steps, lr=ttt_lr)

        # 3. Test-Time Augmentation with TRM-style Majority Voting
        votes = {}  # hash -> {'count': int, 'conf_sum': float, 'grid': np.array}

        # Determine strategy: TRM random sampling vs systematic
        use_random_sampling = (num_aug > 100)

        if use_random_sampling:
            # TRM-Style: Random sampling of Translation + Dihedral + Color
            augmentations = self._generate_random_augmentations(num_aug, enable_color_perm)
        else:
            # Systematic: D8 + Color permutations
            augmentations = self._generate_systematic_augmentations(num_aug, enable_color_perm, num_color_perms)

        with torch.no_grad():
            for aug_data in augmentations:
                try:
                    if use_random_sampling:
                        # Random sampling mode: aug_data = (aug_fn, inv_aug_fn, color_map, use_translation)
                        aug_fn, inv_aug_fn, color_map, use_translation = aug_data

                        # Apply dihedral + color to demos
                        aug_demos_in = aug_fn(demos_in)
                        aug_demos_out = aug_fn(demos_out)
                        if color_map is not None:
                            aug_demos_in = self._apply_color_perm(aug_demos_in, color_map)
                            aug_demos_out = self._apply_color_perm(aug_demos_out, color_map)

                        # Apply dihedral + color to test input
                        aug_test_in = aug_fn(test_in.unsqueeze(0)).squeeze(0)
                        if color_map is not None:
                            aug_test_in = self._apply_color_perm(aug_test_in.unsqueeze(0), color_map).squeeze(0)

                        # Only apply translation if enabled (TRM-style TTA doesn't use translation)
                        if use_translation:
                            # Translate demos CONSISTENTLY (Input & Output move together)
                            aug_demos_in, aug_demos_out, _ = self._apply_consistent_translation(
                                aug_demos_in, aug_demos_out
                            )
                            # Translate test input
                            aug_test_batch, _, test_shifts = self._apply_consistent_translation(
                                aug_test_in.unsqueeze(0), None
                            )
                            aug_test_in = aug_test_batch.squeeze(0)
                    else:
                        # Systematic mode: aug_data = (aug_fn, inv_aug_fn, color_map)
                        aug_fn, inv_aug_fn, color_map = aug_data
                        test_shift = (0, 0)

                        # Apply consistent augmentation to demos AND test
                        aug_demos_in = aug_fn(demos_in)
                        aug_demos_out = aug_fn(demos_out)
                        aug_test_in = aug_fn(test_in.unsqueeze(0)).squeeze(0)

                        # Apply color permutation if provided
                        if color_map is not None:
                            aug_demos_in = self._apply_color_perm(aug_demos_in, color_map)
                            aug_demos_out = self._apply_color_perm(aug_demos_out, color_map)
                            aug_test_in = self._apply_color_perm(aug_test_in.unsqueeze(0), color_map).squeeze(0)

                    # Format for model: [1, n_demos, C, H, W]
                    batch_demos_in = aug_demos_in.unsqueeze(0).to(self.device)
                    batch_demos_out = aug_demos_out.unsqueeze(0).to(self.device)
                    batch_test_in = aug_test_in.unsqueeze(0).to(self.device)

                    # Demo mask (all valid)
                    demo_mask = torch.zeros(1, aug_demos_in.shape[0], dtype=torch.bool, device=self.device)

                    # Forward pass
                    logits, _, _ = self.model(
                        batch_demos_in,
                        batch_demos_out,
                        batch_test_in,
                        demo_mask=demo_mask
                    )

                    # Get prediction
                    pred_grid = logits.argmax(dim=1)  # [1, H, W]

                    # Calculate confidence score
                    probs = F.softmax(logits, dim=1)  # [1, C, H, W]
                    pred_probs = probs.gather(1, pred_grid.unsqueeze(1))  # [1, 1, H, W]
                    confidence = pred_probs.mean().item()

                    # Inverse transforms (reverse order)
                    # NOTE: We do NOT inverse-translate predictions.
                    # Translation changes absolute position but the model's output position
                    # depends on the TASK, not our artificial shift. Just crop and vote.

                    # 1. Inverse color permutation
                    if color_map is not None:
                        inv_color_map = {v: k for k, v in color_map.items()}
                        pred_grid = self._apply_color_perm_indices(pred_grid, inv_color_map)

                    # 3. Inverse spatial (dihedral) transform
                    pred_orig = inv_aug_fn(pred_grid.float()).long()

                    # Get full 30x30 grid (DO NOT CROP - match trainer approach)
                    pred_np = pred_orig[0].cpu().numpy().astype(int)
                    pred_np = np.clip(pred_np, 0, 10)

                    # Vote on FULL 30x30 grids (like trainer does)
                    # This is more robust than cropping because:
                    # 1. Different augmentations may produce slightly different positions
                    # 2. Masked comparison handles this correctly
                    grid_hash = self._hash_grid(pred_np)
                    if grid_hash not in votes:
                        votes[grid_hash] = {'count': 0, 'conf_sum': 0.0, 'grid': pred_np}
                    votes[grid_hash]['count'] += 1
                    votes[grid_hash]['conf_sum'] += confidence

                except Exception as e:
                    if self.verbose:
                        print(f"    ! Augmentation failed: {e}")
                    continue

        # 4. Restore original model state
        self.model.program_token_init.data = self._original_program_tokens.clone()

        # 5. Return top-k results (count-first sorting)
        if not votes:
            return [np.array([[0]], dtype=np.uint8)]

        candidates = sorted(
            votes.values(),
            key=lambda x: (x['count'], x['conf_sum'] / x['count'] if x['count'] > 0 else 0),
            reverse=True
        )

        if self.verbose:
            top = candidates[0]
            total_votes = sum(v['count'] for v in votes.values())
            mean_conf = top['conf_sum'] / top['count'] if top['count'] > 0 else 0
            print(f"  > Top vote: {top['count']}/{total_votes} votes, {mean_conf:.3f} mean conf ({len(votes)} unique)")

        # Crop ONLY when returning final results (for ARC submission format)
        # Voting was done on full 30x30 grids for robustness
        results = []
        for c in candidates[:top_k]:
            full_grid = c['grid']
            cropped = _crop_to_content(full_grid)
            if cropped.size == 0:
                cropped = np.array([[0]], dtype=np.uint8)
            results.append(cropped)

        return results

    def _run_ttt(
        self,
        demos_in: torch.Tensor,
        demos_out: torch.Tensor,
        steps: int = 10,
        lr: float = 0.05
    ):
        """
        Optimize program tokens using Leave-One-Out Cross-Validation.

        Uses N-1 demos as context to predict the Nth demo, cycling through all.
        """
        num_demos = demos_in.shape[0]
        if num_demos < 2:
            return

        # Create optimizable parameter
        param = nn.Parameter(self.model.program_token_init.data.clone())
        # SGD + Momentum for stability on few-shot TTT (Gemini recommendation)
        optimizer = optim.SGD([param], lr=lr, momentum=0.9)

        # Temporarily replace model's program tokens
        original_param = self.model.program_token_init
        self.model.program_token_init = param

        self.model.train()

        for step in range(steps):
            total_loss = 0.0
            optimizer.zero_grad()

            # Leave-One-Out loop
            for i in range(num_demos):
                # Target: demo i
                val_in = demos_in[i].unsqueeze(0).to(self.device)  # [1, C, H, W]
                val_out = demos_out[i].unsqueeze(0).to(self.device)  # [1, C, H, W]

                # Context: all other demos
                mask = torch.ones(num_demos, dtype=torch.bool)
                mask[i] = False
                ctx_in = demos_in[mask].unsqueeze(0).to(self.device)  # [1, N-1, C, H, W]
                ctx_out = demos_out[mask].unsqueeze(0).to(self.device)

                # Demo mask for context (all valid)
                demo_mask = torch.zeros(1, num_demos - 1, dtype=torch.bool, device=self.device)

                # Forward: use val_in as test input, predict val_out
                logits, _, _ = self.model(
                    ctx_in,
                    ctx_out,
                    val_in,
                    demo_mask=demo_mask
                )

                # Loss: cross entropy on valid pixels only
                # Get valid mask from target (non-PAD pixels)
                target_classes = val_out.argmax(dim=1)  # [1, H, W]
                valid_mask = (target_classes != PAD_CLASS)

                # Flatten for loss
                logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, NUM_COLORS)  # [H*W, C]
                target_flat = target_classes.reshape(-1)  # [H*W]
                valid_flat = valid_mask.reshape(-1)  # [H*W]

                if valid_flat.sum() > 0:
                    loss = F.cross_entropy(
                        logits_flat[valid_flat],
                        target_flat[valid_flat]
                    )
                    total_loss += loss

            # Update
            if total_loss > 0:
                total_loss.backward()
                optimizer.step()

        self.model.eval()
        # Note: param is still in model.program_token_init, will be restored in solve_puzzle

    def _generate_random_augmentations(
        self,
        n_samples: int,
        enable_color_perm: bool
    ) -> Generator:
        """
        TRM-Style: Generate n_samples of RANDOM (Dihedral + Color).
        NO translation at inference - TRM only uses translation for training data augmentation.

        Yields: (aug_fn, inv_aug_fn, color_map, use_translation=False)
        """
        for _ in range(n_samples):
            # 1. Random Dihedral (0-7) - TRM encoding
            k = random.randint(0, 7)
            aug_fn, inv_aug_fn = self._get_dihedral_fns(k)

            # 2. Random Color permutation - TRM style: keep 0 (background) fixed
            color_map = None
            if enable_color_perm:
                # Permute colors 1-9, keep 0 fixed (TRM style)
                perm = [0] + list(np.random.permutation(range(1, 10)))
                color_map = {i: perm[i] for i in range(10)}
                color_map[PAD_CLASS] = PAD_CLASS

            # NO translation at inference (False flag)
            yield aug_fn, inv_aug_fn, color_map, False

    def _generate_systematic_augmentations(
        self,
        num_dihedral: int = 8,
        enable_color_perm: bool = False,
        num_color_perms: int = 10
    ) -> List[Tuple]:
        """
        Systematic: Full D8 + optional color permutations.
        TRM-style: keep color 0 (background) fixed.

        Returns list of (aug_fn, inv_aug_fn, color_map) tuples.
        """
        augmentations = []

        # D8 Dihedral group (TRM encoding)
        for k in range(min(num_dihedral, 8)):
            aug_fn, inv_aug_fn = self._get_dihedral_fns(k)
            augmentations.append((aug_fn, inv_aug_fn, None))

        # Color permutations (optional) - TRM style: keep 0 fixed
        if enable_color_perm and num_color_perms > 0:
            base_augs = augmentations.copy()

            for _ in range(num_color_perms):
                # Permute colors 1-9, keep 0 fixed (TRM style)
                perm = [0] + list(np.random.permutation(range(1, 10)))
                color_map = {i: perm[i] for i in range(10)}
                color_map[PAD_CLASS] = PAD_CLASS

                for aug_fn, inv_aug_fn, _ in base_augs:
                    augmentations.append((aug_fn, inv_aug_fn, color_map))

        return augmentations

    # TRM D8 inverse mapping
    DIHEDRAL_INVERSE = [0, 3, 2, 1, 4, 5, 6, 7]

    def _get_dihedral_fns(self, k: int) -> Tuple:
        """
        Get aug/inv functions for D8 transform k (0-7).

        TRM encoding (must match training data augmentation):
        - 0: identity
        - 1: rot90
        - 2: rot180
        - 3: rot270
        - 4: flipLR (horizontal flip)
        - 5: flipUD (vertical flip)
        - 6: transpose
        - 7: flipLR(rot90)
        """
        def aug_fn(x):
            if x.dim() == 3:
                x = x.unsqueeze(0)
                squeeze = True
            else:
                squeeze = False

            # Apply TRM-style dihedral transform
            if k == 0:
                pass  # identity
            elif k == 1:
                x = torch.rot90(x, k=1, dims=[2, 3])
            elif k == 2:
                x = torch.rot90(x, k=2, dims=[2, 3])
            elif k == 3:
                x = torch.rot90(x, k=3, dims=[2, 3])
            elif k == 4:
                x = torch.flip(x, dims=[3])  # flipLR
            elif k == 5:
                x = torch.flip(x, dims=[2])  # flipUD
            elif k == 6:
                x = x.transpose(2, 3)  # transpose
            elif k == 7:
                x = torch.flip(torch.rot90(x, k=1, dims=[2, 3]), dims=[3])  # flipLR(rot90)

            return x.squeeze(0) if squeeze else x

        def inv_aug_fn(x):
            has_channel = x.dim() == 4

            if x.dim() == 2:
                x = x.unsqueeze(0)
                squeeze = True
            else:
                squeeze = False

            if has_channel:
                dims_hw = [2, 3]
                flip_lr = 3
                flip_ud = 2
            else:
                dims_hw = [1, 2]
                flip_lr = 2
                flip_ud = 1

            # Apply inverse transform using TRM's inverse mapping
            inv_k = self.DIHEDRAL_INVERSE[k]
            if inv_k == 0:
                pass  # identity
            elif inv_k == 1:
                x = torch.rot90(x, k=1, dims=dims_hw)
            elif inv_k == 2:
                x = torch.rot90(x, k=2, dims=dims_hw)
            elif inv_k == 3:
                x = torch.rot90(x, k=3, dims=dims_hw)
            elif inv_k == 4:
                x = torch.flip(x, dims=[flip_lr])  # flipLR
            elif inv_k == 5:
                x = torch.flip(x, dims=[flip_ud])  # flipUD
            elif inv_k == 6:
                x = x.transpose(dims_hw[0], dims_hw[1])  # transpose
            elif inv_k == 7:
                x = torch.flip(torch.rot90(x, k=1, dims=dims_hw), dims=[flip_lr])

            return x.squeeze(0) if squeeze else x

        return aug_fn, inv_aug_fn

    def _apply_consistent_translation(
        self,
        inputs: torch.Tensor,
        outputs: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[Tuple[int, int]]]:
        """
        TRUE translation: shift entire grids by (dr, dc), preserving spatial relationships.

        Unlike extract-reposition, this maintains relative positions between input/output.
        E.g., if output was 2 pixels right of input, it stays 2 pixels right after shift.
        """
        N, C, H, W = inputs.shape
        shifts = []

        # Find combined bounding box across ALL inputs and outputs
        # to compute a valid shift range
        for i in range(N):
            # Get input content bounds
            rows_i, cols_i = torch.where(inputs[i, PAD_CLASS, :, :] < 0.5)
            if len(rows_i) == 0:
                min_r_i, max_r_i, min_c_i, max_c_i = 0, 0, 0, 0
            else:
                min_r_i, max_r_i = rows_i.min().item(), rows_i.max().item()
                min_c_i, max_c_i = cols_i.min().item(), cols_i.max().item()

            # Get output content bounds (if exists)
            if outputs is not None:
                rows_o, cols_o = torch.where(outputs[i, PAD_CLASS, :, :] < 0.5)
                if len(rows_o) > 0:
                    min_r_o, max_r_o = rows_o.min().item(), rows_o.max().item()
                    min_c_o, max_c_o = cols_o.min().item(), cols_o.max().item()
                else:
                    min_r_o, max_r_o, min_c_o, max_c_o = min_r_i, max_r_i, min_c_i, max_c_i
            else:
                min_r_o, max_r_o, min_c_o, max_c_o = min_r_i, max_r_i, min_c_i, max_c_i

            # Combined bounding box (union of input and output)
            union_min_r = min(min_r_i, min_r_o)
            union_max_r = max(max_r_i, max_r_o)
            union_min_c = min(min_c_i, min_c_o)
            union_max_c = max(max_c_i, max_c_o)

            # Valid shift range: shift must keep union bbox within [0, H) x [0, W)
            # After shift by (dr, dc): new_min_r = union_min_r + dr, new_max_r = union_max_r + dr
            # Constraints: new_min_r >= 0, new_max_r < H
            # => dr >= -union_min_r, dr <= H - 1 - union_max_r
            dr_min = -union_min_r
            dr_max = H - 1 - union_max_r
            dc_min = -union_min_c
            dc_max = W - 1 - union_max_c

            # Sample random shift
            if dr_max >= dr_min:
                dr = random.randint(dr_min, dr_max)
            else:
                dr = 0
            if dc_max >= dc_min:
                dc = random.randint(dc_min, dc_max)
            else:
                dc = 0
            shifts.append((dr, dc))

        # Apply shifts using torch.roll (true translation with wrap, then mask)
        out_in = torch.zeros_like(inputs)
        out_in[:, PAD_CLASS, :, :] = 1.0
        out_out = None
        if outputs is not None:
            out_out = torch.zeros_like(outputs)
            out_out[:, PAD_CLASS, :, :] = 1.0

        for i, (dr, dc) in enumerate(shifts):
            if dr == 0 and dc == 0:
                out_in[i] = inputs[i]
                if outputs is not None:
                    out_out[i] = outputs[i]
                continue

            # Shift input: content at (r, c) moves to (r+dr, c+dc)
            # Use slicing to avoid wrap-around issues
            src_r = slice(max(0, -dr), min(H, H - dr))
            src_c = slice(max(0, -dc), min(W, W - dc))
            dst_r = slice(max(0, dr), min(H, H + dr))
            dst_c = slice(max(0, dc), min(W, W + dc))

            out_in[i, :, dst_r, dst_c] = inputs[i, :, src_r, src_c]

            if outputs is not None:
                out_out[i, :, dst_r, dst_c] = outputs[i, :, src_r, src_c]

        return out_in, out_out, shifts

    def _inverse_translation(
        self,
        pred: torch.Tensor,
        shift: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Inverse shift prediction by (-dr, -dc).

        pred: [1, 30, 30] class indices
        shift: (dr, dc) applied to input

        Logic: If input was shifted +dr, output is likely shifted +dr.
        We want output at origin (aligned), so: Sol[y,x] = Pred[y+dr, x+dc]
        """
        dr, dc = shift
        if dr == 0 and dc == 0:
            return pred

        H, W = 30, 30
        out = torch.full_like(pred, PAD_CLASS)

        # Safe slicing: copy from shifted region to origin
        src_r_start, src_r_end = dr, H
        src_c_start, src_c_end = dc, W

        dst_r_start, dst_r_end = 0, H - dr
        dst_c_start, dst_c_end = 0, W - dc

        if src_r_end > src_r_start and src_c_end > src_c_start:
            out[:, dst_r_start:dst_r_end, dst_c_start:dst_c_end] = \
                pred[:, src_r_start:src_r_end, src_c_start:src_c_end]

        return out

    def _apply_color_perm(self, x: torch.Tensor, color_map: Dict[int, int]) -> torch.Tensor:
        """
        Apply color permutation to one-hot encoded tensor.
        x: [..., C, H, W] where C is color channels
        """
        # Reorder channels according to color_map
        new_x = torch.zeros_like(x)
        for old_c, new_c in color_map.items():
            if old_c < x.shape[-3] and new_c < x.shape[-3]:
                new_x[..., new_c, :, :] = x[..., old_c, :, :]
        return new_x

    def _apply_color_perm_indices(self, x: torch.Tensor, color_map: Dict[int, int]) -> torch.Tensor:
        """
        Apply color permutation to class index tensor.
        x: [..., H, W] with integer class indices
        """
        new_x = x.clone()
        for old_c, new_c in color_map.items():
            new_x[x == old_c] = new_c
        return new_x

    def _pack_grid(self, grid: List[List[int]]) -> torch.Tensor:
        """
        Convert raw grid (list of lists) to one-hot tensor [C, H, W].
        Pads to ARC_GRID_SIZE x ARC_GRID_SIZE.
        """
        grid_np = np.array(grid, dtype=np.int64)
        H, W = grid_np.shape

        # Create one-hot tensor
        tensor = torch.zeros(NUM_COLORS, ARC_GRID_SIZE, ARC_GRID_SIZE)

        # Fill valid region
        for r in range(min(H, ARC_GRID_SIZE)):
            for c in range(min(W, ARC_GRID_SIZE)):
                val = grid_np[r, c]
                if 0 <= val < 10:  # Valid color
                    tensor[val, r, c] = 1.0
                else:
                    tensor[PAD_CLASS, r, c] = 1.0

        # Fill padding region with PAD class
        if H < ARC_GRID_SIZE:
            tensor[PAD_CLASS, H:, :] = 1.0
        if W < ARC_GRID_SIZE:
            tensor[PAD_CLASS, :, W:] = 1.0

        return tensor

    def _pack_demos(self, pairs: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pack list of demo pairs into tensors."""
        ins, outs = [], []
        for p in pairs:
            ins.append(self._pack_grid(p['input']))
            outs.append(self._pack_grid(p['output']))
        return torch.stack(ins), torch.stack(outs)

    def _hash_grid(self, grid: np.ndarray) -> tuple:
        """Create hashable representation of grid."""
        return tuple(grid.flatten().tolist())


def evaluate_arc_submission(
    model: nn.Module,
    challenges_path: str,
    solutions_path: Optional[str] = None,
    device: str = 'cuda',
    ttt_steps: int = 10,
    ttt_lr: float = 0.05,
    num_aug: int = 8,
    enable_ttt: bool = True,
    enable_color_perm: bool = False,
    num_color_perms: int = 10,
    verbose: bool = True,
    viz_path: Optional[str] = None,
    log_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate model on ARC challenge set with TTT + TTA.

    Args:
        model: TOPASDSPLModel instance
        challenges_path: Path to challenges JSON
        solutions_path: Optional path to solutions JSON for accuracy calculation
        device: Device to use
        ttt_steps, ttt_lr: TTT hyperparameters
        num_aug: Number of D8 augmentations
        enable_ttt: Whether to use TTT
        enable_color_perm: Whether to use color permutation TTA
        num_color_perms: Number of color permutations
        verbose: Print progress
        viz_path: Optional path to save PDF visualization
        log_path: Optional path to save text log with pre/post TTT comparison

    Returns:
        Dict with 'predictions' and optionally 'accuracy' metrics
    """
    # Load data
    with open(challenges_path, 'r') as f:
        challenges = json.load(f)

    solutions = None
    if solutions_path:
        with open(solutions_path, 'r') as f:
            solutions = json.load(f)

    # Create evaluator
    evaluator = TOPASEvaluator(model, device=device, verbose=verbose)

    # Run evaluation
    predictions = {}
    correct = 0
    correct_pre_ttt = 0
    total = 0

    # Visualization data collection
    viz_data = [] if viz_path and VIZ_AVAILABLE else None

    # Log file
    log_file = open(log_path, 'w') if log_path else None
    if log_file:
        log_file.write(f"TOPAS Evaluation Log\n")
        log_file.write(f"====================\n")
        log_file.write(f"Checkpoint: {challenges_path}\n")
        log_file.write(f"TTT Steps: {ttt_steps}, TTT LR: {ttt_lr}\n")
        log_file.write(f"Augmentations: {num_aug} D8, Color Perms: {enable_color_perm} ({num_color_perms})\n")
        log_file.write(f"\n{'='*80}\n\n")

    task_ids = list(challenges.keys())
    iterator = tqdm(task_ids, desc="Evaluating") if verbose else task_ids

    for task_id in iterator:
        puzzle = challenges[task_id]

        # Solve each test case
        task_preds = []
        for test_idx, test_case in enumerate(puzzle['test']):
            # Create puzzle data for this test case
            puzzle_data = {
                'train': puzzle['train'],
                'test': [test_case]
            }

            # === Pre-TTT prediction (baseline) ===
            preds_pre = evaluator.solve_puzzle(
                puzzle_data,
                ttt_steps=0,  # No TTT
                ttt_lr=ttt_lr,
                num_aug=num_aug,
                enable_ttt=False,
                enable_color_perm=enable_color_perm,
                num_color_perms=num_color_perms,
                top_k=1
            )

            # === Post-TTT prediction ===
            preds = evaluator.solve_puzzle(
                puzzle_data,
                ttt_steps=ttt_steps,
                ttt_lr=ttt_lr,
                num_aug=num_aug,
                enable_ttt=enable_ttt,
                enable_color_perm=enable_color_perm,
                num_color_perms=num_color_perms,
                top_k=3
            )

            # Predictions are already cropped from solve_puzzle (TRM-style)
            # Just convert to list format for submission
            cropped_preds = [pred.tolist() for pred in preds]
            task_preds.append(cropped_preds)

            # Check accuracy if solutions available
            if solutions and task_id in solutions:
                expected_raw = np.array(solutions[task_id][test_idx])
                eh, ew = expected_raw.shape

                # === Pre-TTT accuracy (predictions are already cropped) ===
                pred_pre = preds_pre[0]
                ph_pre, pw_pre = pred_pre.shape
                # Exact match: same shape AND same content
                is_correct_pre = (ph_pre == eh and pw_pre == ew and
                                  np.array_equal(pred_pre, expected_raw))
                # Pixel accuracy: compare overlapping region
                min_h_pre, min_w_pre = min(ph_pre, eh), min(pw_pre, ew)
                if min_h_pre > 0 and min_w_pre > 0:
                    overlap_correct_pre = (pred_pre[:min_h_pre, :min_w_pre] ==
                                           expected_raw[:min_h_pre, :min_w_pre]).sum()
                    pixel_acc_pre = overlap_correct_pre / expected_raw.size * 100
                else:
                    pixel_acc_pre = 0.0
                if is_correct_pre:
                    correct_pre_ttt += 1

                # === Post-TTT accuracy (predictions are already cropped) ===
                pred = preds[0]
                ph, pw = pred.shape
                # Exact match: same shape AND same content
                is_correct = (ph == eh and pw == ew and
                              np.array_equal(pred, expected_raw))
                # Pixel accuracy: compare overlapping region
                min_h, min_w = min(ph, eh), min(pw, ew)
                if min_h > 0 and min_w > 0:
                    overlap_correct = (pred[:min_h, :min_w] ==
                                       expected_raw[:min_h, :min_w]).sum()
                    pixel_acc = overlap_correct / expected_raw.size * 100
                else:
                    pixel_acc = 0.0

                if is_correct:
                    correct += 1
                total += 1

                # Delta
                pixel_delta = pixel_acc - pixel_acc_pre
                delta_str = f"+{pixel_delta:.1f}" if pixel_delta >= 0 else f"{pixel_delta:.1f}"

                # Dopamine logging
                if verbose:
                    status = "SOLVED" if is_correct else "Failed"
                    icon = "+" if is_correct else "-"
                    acc_pct = (correct / total * 100) if total > 0 else 0.0
                    pre_str = "PRE:SOLVED" if is_correct_pre else f"PRE:{pixel_acc_pre:.1f}%"
                    shape_match = "✓" if (ph == eh and pw == ew) else f"✗ pred={ph}x{pw} exp={eh}x{ew}"
                    print(f"  [{icon}] {task_id}[{test_idx}]: {status} | {shape_match} | ACC: {correct}/{total} ({acc_pct:.1f}%) | {pre_str} -> Pix: {pixel_acc:.1f}% ({delta_str})", flush=True)

                # Log file
                if log_file:
                    log_file.write(f"{task_id}[{test_idx}]\n")
                    log_file.write(f"  Pre-TTT:  {'SOLVED' if is_correct_pre else 'Failed'} | Pix: {pixel_acc_pre:.1f}%\n")
                    log_file.write(f"  Post-TTT: {'SOLVED' if is_correct else 'Failed'} | Pix: {pixel_acc:.1f}%\n")
                    log_file.write(f"  Delta: {delta_str}%\n")
                    log_file.write(f"\n")
                    log_file.flush()

                # Collect visualization data (use raw expected, not padded)
                if viz_data is not None:
                    viz_data.append({
                        'task_id': task_id,
                        'test_idx': test_idx,
                        'demos_in': [np.array(p['input']) for p in puzzle['train']],
                        'demos_out': [np.array(p['output']) for p in puzzle['train']],
                        'test_input': np.array(test_case['input']),
                        'prediction': pred,
                        'target': expected_raw,
                        'is_correct': is_correct
                    })

        predictions[task_id] = task_preds

    # Results
    results = {'predictions': predictions}

    if solutions:
        accuracy = correct / total if total > 0 else 0.0
        accuracy_pre = correct_pre_ttt / total if total > 0 else 0.0
        results['accuracy'] = accuracy
        results['accuracy_pre_ttt'] = accuracy_pre
        results['correct'] = correct
        results['correct_pre_ttt'] = correct_pre_ttt
        results['total'] = total

        summary = f"\n{'='*60}\n"
        summary += f"FINAL RESULTS\n"
        summary += f"{'='*60}\n"
        summary += f"Pre-TTT:  {correct_pre_ttt}/{total} = {accuracy_pre*100:.2f}%\n"
        summary += f"Post-TTT: {correct}/{total} = {accuracy*100:.2f}%\n"
        summary += f"TTT Delta: {correct - correct_pre_ttt:+d} solves ({(accuracy - accuracy_pre)*100:+.2f}%)\n"
        summary += f"{'='*60}\n"

        if verbose:
            print(summary)

        if log_file:
            log_file.write(summary)
            log_file.close()
            if verbose:
                print(f"Saved log to: {log_path}")

    # Generate PDF visualization
    if viz_data and viz_path:
        _generate_eval_pdf(viz_data, viz_path, correct, total, verbose)

    return results


def _generate_eval_pdf(viz_data: List[Dict], output_path: str, correct: int, total: int, verbose: bool = True):
    """
    Generate a PDF visualization of evaluation results.

    Args:
        viz_data: List of dicts with task visualization data
        output_path: Path to save PDF
        correct: Number of correct predictions
        total: Total predictions
        verbose: Print status messages
    """
    if not VIZ_AVAILABLE:
        if verbose:
            print("Warning: Visualization not available (matplotlib not installed)")
        return

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # Sort: correct tasks first, then by task_id
    viz_data.sort(key=lambda x: (-int(x['is_correct']), x['task_id']))

    tasks_per_page = 5
    n_pages = math.ceil(len(viz_data) / tasks_per_page)

    if verbose:
        print(f"\nGenerating visualization PDF: {output_path}")

    with PdfPages(output_path) as pdf:
        for page in range(n_pages):
            start_idx = page * tasks_per_page
            end_idx = min(start_idx + tasks_per_page, len(viz_data))
            page_data = viz_data[start_idx:end_idx]

            # 6 columns: Demo1 In, Demo1 Out, Test In, Prediction, Target, Error Map
            fig, axes = plt.subplots(len(page_data), 6, figsize=(16, 3 * len(page_data)))
            if len(page_data) == 1:
                axes = axes.reshape(1, -1)

            for row, data in enumerate(page_data):
                pred = data['prediction']
                target = data['target']
                test_input = data['test_input']

                # Demo 1 (first demo pair)
                if len(data['demos_in']) > 0:
                    plot_grid(axes[row, 0], data['demos_in'][0], "Demo In")
                    plot_grid(axes[row, 1], data['demos_out'][0], "Demo Out")
                else:
                    axes[row, 0].axis('off')
                    axes[row, 1].axis('off')

                # Test input
                plot_grid(axes[row, 2], test_input, "Test Input")

                # Prediction with status (show full prediction)
                status = "CORRECT" if data['is_correct'] else "WRONG"
                color = 'green' if data['is_correct'] else 'red'
                plot_grid(axes[row, 3], pred, f"Pred ({status})")
                axes[row, 3].title.set_color(color)

                # Target (show full target)
                plot_grid(axes[row, 4], target, "Target")

                # Error map - crop both to overlapping region for comparison
                ph, pw = pred.shape
                th, tw = target.shape
                min_h, min_w = min(ph, th), min(pw, tw)
                pred_for_error = pred[:min_h, :min_w]
                target_for_error = target[:min_h, :min_w]
                error_map = create_error_map(pred_for_error, target_for_error, pad_class=10)
                plot_error_map(axes[row, 5], error_map, "Error Map")

                # Add task ID as row label
                axes[row, 0].set_ylabel(f"{data['task_id']}[{data['test_idx']}]",
                                        fontsize=8, rotation=0, ha='right', va='center')

            # Page title
            page_correct = sum(1 for d in page_data if d['is_correct'])
            fig.suptitle(
                f"TOPAS Evaluation Results | Page {page + 1}/{n_pages} | "
                f"Tasks {start_idx + 1}-{end_idx} | {page_correct}/{len(page_data)} correct on page",
                fontsize=12
            )

            plt.tight_layout(rect=[0.02, 0, 1, 0.97])
            pdf.savefig(fig, dpi=100)
            plt.close(fig)

        # Summary page
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Accuracy pie chart
        axes[0, 0].pie([correct, total - correct],
                       labels=['Correct', 'Incorrect'],
                       colors=['#2ECC40', '#FF4136'],
                       autopct='%1.1f%%',
                       startangle=90)
        axes[0, 0].set_title(f'Exact Match: {correct}/{total} ({correct/total*100:.1f}%)')

        # Correct vs Incorrect bar
        axes[0, 1].barh(['Correct', 'Incorrect'], [correct, total - correct],
                        color=['#2ECC40', '#FF4136'])
        axes[0, 1].set_xlabel('Count')
        axes[0, 1].set_title('Task Results')

        # Error type breakdown (aggregate)
        total_wrong_color = 0
        total_missed = 0
        total_false_pos = 0
        for data in viz_data:
            pred = data['prediction']
            target = data['target']
            # Crop both to overlapping region
            ph, pw = pred.shape
            th, tw = target.shape
            min_h, min_w = min(ph, th), min(pw, tw)
            pred_cropped = pred[:min_h, :min_w]
            target_cropped = target[:min_h, :min_w]
            error_map = create_error_map(pred_cropped, target_cropped, pad_class=10)
            total_wrong_color += (error_map == 1).sum()
            total_missed += (error_map == 2).sum()
            total_false_pos += (error_map == 3).sum()

        axes[1, 0].bar(['Wrong Color', 'Missed', 'False Pos'],
                       [total_wrong_color, total_missed, total_false_pos],
                       color=['#FF4136', '#0074D9', '#FFDC00'])
        axes[1, 0].set_ylabel('Pixel Count')
        axes[1, 0].set_title('Error Type Breakdown (All Tasks)')

        # Legend
        axes[1, 1].axis('off')
        legend_text = (
            "Error Map Legend:\n"
            "─────────────────\n"
            "Black  = Correct\n"
            "Red    = Wrong Color\n"
            "Blue   = Missed (should be colored)\n"
            "Yellow = False Positive (extra color)\n"
            "Gray   = Padding\n"
            "\n"
            f"Total Tasks: {total}\n"
            f"Correct: {correct} ({correct/total*100:.1f}%)\n"
            f"Wrong Color Pixels: {total_wrong_color}\n"
            f"Missed Pixels: {total_missed}\n"
            f"False Positive Pixels: {total_false_pos}"
        )
        axes[1, 1].text(0.1, 0.5, legend_text, transform=axes[1, 1].transAxes,
                        fontsize=12, verticalalignment='center',
                        family='monospace',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        fig.suptitle('Evaluation Summary', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, dpi=100)
        plt.close(fig)

    if verbose:
        print(f"Saved visualization to: {output_path}")


def save_submission(predictions: Dict, output_path: str):
    """Save predictions in ARC submission format."""
    # Format: {"task_id": [[attempt1, attempt2, ...], ...], ...}
    submission = {}
    for task_id, test_preds in predictions.items():
        submission[task_id] = test_preds

    with open(output_path, 'w') as f:
        json.dump(submission, f)

    print(f"Saved submission to {output_path}")


if __name__ == "__main__":
    import argparse
    from topas_dslp_model import TOPASDSPLModel

    parser = argparse.ArgumentParser(description="TOPAS ARC Evaluator")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--challenges", type=str, required=True, help="Challenges JSON path")
    parser.add_argument("--solutions", type=str, default=None, help="Solutions JSON path")
    parser.add_argument("--output", type=str, default="submission.json", help="Output path")
    parser.add_argument("--ttt-steps", type=int, default=10, help="TTT optimization steps")
    parser.add_argument("--ttt-lr", type=float, default=0.01, help="TTT learning rate")
    parser.add_argument("--num-aug", type=int, default=8, help="Number of D8 augmentations")
    parser.add_argument("--no-ttt", action="store_true", help="Disable TTT")
    parser.add_argument("--no-color-perm", action="store_true", help="Disable color permutation TTA (enabled by default)")
    parser.add_argument("--num-color-perms", type=int, default=10, help="Number of color permutations")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--save-viz", type=str, default=None, help="Path to save PDF visualization")
    parser.add_argument("--save-log", type=str, default=None, help="Path to save text log with pre/post TTT comparison")

    args = parser.parse_args()

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)

    # Extract config from checkpoint
    config = checkpoint.get('config', {})
    model_cfg = config.get('model', {})

    model = TOPASDSPLModel(
        d_model=model_cfg.get('d_model', 480),
        n_heads=model_cfg.get('n_heads', 8),
        expansion=model_cfg.get('expansion', 4.0),
        dropout=0.0,  # No dropout at eval
        H_cycles=model_cfg.get('H_cycles', 3),
        L_cycles=model_cfg.get('L_cycles', 4),
        L_layers=model_cfg.get('L_layers', 2),
        img_size=model_cfg.get('img_size', 30),
        num_colors=model_cfg.get('num_colors', 11),
        max_demos=model_cfg.get('max_demos', 3),
        puzzle_emb_ndim=model_cfg.get('puzzle_emb_ndim', 0),
        num_tasks=model_cfg.get('num_tasks', 100000),
        halt_max_steps=model_cfg.get('halt_max_steps', 16),
        halt_exploration_prob=0.0  # No exploration at eval
    ).to(args.device)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Run evaluation
    results = evaluate_arc_submission(
        model=model,
        challenges_path=args.challenges,
        solutions_path=args.solutions,
        device=args.device,
        ttt_steps=args.ttt_steps,
        ttt_lr=args.ttt_lr,
        num_aug=args.num_aug,
        enable_ttt=not args.no_ttt,
        enable_color_perm=not args.no_color_perm,
        num_color_perms=args.num_color_perms,
        verbose=True,
        viz_path=args.save_viz,
        log_path=args.save_log
    )

    # Save submission
    save_submission(results['predictions'], args.output)
