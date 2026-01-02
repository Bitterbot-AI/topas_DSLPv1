"""
TOPAS Evaluator: Test-Time Training + Augmentation for ARC-AGI

Features:
1. TTT (Test-Time Training): Optimizes program tokens on demos via Leave-One-Out CV
2. TTA (Test-Time Augmentation): Full D8 dihedral + optional color permutations
3. Majority Voting: Robust ensemble prediction with top-k candidates

Compatible with TOPASDSPLModel from topas_dslp_model.py
"""

import copy
import itertools
import json
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

# Constants
ARC_GRID_SIZE = 30
NUM_COLORS = 11  # 0-9 + PAD (index 10)
PAD_CLASS = 10


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

        Args:
            puzzle_data: dict with 'train' (list of {'input': grid, 'output': grid})
                        and 'test' (list of {'input': grid})
            ttt_steps: Number of gradient steps for TTT optimization
            ttt_lr: Learning rate for TTT
            num_aug: Number of D8 augmentations (max 8)
            enable_ttt: Whether to run TTT before inference
            enable_color_perm: Whether to include color permutations in TTA
            num_color_perms: Number of random color permutations to try
            top_k: Return top-k candidates by vote count

        Returns:
            List of top-k predicted grids as numpy arrays
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

        # 3. Test-Time Augmentation
        votes = Counter()
        grid_cache = {}  # hash -> grid

        # Generate augmentations
        augmentations = self._generate_augmentations(num_aug, enable_color_perm, num_color_perms)

        with torch.no_grad():
            for aug_fn, inv_aug_fn, color_map in augmentations:
                try:
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

                    # Calculate confidence score (mean probability of predicted pixels)
                    probs = F.softmax(logits, dim=1)  # [1, C, H, W]
                    pred_probs = probs.gather(1, pred_grid.unsqueeze(1))  # [1, 1, H, W]
                    confidence = pred_probs[:, :, :test_h, :test_w].mean().item()

                    # Inverse color permutation
                    if color_map is not None:
                        inv_color_map = {v: k for k, v in color_map.items()}
                        pred_grid = self._apply_color_perm_indices(pred_grid, inv_color_map)

                    # Inverse spatial transform
                    pred_orig = inv_aug_fn(pred_grid.float()).long()

                    # Crop to original size and convert to numpy
                    pred_np = pred_orig[0, :test_h, :test_w].cpu().numpy().astype(int)

                    # Handle any remaining PAD tokens (shouldn't be in valid area)
                    pred_np = np.clip(pred_np, 0, 9)

                    # Vote (confidence-weighted)
                    grid_hash = self._hash_grid(pred_np)
                    votes[grid_hash] += confidence
                    grid_cache[grid_hash] = pred_np

                except Exception as e:
                    if self.verbose:
                        print(f"    ! Augmentation failed: {e}")
                    continue

        # 4. Restore original model state
        self.model.program_token_init.data = self._original_program_tokens.clone()

        # 5. Return top-k results
        if not votes:
            # Fallback: return empty grid
            return [np.zeros((test_h, test_w), dtype=int)]

        top_hashes = [h for h, _ in votes.most_common(top_k)]
        results = [grid_cache[h] for h in top_hashes]

        if self.verbose:
            top_conf = votes.most_common(1)[0][1]
            total_conf = sum(votes.values())
            print(f"  > Top vote: {top_conf:.3f}/{total_conf:.3f} confidence ({len(votes)} unique predictions)")

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
        optimizer = optim.Adam([param], lr=lr)

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

    def _generate_augmentations(
        self,
        num_dihedral: int = 8,
        enable_color_perm: bool = False,
        num_color_perms: int = 10
    ) -> List[Tuple]:
        """
        Generate augmentation functions.

        Returns list of (aug_fn, inv_aug_fn, color_map) tuples.
        aug_fn/inv_aug_fn work on [N, C, H, W] or [C, H, W] tensors.
        """
        augmentations = []

        # D8 Dihedral group: 4 rotations x 2 (with/without flip)
        for k in range(min(num_dihedral, 8)):
            rot = k % 4  # 0, 1, 2, 3 = 0, 90, 180, 270 degrees
            flip = k >= 4  # Whether to flip horizontally

            def make_aug(rot, flip):
                def aug_fn(x):
                    # x: [N, C, H, W] or [C, H, W]
                    if x.dim() == 3:
                        x = x.unsqueeze(0)
                        squeeze = True
                    else:
                        squeeze = False

                    if flip:
                        x = torch.flip(x, dims=[3])  # Horizontal flip
                    if rot > 0:
                        x = torch.rot90(x, k=rot, dims=[2, 3])

                    return x.squeeze(0) if squeeze else x

                def inv_aug_fn(x):
                    # x: [N, H, W] (class indices) or [N, C, H, W]
                    has_channel = x.dim() == 4

                    if x.dim() == 2:
                        x = x.unsqueeze(0)
                        squeeze = True
                    else:
                        squeeze = False

                    if has_channel:
                        dims_hw = [2, 3]
                        flip_dim = 3
                    else:
                        dims_hw = [1, 2]
                        flip_dim = 2

                    # Inverse: undo rotation first, then flip
                    if rot > 0:
                        x = torch.rot90(x, k=-rot, dims=dims_hw)
                    if flip:
                        x = torch.flip(x, dims=[flip_dim])

                    return x.squeeze(0) if squeeze else x

                return aug_fn, inv_aug_fn

            aug_fn, inv_aug_fn = make_aug(rot, flip)
            augmentations.append((aug_fn, inv_aug_fn, None))

        # Color permutations (optional)
        if enable_color_perm and num_color_perms > 0:
            base_augs = augmentations.copy()

            for _ in range(num_color_perms):
                # Random permutation of colors 0-9 (not PAD)
                perm = list(range(10))
                random.shuffle(perm)
                color_map = {i: perm[i] for i in range(10)}
                color_map[PAD_CLASS] = PAD_CLASS  # Keep PAD unchanged

                # Combine with each dihedral augmentation
                for aug_fn, inv_aug_fn, _ in base_augs:
                    augmentations.append((aug_fn, inv_aug_fn, color_map))

        return augmentations

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
    verbose: bool = True
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
    total = 0

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

            # Get predictions (top-3)
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

            # Convert to list format for submission
            task_preds.append([pred.tolist() for pred in preds])

            # Check accuracy if solutions available
            if solutions and task_id in solutions:
                expected = np.array(solutions[task_id][test_idx])
                if np.array_equal(preds[0], expected):
                    correct += 1
                total += 1

        predictions[task_id] = task_preds

    # Results
    results = {'predictions': predictions}

    if solutions:
        accuracy = correct / total if total > 0 else 0.0
        results['accuracy'] = accuracy
        results['correct'] = correct
        results['total'] = total
        if verbose:
            print(f"\nAccuracy: {correct}/{total} = {accuracy*100:.2f}%")

    return results


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
    parser.add_argument("--ttt-lr", type=float, default=0.05, help="TTT learning rate")
    parser.add_argument("--num-aug", type=int, default=8, help="Number of D8 augmentations")
    parser.add_argument("--no-ttt", action="store_true", help="Disable TTT")
    parser.add_argument("--color-perm", action="store_true", help="Enable color permutation TTA")
    parser.add_argument("--num-color-perms", type=int, default=10, help="Number of color permutations")
    parser.add_argument("--device", type=str, default="cuda", help="Device")

    args = parser.parse_args()

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

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
        enable_color_perm=args.color_perm,
        num_color_perms=args.num_color_perms,
        verbose=True
    )

    # Save submission
    save_submission(results['predictions'], args.output)
