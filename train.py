import os
import math
import json
import yaml
import random
import glob
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.data import Dataset, DataLoader, DistributedSampler, Sampler
from torch import optim
from topas_dslp_model import TOPASDSPLModel
from logger import DSPLLogger
from ema import EMAHelper
from puzzle_dataset import PuzzleDataset

# TPU support (optional - only imported if --tpu flag is used)
_TPU_AVAILABLE = False
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    _TPU_AVAILABLE = True
except ImportError:
    pass

# Import optimizers from parent directory
import sys
_v3_code_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_v3_code_dir)
if _v3_code_dir not in sys.path:
    sys.path.insert(0, _v3_code_dir)
if _parent_dir not in sys.path:
    sys.path.append(_parent_dir)  # append, not insert, so v3_code takes priority

# AdamAtan2 from pip package (commented out - not used, AdamW+MuonClip is used instead)
# from adam_atan2_pytorch import AdamAtan2
# Import MuonClip from parent directory (use append to not override v3_code priority)
_parent_for_muon = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_for_muon not in sys.path:
    sys.path.append(_parent_for_muon)
from muonclip import MuonClip, MuonClipConfig


def cosine_schedule_with_warmup(
    current_step: int,
    base_lr: float,
    num_warmup_steps: int,
    num_training_steps: int,
    min_ratio: float = 0.01
) -> float:
    """
    Compute learning rate with linear warmup and cosine decay.
    Cosine schedule with linear warmup.

    Args:
        current_step: Current training step (0-indexed)
        base_lr: Base learning rate
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        min_ratio: Minimum LR ratio at end of training (default 0.01 = 1% of base_lr)

    Returns:
        Learning rate for this step
    """
    if current_step < num_warmup_steps:
        # Linear warmup
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    # Cosine decay
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))))


class EpochSampler(Sampler):
    """Sampler that yields a fixed number of random samples per epoch."""

    def __init__(self, data_source, samples_per_epoch: int, seed: int = None):
        self.data_source = data_source
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch if self.seed else None)
        indices = list(range(len(self.data_source)))

        # Sample with replacement if needed
        if self.samples_per_epoch <= len(indices):
            sampled = rng.sample(indices, self.samples_per_epoch)
        else:
            sampled = rng.choices(indices, k=self.samples_per_epoch)

        return iter(sampled)

    def __len__(self):
        return self.samples_per_epoch

    def set_epoch(self, epoch: int):
        self.epoch = epoch


class SequenceDataset(Dataset):
    """
    Dataset that returns flattened token sequences directly.

    For use with TOPAS model - no conversion to one-hot grids.

    Returns:
        inputs: [900] input token sequence (PAD=0, EOS=1, colors 2-11)
        labels: [900] label token sequence
        puzzle_id: task identifier for puzzle embedding
    """

    def __init__(self, data_dir, split="train", use_original_only=False):
        """
        Args:
            data_dir: Path to augmented dataset directory
            split: "train" or "test"
            use_original_only: If True, always use original (for eval)
        """
        self.split = split
        self.use_original_only = use_original_only

        split_dir = os.path.join(data_dir, split)

        # Load numpy arrays with memory mapping for large arrays
        self.inputs = np.load(os.path.join(split_dir, "all__inputs.npy"), mmap_mode='r')  # [N, 900]
        self.labels = np.load(os.path.join(split_dir, "all__labels.npy"), mmap_mode='r')  # [N, 900]
        # Small index arrays kept in RAM
        self.puzzle_ids = np.load(os.path.join(split_dir, "all__puzzle_identifiers.npy"))
        self.puzzle_indices = np.load(os.path.join(split_dir, "all__puzzle_indices.npy"))
        self.group_indices = np.load(os.path.join(split_dir, "all__group_indices.npy"))

        self.num_groups = len(self.group_indices) - 1
        self.num_puzzles = len(self.puzzle_indices) - 1

        # Sample individual examples for training
        # Each puzzle has multiple examples, we pick the test example (last one)
        self.total_examples = len(self.inputs)

    def __len__(self):
        # Return number of groups (for sampling augmentations)
        return self.num_groups

    def __getitem__(self, idx):
        """Get a puzzle's test example as token sequences."""
        # Get puzzle range for this group
        puzz_start = self.group_indices[idx]
        puzz_end = self.group_indices[idx + 1]

        # For eval: use original puzzle (first in group)
        # For train: sample random augmentation
        if self.use_original_only:
            puzz_idx = puzz_start
        else:
            puzz_idx = np.random.randint(puzz_start, puzz_end)

        # Get example range for this puzzle
        ex_start = self.puzzle_indices[puzz_idx]
        ex_end = self.puzzle_indices[puzz_idx + 1]

        # Task ID for puzzle embedding
        task_id = self.puzzle_ids[puzz_idx]

        # Use test example (last one in puzzle)
        test_idx = ex_end - 1

        inputs = torch.from_numpy(self.inputs[test_idx].astype(np.int64))
        labels = torch.from_numpy(self.labels[test_idx].astype(np.int64))

        return {
            'inputs': inputs,           # [900] token sequence
            'labels': labels,           # [900] token sequence
            'puzzle_id': torch.tensor(task_id, dtype=torch.long),
        }


class ARCEvalDataset(Dataset):
    """
    Dataset for ARC evaluation from JSON files.

    Returns full task in V3 grid format (one-hot tensors with demos).
    """

    def __init__(self, challenges_path, solutions_path, img_size=30, max_demos=3, num_colors=11):
        """
        Args:
            challenges_path: Path to challenges JSON file
            solutions_path: Path to solutions JSON file
            img_size: Grid size (30x30)
            max_demos: Maximum number of demo examples
            num_colors: Number of color classes (10 colors + PAD)
        """
        self.img_size = img_size
        self.max_demos = max_demos
        self.num_colors = num_colors

        with open(challenges_path, 'r') as f:
            self.challenges = json.load(f)
        with open(solutions_path, 'r') as f:
            self.solutions = json.load(f)

        self.task_ids = list(self.challenges.keys())

    def __len__(self):
        return len(self.task_ids)

    def _grid_to_onehot(self, grid):
        """Convert 2D grid to one-hot tensor [C, H, W]."""
        h, w = len(grid), len(grid[0]) if grid else 0
        # Initialize with PAD (last channel)
        onehot = torch.zeros(self.num_colors, self.img_size, self.img_size)
        onehot[self.num_colors - 1, :, :] = 1.0  # PAD channel

        for i in range(min(h, self.img_size)):
            for j in range(min(w, self.img_size)):
                color = grid[i][j]
                onehot[self.num_colors - 1, i, j] = 0.0  # Clear PAD
                onehot[color, i, j] = 1.0  # Set color

        return onehot

    def __getitem__(self, idx):
        """Get a full task in V3 format (demos + test)."""
        task_id = self.task_ids[idx]
        task = self.challenges[task_id]
        solution = self.solutions[task_id]

        # Process demos (train examples)
        train_examples = task['train']
        n_demos = min(len(train_examples), self.max_demos)

        train_in = torch.zeros(self.max_demos, self.num_colors, self.img_size, self.img_size)
        train_out = torch.zeros(self.max_demos, self.num_colors, self.img_size, self.img_size)
        demo_mask = torch.ones(self.max_demos, dtype=torch.bool)  # True = masked/invalid

        for i in range(n_demos):
            train_in[i] = self._grid_to_onehot(train_examples[i]['input'])
            train_out[i] = self._grid_to_onehot(train_examples[i]['output'])
            demo_mask[i] = False  # Valid demo

        # Process test example
        test_input = task['test'][0]['input']
        test_output = solution[0]

        test_in = self._grid_to_onehot(test_input)
        target_out = self._grid_to_onehot(test_output)

        task_id_tensor = torch.tensor(idx, dtype=torch.long)

        # Return as tuple (V3 format): 6-tuple
        return (train_in, train_out, test_in, target_out, demo_mask, task_id_tensor)


def compute_object_mask(grid_np, pad_class=10):
    """
    Compute binary object mask from a grid.

    Args:
        grid_np: [H, W] numpy array of class indices
        pad_class: PAD class index (default 10)

    Returns:
        object_mask: [H, W] binary mask (1.0 for objects, 0.0 for background/PAD)
    """
    # Objects are non-background (0) and non-PAD pixels
    return ((grid_np != 0) & (grid_np != pad_class)).astype(np.float32)


def compute_object_info(grid_np, pad_class=10):
    """
    Compute object count and centroid from a grid.

    Args:
        grid_np: [H, W] numpy array of class indices
        pad_class: PAD class index (default 10)

    Returns:
        object_count: int (clamped to 0-20)
        centroid: [2] numpy array of (y, x) normalized to [0, 1]
    """
    from scipy import ndimage

    H, W = grid_np.shape
    non_bg_mask = (grid_np != 0) & (grid_np != pad_class)

    if non_bg_mask.sum() == 0:
        return 0, np.array([0.5, 0.5], dtype=np.float32)  # Default center

    # Count objects
    labeled, num_objects = ndimage.label(non_bg_mask)
    object_count = min(num_objects, 20)

    # Compute centroid (center of mass of all non-bg pixels)
    coords = np.where(non_bg_mask)
    cy = coords[0].mean() / H  # Normalize to [0, 1]
    cx = coords[1].mean() / W
    centroid = np.array([cy, cx], dtype=np.float32)

    return object_count, centroid


def count_objects_in_grid(grid_onehot):
    """
    Count distinct connected components (objects) in a grid.

    Args:
        grid_onehot: [B, num_colors, H, W] one-hot encoded grid

    Returns:
        object_counts: [B] tensor with object count per sample (clamped to 0-20)
    """
    from scipy import ndimage

    B = grid_onehot.shape[0]
    PAD_CLASS = 10
    grid_classes = grid_onehot.argmax(dim=1).cpu().numpy()  # [B, H, W]

    counts = []
    for b in range(B):
        grid = grid_classes[b]
        # Create binary mask of non-background, non-PAD pixels
        # Background is typically color 0, PAD is color 10
        non_bg_mask = (grid != 0) & (grid != PAD_CLASS)

        if non_bg_mask.sum() == 0:
            counts.append(0)
        else:
            # Label connected components
            labeled, num_objects = ndimage.label(non_bg_mask)
            counts.append(min(num_objects, 20))  # Clamp to max 20

    return torch.tensor(counts, dtype=torch.long, device=grid_onehot.device)


def compute_batch_object_info(grid_onehot):
    """
    Compute object counts and centroids for a batch.

    Args:
        grid_onehot: [B, num_colors, H, W] one-hot encoded grid

    Returns:
        object_counts: [B] tensor with object count per sample
        centroids: [B, 2] tensor with (y, x) centroids normalized to [0, 1]
    """
    B = grid_onehot.shape[0]
    grid_classes = grid_onehot.argmax(dim=1).cpu().numpy()  # [B, H, W]

    counts = []
    centroids = []
    for b in range(B):
        count, centroid = compute_object_info(grid_classes[b])
        counts.append(count)
        centroids.append(centroid)

    return (
        torch.tensor(counts, dtype=torch.long, device=grid_onehot.device),
        torch.tensor(np.array(centroids), dtype=torch.float32, device=grid_onehot.device)
    )


def compute_loss(final_logits, inter_logits_list, halting_list, target_out,
                 logic_states, ponder_cost, w_primary, w_deep, w_logic, w_halt, w_comp,
                 content_mask=None, test_in=None, change_weight=10.0,
                 object_count_logits=None, target_object_counts=None, w_object=0.0,
                 centroid_pred=None, target_centroids=None, w_centroid=0.0,
                 q_logits=None, w_q_halt=0.1):
    """
    Compute loss with "Keep-Alive" penalty, Backloaded Supervision, Weighted Change Loss,
    Object Count Loss, and Centroid Loss.

    Args:
        final_logits: [B, num_colors, H, W] - num_colors=11 (10 colors + PAD)
        inter_logits_list: list of intermediate logits
        halting_list: list of halting probabilities per step
        target_out: [B, num_colors, H, W] target grid (PAD class=10 in padding areas)
        logic_states: list of program states [B, d_model] per step (for temporal consistency)
        test_in: [B, num_colors, H, W] test input (for weighted change loss)
        change_weight: multiplier for pixels that need to change (default 10.0)
        ponder_cost: scalar, average number of steps used (from ACT)
        w_*: loss weights
        content_mask: [B, H, W] boolean mask (unused now - PAD class handles it)
        centroid_pred: [B, 2] predicted object centroids (y, x) normalized
        target_centroids: [B, 2] target object centroids (y, x) normalized
        w_centroid: weight for centroid loss
    """
    PAD_CLASS = 10  # PAD is a separate class
    NUM_COLORS = 10  # Actual ARC colors (0-9)
    B, num_colors, H, W = final_logits.shape
    device = final_logits.device

    # Get target class indices - will be 0-9 for colors, 10 for PAD
    target_classes = target_out.argmax(dim=1)  # [B, H, W]
    target_flat = target_classes.view(B, -1)  # [B, H*W]

    # Class weights: Balance PAD importance vs numerical stability
    class_weights = torch.ones(num_colors, device=device)
    class_weights[PAD_CLASS] = 0.3  # Compromise: learns geometry without dominating loss

    # === Weighted Change Loss ===
    # Pixels that need to change (target != input) get higher weight
    if test_in is not None and change_weight > 1.0:
        input_classes = test_in.argmax(dim=1)  # [B, H, W]

        # Create change mask: where target differs from input (and not PAD)
        change_mask = (target_classes != input_classes) & (target_classes != PAD_CLASS)
        change_mask_flat = change_mask.view(B, -1).float()  # [B, H*W]

        # Weight map: static=1.0, changed=change_weight
        pixel_weights = 1.0 + change_mask_flat * (change_weight - 1.0)  # [B, H*W]

        # Per-pixel cross entropy (no reduction)
        logits_flat = final_logits.view(B, num_colors, -1).transpose(1, 2).reshape(-1, num_colors)
        pixel_losses = F.cross_entropy(logits_flat, target_flat.view(-1),
                                       weight=class_weights, reduction='none')  # [B*H*W]
        pixel_losses = pixel_losses.view(B, -1)  # [B, H*W]

        # Apply per-pixel weights and average
        weighted_losses = pixel_losses * pixel_weights
        loss_primary = weighted_losses.mean()

        # Track change ratio for logging
        change_ratio = change_mask.float().sum() / (change_mask.numel() - (target_classes == PAD_CLASS).sum())
    else:
        # Standard cross-entropy (no change weighting)
        loss_primary = F.cross_entropy(
            final_logits.view(B, num_colors, -1).transpose(1, 2).reshape(-1, num_colors),
            target_flat.view(-1),
            weight=class_weights
        )
        change_ratio = torch.tensor(0.0)

    # === FIX #4: Backloaded Deep Supervision ===
    # Reduce weight for early steps (t=0) to let the model "draft" freely.
    # Use quadratic schedule (t/T)^2 instead of linear.
    loss_deep = 0.0
    T = len(inter_logits_list)
    if T > 1:
        for t, logits in enumerate(inter_logits_list[:-1]):  # Exclude final
            # Quadratic weighting: penalize step 1 very little, step 30 heavily
            weight = ((t + 1) / (T - 1)) ** 2
            step_loss = F.cross_entropy(
                logits.view(B, num_colors, -1).transpose(1, 2).reshape(-1, num_colors),
                target_classes.view(-1),
                weight=class_weights
            )
            loss_deep += weight * step_loss
        loss_deep = loss_deep / (T - 1)

    # === FIX #1: Logic Consistency with "Keep-Alive" Penalty ===
    loss_logic = 0.0
    loss_stagnation = 0.0

    if logic_states is not None and len(logic_states) > 1:
        temporal_diffs = []
        for t in range(1, len(logic_states)):
            # 1. Standard Consistency: Don't oscillate wildly
            # We treat the *previous* state as the target to smooth trajectory
            diff = F.mse_loss(logic_states[t], logic_states[t-1].detach())

            # 2. "Keep-Alive" Penalty: Force movement!
            # Calculate Euclidean distance of update
            delta_z = torch.norm(logic_states[t] - logic_states[t-1], p=2, dim=-1).mean()
            # If update is smaller than 0.05, penalize it heavily.
            # This forces the logic core to "do work" every step.
            stagnation_penalty = torch.relu(0.05 - delta_z)
            loss_stagnation += stagnation_penalty

            weight = t / len(logic_states)
            temporal_diffs.append(weight * diff)

        loss_logic = sum(temporal_diffs) / len(temporal_diffs)
        loss_logic += loss_stagnation * 2.0  # High weight on stagnation penalty

    # Halting loss (unchanged)
    loss_halt = 0.0
    T_halt = len(halting_list)
    if T_halt > 0:
        H_mat = torch.stack(halting_list, dim=0)
        target = torch.zeros_like(H_mat)
        for t in range(T_halt):
            target[t, :] = min(1.0, (t + 1) / (T_halt * 0.5))
        target[-1, :] = 1.0
        loss_halt = F.mse_loss(H_mat, target)

    # Complexity penalty (unchanged)
    loss_comp = ponder_cost / T if T > 0 else ponder_cost

    # Object count loss - forces model to learn object-centric representations
    loss_object = torch.tensor(0.0, device=device)
    if object_count_logits is not None and target_object_counts is not None and w_object > 0:
        loss_object = F.cross_entropy(object_count_logits, target_object_counts)

    # Centroid loss - forces model to learn spatial object positions
    loss_centroid = torch.tensor(0.0, device=device)
    if centroid_pred is not None and target_centroids is not None and w_centroid > 0:
        # Use smooth L1 loss (Huber) for robustness to outliers
        loss_centroid = F.smooth_l1_loss(centroid_pred, target_centroids)

    # Q-learning halting loss
    # Teaches model to predict q_halt > 0 when sequence is correct
    loss_q_halt = torch.tensor(0.0, device=device)
    if q_logits is not None and w_q_halt > 0:
        # q_logits: [B, 2] where [:, 0] is q_halt, [:, 1] is q_continue
        q_halt = q_logits[:, 0]  # [B]

        # Compute per-sequence correctness (all content pixels match)
        pred_classes = final_logits.argmax(dim=1)  # [B, H, W]
        content_mask_qhalt = (target_classes != PAD_CLASS)  # [B, H, W]

        # Check if each sequence is fully correct (only content pixels)
        seq_correct = ((pred_classes == target_classes) | ~content_mask_qhalt).all(dim=-1).all(dim=-1)  # [B]
        seq_correct_float = seq_correct.float()

        # BCE loss: q_halt should be positive when correct, negative when wrong
        loss_q_halt = F.binary_cross_entropy_with_logits(q_halt, seq_correct_float)

    # Total loss
    total_loss = (
        w_primary * loss_primary +
        w_deep * loss_deep +
        w_logic * loss_logic +
        w_halt * loss_halt +
        w_comp * loss_comp +
        w_object * loss_object +
        w_centroid * loss_centroid +
        w_q_halt * loss_q_halt
    )

    return total_loss, {
        'primary': loss_primary.item(),
        'deep': loss_deep.item() if isinstance(loss_deep, torch.Tensor) else loss_deep,
        'logic': loss_logic.item() if isinstance(loss_logic, torch.Tensor) else loss_logic,
        'halt': loss_halt.item() if isinstance(loss_halt, torch.Tensor) else loss_halt,
        'comp': loss_comp.item() if isinstance(loss_comp, torch.Tensor) else loss_comp,
        'object': loss_object.item() if isinstance(loss_object, torch.Tensor) else loss_object,
        'centroid': loss_centroid.item() if isinstance(loss_centroid, torch.Tensor) else loss_centroid,
        'q_halt': loss_q_halt.item() if isinstance(loss_q_halt, torch.Tensor) else loss_q_halt,
        'change_pct': (change_ratio.item() if isinstance(change_ratio, torch.Tensor) else change_ratio) * 100
    }


def compute_iou(pred, true, num_classes=10):
    """Compute per-class IoU and mean IoU."""
    ious = []
    for c in range(num_classes):
        pred_c = (pred == c)
        true_c = (true == c)
        intersection = (pred_c & true_c).sum().item()
        union = (pred_c | true_c).sum().item()
        if union > 0:
            ious.append(intersection / union)
    return ious


def evaluate(model, eval_data, device, num_classes=11, verbose=True, task_id_offset=0, min_steps=0):
    """Evaluate model on the evaluation set with granular metrics.

    Args:
        model: The model to evaluate
        eval_data: Iterable dataset for evaluation (PuzzleDataset or DataLoader)
        device: Device to run on
        num_classes: Number of classes (11 = 10 colors + PAD)
        min_steps: Minimum steps to force during evaluation
        verbose: Whether to print progress
        task_id_offset: Offset for task IDs (for eval set = num_train_tasks)
    """
    PAD_CLASS = 10  # PAD class
    NUM_COLORS = 10  # Actual color classes (0-9)
    PAD_TOKEN = 0  # PAD token for sequence format

    model.eval()
    solved = 0
    total = 0
    total_steps_used = 0

    # Granular metrics
    total_pixels = 0
    correct_pixels = 0
    total_nonzero_pixels = 0
    correct_nonzero_pixels = 0
    task_pixel_accs = []
    task_nonbg_accs = []  # Per-task non-background accuracy
    task_ious = []
    partial_matches = {'>90%': 0, '>80%': 0, '>70%': 0, '>50%': 0}

    # Per-class metrics (only 10 colors, not PAD)
    class_correct = [0] * NUM_COLORS
    class_total = [0] * NUM_COLORS

    with torch.no_grad():
        for task_idx, batch in enumerate(eval_data):
            # NOTE: Dict-style sequence batches are disabled - TOPAS-DSPL uses v3 grids
            # If dict batch encountered, skip it (shouldn't happen with current config)
            if isinstance(batch, dict):
                continue

            # V3 tuple-style batch (one-hot grids)
            # Handle different tuple formats:
            # 8-tuple: with object mask (newest)
            # 7-tuple: with content mask
            # 6-tuple: with task_id
            # 5-tuple: legacy
            if len(batch) == 8:
                train_in, train_out, test_in, target_out, demo_mask, task_ids, content_mask, test_in_obj_mask = batch
                task_ids = task_ids.to(device)
                content_mask = content_mask.to(device)
                test_in_obj_mask = test_in_obj_mask.to(device)
            elif len(batch) == 7:
                train_in, train_out, test_in, target_out, demo_mask, task_ids, content_mask = batch
                task_ids = task_ids.to(device)
                content_mask = content_mask.to(device)
                test_in_obj_mask = None
            elif len(batch) == 6:
                train_in, train_out, test_in, target_out, demo_mask, task_ids = batch
                task_ids = task_ids.to(device)
                content_mask = None
                test_in_obj_mask = None
            else:
                train_in, train_out, test_in, target_out, demo_mask = batch
                task_ids = None
                content_mask = None
                test_in_obj_mask = None

            train_in = train_in.to(device)
            train_out = train_out.to(device)
            test_in = test_in.to(device)
            target_out = target_out.to(device)
            demo_mask = demo_mask.to(device)

            # Add batch dimension if missing (single sample from Dataset)
            if train_in.dim() == 4:  # [n_demos, C, H, W] -> [1, n_demos, C, H, W]
                train_in = train_in.unsqueeze(0)
                train_out = train_out.unsqueeze(0)
                test_in = test_in.unsqueeze(0)
                target_out = target_out.unsqueeze(0)
                demo_mask = demo_mask.unsqueeze(0)
                if task_ids is not None and task_ids.dim() == 0:
                    task_ids = task_ids.unsqueeze(0)

            # Unpack all return values (object_count_logits and centroid_pred are unused in eval)
            result = model(
                train_in, train_out, test_in, demo_mask,
                task_ids=task_ids,
                return_intermediate=True,
                min_steps=min_steps,
                object_mask=test_in_obj_mask
            )
            final_logits, inter_list, halt_list = result[0], result[1], result[2]

            # Determine predicted output grid
            pred = final_logits.argmax(dim=1)  # [B, H, W]
            true = target_out.argmax(dim=1)    # [B, H, W]

            # Create eval mask from PAD class - evaluate only non-PAD pixels
            # This is cleaner than using content_mask since target already has PAD=10 for padding
            eval_mask = (true != PAD_CLASS)  # True for actual content, False for padding

            # Exact match (on content pixels only)
            content_matches = (pred == true) | ~eval_mask  # Padding pixels always "match"
            is_solved = content_matches.all().item()
            if is_solved:
                solved += 1
            total += 1

            # Pixel-wise accuracy for this task (content only)
            content_total = eval_mask.sum().item()
            content_correct = ((pred == true) & eval_mask).sum().item()
            task_acc = content_correct / content_total if content_total > 0 else 1.0
            task_pixel_accs.append(task_acc)

            total_pixels += content_total
            correct_pixels += content_correct

            # Per-class accuracy (content only, skip PAD class)
            for c in range(NUM_COLORS):  # Only 0-9, not PAD
                mask_c = (true == c) & eval_mask
                if mask_c.sum() > 0:
                    class_total[c] += mask_c.sum().item()
                    class_correct[c] += ((pred == c) & mask_c).sum().item()

            # IoU for this task (using masked regions)
            # Mask out PAD pixels before computing IoU
            pred_masked = pred.clone()
            true_masked = true.clone()
            pred_masked[~eval_mask] = -1  # Invalid class (padding)
            true_masked[~eval_mask] = -1
            task_iou_list = compute_iou(pred_masked, true_masked, NUM_COLORS)
            if task_iou_list:
                task_ious.append(np.mean(task_iou_list))

            # Non-zero (non-background) pixel accuracy (content only)
            nonzero_mask = (true != 0) & eval_mask
            if nonzero_mask.sum() > 0:
                nonzero_correct = ((pred == true) & nonzero_mask).sum().item()
                nonzero_total = nonzero_mask.sum().item()
                total_nonzero_pixels += nonzero_total
                correct_nonzero_pixels += nonzero_correct
                # Per-task non-BG accuracy
                task_nonbg_accs.append(nonzero_correct / nonzero_total)
            else:
                # No non-background pixels in target - use 1.0 if prediction also has none
                pred_nonzero = ((pred != 0) & eval_mask).sum().item()
                task_nonbg_accs.append(1.0 if pred_nonzero == 0 else 0.0)

            # Partial match thresholds
            if task_acc > 0.9:
                partial_matches['>90%'] += 1
            if task_acc > 0.8:
                partial_matches['>80%'] += 1
            if task_acc > 0.7:
                partial_matches['>70%'] += 1
            if task_acc > 0.5:
                partial_matches['>50%'] += 1

            # Determine halting step used (actual steps computed, not when halt_prob >= 0.5)
            T = len(halt_list)
            total_steps_used += T  # Actual steps run

    accuracy = solved / total if total > 0 else 0
    avg_steps = total_steps_used / total if total > 0 else 0
    pixel_acc = correct_pixels / total_pixels if total_pixels > 0 else 0
    nonzero_acc = correct_nonzero_pixels / total_nonzero_pixels if total_nonzero_pixels > 0 else 0
    mean_iou = np.mean(task_ious) if task_ious else 0

    # Per-class accuracy (only 10 colors, not PAD)
    per_class_acc = []
    for c in range(NUM_COLORS):
        if class_total[c] > 0:
            per_class_acc.append(class_correct[c] / class_total[c])
        else:
            per_class_acc.append(None)

    return {
        "accuracy": accuracy,
        "avg_steps": avg_steps,
        "solved": solved,
        "total": total,
        "pixel_acc": pixel_acc,
        "nonzero_acc": nonzero_acc,
        "mean_iou": mean_iou,
        "per_class_acc": per_class_acc,
        "partial_matches": partial_matches,
        "task_pixel_accs": task_pixel_accs,
        "task_nonbg_accs": task_nonbg_accs
    }


def evaluate_with_ttt(model, eval_loader, device, ttt_config, num_classes=11, verbose=True):
    """
    Evaluate model with Test-Time Training + Dihedral Voting.

    This is the "nuclear option" - runs TTT optimization on each task's demos
    before inference, with 8x geometric voting for robustness.

    Args:
        model: The model to evaluate
        eval_loader: DataLoader for evaluation data
        device: Device to run on
        ttt_config: Dict with TTT settings:
            'ttt_steps': int (default 50)
            'ttt_lr': float (default 0.01)
            'aug_voting': bool (default True)
        num_classes: Number of classes (11 = 10 colors + PAD)
        verbose: Whether to print progress
    """
    from ttt_engine import TTTEngine

    PAD_CLASS = 10
    NUM_COLORS = 10

    # Initialize TTT Engine
    engine = TTTEngine(model, ttt_config)

    solved = 0
    total = 0
    total_pixels = 0
    correct_pixels = 0
    total_nonzero_pixels = 0
    correct_nonzero_pixels = 0
    task_pixel_accs = []
    task_nonbg_accs = []
    partial_matches = {'>90%': 0, '>80%': 0, '>70%': 0, '>50%': 0}

    # Per-class metrics
    class_correct = [0] * NUM_COLORS
    class_total = [0] * NUM_COLORS

    print(f"Running TTT evaluation: {ttt_config['ttt_steps']} steps, LR={ttt_config['ttt_lr']}, voting={ttt_config.get('aug_voting', True)}")

    for task_idx, batch in enumerate(eval_loader):
        # Unpack batch
        if len(batch) == 7:
            train_in, train_out, test_in, target_out, demo_mask, task_ids, content_mask = batch
        elif len(batch) == 6:
            train_in, train_out, test_in, target_out, demo_mask, task_ids = batch
            content_mask = None
        else:
            train_in, train_out, test_in, target_out, demo_mask = batch
            task_ids = None
            content_mask = None

        # Move to device
        train_in = train_in.to(device)
        train_out = train_out.to(device)
        test_in = test_in.to(device)
        target_out = target_out.to(device)
        demo_mask = demo_mask.to(device)

        # Pack sample for TTT engine
        task_sample = (train_in, train_out, test_in, demo_mask)

        # Run TTT + Voting
        pred = engine.solve(task_sample)  # [B, H, W]

        # Ground truth
        true = target_out.argmax(dim=1)  # [B, H, W]

        # Eval mask (non-PAD pixels)
        eval_mask = (true != PAD_CLASS)

        # Exact match
        content_matches = (pred == true) | ~eval_mask
        is_solved = content_matches.all().item()
        if is_solved:
            solved += 1
        total += 1

        # Pixel accuracy
        content_total = eval_mask.sum().item()
        content_correct = ((pred == true) & eval_mask).sum().item()
        task_acc = content_correct / content_total if content_total > 0 else 1.0
        task_pixel_accs.append(task_acc)

        total_pixels += content_total
        correct_pixels += content_correct

        # Non-background accuracy
        nonbg_mask = (true != 0) & eval_mask
        nonbg_total = nonbg_mask.sum().item()
        nonbg_correct = ((pred == true) & nonbg_mask).sum().item()
        task_nonbg_acc = nonbg_correct / nonbg_total if nonbg_total > 0 else 1.0
        task_nonbg_accs.append(task_nonbg_acc)

        total_nonzero_pixels += nonbg_total
        correct_nonzero_pixels += nonbg_correct

        # Per-class accuracy
        for c in range(NUM_COLORS):
            mask_c = (true == c) & eval_mask
            if mask_c.sum() > 0:
                class_total[c] += mask_c.sum().item()
                class_correct[c] += ((pred == c) & mask_c).sum().item()

        # Partial matches
        if task_acc >= 0.9:
            partial_matches['>90%'] += 1
        if task_acc >= 0.8:
            partial_matches['>80%'] += 1
        if task_acc >= 0.7:
            partial_matches['>70%'] += 1
        if task_acc >= 0.5:
            partial_matches['>50%'] += 1

        if verbose and (task_idx + 1) % 10 == 0:
            print(f"  TTT Eval: {task_idx+1}/{len(eval_loader)} - Solved: {solved}/{total} ({100*solved/total:.1f}%)")

    # Compute final metrics
    accuracy = solved / total if total > 0 else 0
    pixel_acc = correct_pixels / total_pixels if total_pixels > 0 else 0
    nonzero_acc = correct_nonzero_pixels / total_nonzero_pixels if total_nonzero_pixels > 0 else 0

    per_class_acc = []
    for c in range(NUM_COLORS):
        if class_total[c] > 0:
            per_class_acc.append(class_correct[c] / class_total[c])
        else:
            per_class_acc.append(None)

    return {
        "accuracy": accuracy,
        "solved": solved,
        "total": total,
        "pixel_acc": pixel_acc,
        "nonzero_acc": nonzero_acc,
        "per_class_acc": per_class_acc,
        "partial_matches": partial_matches,
        "task_pixel_accs": task_pixel_accs,
        "task_nonbg_accs": task_nonbg_accs
    }


def train(config_path="config.yaml", resume_checkpoint=None, use_tpu=False, tpu_cores=8):
    """Main training function.

    Args:
        config_path: Path to config file (default: config.yaml)
        resume_checkpoint: Path to checkpoint file to resume from (overrides config)
        use_tpu: Whether to use TPU instead of GPU
        tpu_cores: Number of TPU cores to use (default: 8 for v5e-8)
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Check for resume in config if not provided via CLI
    if resume_checkpoint is None:
        resume_checkpoint = config.get("training", {}).get("resume", None)

    model_cfg = config["model"]
    train_cfg = config["training"]
    reg_cfg = config.get("regularization", {})
    log_cfg = config.get("logging", {})

    # Device selection: TPU > CUDA > CPU
    if use_tpu:
        if not _TPU_AVAILABLE:
            raise RuntimeError("TPU requested but torch_xla not available. Install: pip install torch_xla")
        device = xm.xla_device()
        print(f"TPU device: {device}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize logger
    logger = DSPLLogger(
        output_dir=train_cfg.get("output_dir", "./output"),
        experiment_name=log_cfg.get("experiment_name", None),
        use_tensorboard=log_cfg.get("use_tensorboard", True)
    )
    logger.info(f"Using device: {device}")
    logger.log_config(config)

    # Distributed setup (TPU or GPU)
    if use_tpu:
        world_size = xm.xrt_world_size()
        rank = xm.get_ordinal()
        logger.info(f"TPU distributed: rank {rank}/{world_size}")
    else:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        if world_size > 1:
            torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)

    # Build dataset and dataloader - TOPAS mode with IterableDataset (like TRM)
    train_dir = train_cfg.get("train_data_dir", train_cfg.get("augmented_data_dir"))
    eval_dir = train_cfg.get("eval_data_dir", "")
    epochs_per_iter = train_cfg.get("epochs_per_iter", 1)
    batch_size = train_cfg["batch_size"]

    logger.info(f"TOPAS IterableDataset loading (epochs_per_iter={epochs_per_iter})")
    logger.info(f"Train dataset: {train_dir}")

    # PuzzleDataset with v3 output mode (one-hot grids) for dual-stream architecture
    train_data = PuzzleDataset(
        dataset_paths=[train_dir],
        split="train",
        seed=train_cfg.get("seed", 42),
        global_batch_size=batch_size,
        epochs_per_iter=epochs_per_iter,
        rank=rank,
        num_replicas=world_size,
        output_mode="v3",
        test_set_mode=False
    )
    num_train_tasks = train_data.metadata.get("total_puzzles", 1000)
    num_train_groups = train_data.metadata.get("total_groups", 1000)

    # Eval dataset - use ARC eval JSON if available, otherwise fall back to train data
    eval_challenges = train_cfg.get("eval_challenges", "")
    eval_solutions = train_cfg.get("eval_solutions", "")

    if eval_challenges and eval_solutions and os.path.exists(eval_challenges) and os.path.exists(eval_solutions):
        eval_data = ARCEvalDataset(eval_challenges, eval_solutions)
        logger.info(f"Eval: {len(eval_data)} tasks from ARC eval JSON")
    else:
        # Fall back to train data for eval (not ideal but works)
        eval_data = PuzzleDataset(
            dataset_paths=[train_dir],
            split="train",
            seed=train_cfg.get("seed", 42),
            global_batch_size=1,
            epochs_per_iter=1,
            rank=0,
            num_replicas=1,
            output_mode="v3",
            test_set_mode=True
        )
        logger.info(f"Eval: {num_train_groups} puzzles from train data (no eval JSON found)")

    train_cfg["_topas_mode"] = True
    logger.info(f"Train: {num_train_groups} groups, {num_train_tasks} puzzles")

    # TPU: Wrap with ParallelLoader for async prefetching
    if use_tpu:
        # Create a simple DataLoader wrapper for ParallelLoader compatibility
        train_loader = DataLoader(train_data, batch_size=None, num_workers=0)
        train_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)
        logger.info("TPU: Using ParallelLoader for async data prefetching")
    else:
        train_loader = train_data  # Direct iteration for GPU

    # Calculate total number of tasks for puzzle embeddings
    # IterableDataset doesn't have len(), use metadata instead
    if hasattr(eval_data, 'metadata'):
        num_eval_tasks = eval_data.metadata.get("num_puzzle_identifiers", 0)
    else:
        num_eval_tasks = len(eval_data)
    num_total_tasks = model_cfg.get("num_tasks", num_train_tasks + num_eval_tasks)
    puzzle_emb_ndim = model_cfg.get("puzzle_emb_ndim", 0)
    batch_size = train_cfg["batch_size"]

    # TOPAS model with recursive architecture
    model = TOPASDSPLModel(
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        expansion=model_cfg.get("expansion", 4.0),
        dropout=model_cfg.get("dropout", 0.1),
        H_cycles=model_cfg.get("H_cycles", 3),
        L_cycles=model_cfg.get("L_cycles", 4),
        L_layers=model_cfg.get("L_layers", 2),
        puzzle_emb_ndim=puzzle_emb_ndim,
        num_tasks=num_total_tasks,
        halt_max_steps=model_cfg.get("halt_max_steps", 16),
        halt_exploration_prob=model_cfg.get("halt_exploration_prob", 0.1),
    ).to(device)
    logger.info(f"Using TOPAS model: d_model={model_cfg['d_model']}, H={model_cfg.get('H_cycles', 3)}, L={model_cfg.get('L_cycles', 4)}, L_layers={model_cfg.get('L_layers', 2)}")

    logger.log_model_info(model)
    logger.info(f"Puzzle embedding: ndim={puzzle_emb_ndim}, num_tasks={num_total_tasks}")

    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index], output_device=device.index
        )

    # Set up optimizers
    lr = train_cfg.get("learning_rate", 1e-4)
    base_lr = lr  # For LR scheduler
    wd = reg_cfg.get("weight_decay", 1e-4)
    freeze_weights = reg_cfg.get("freeze_weights", False)

    # LR schedule settings (per-step)
    lr_warmup_steps = reg_cfg.get("lr_warmup_steps", 100)
    lr_min_ratio = reg_cfg.get("lr_min_ratio", 0.01)

    # Get model reference (handle DDP wrapper)
    raw_model = model.module if hasattr(model, 'module') else model

    # Calculate total training steps for LR schedule (accounting for gradient accumulation)
    # With IterableDataset: batches_per_iter = (num_groups * epochs_per_iter) / batch_size
    accumulation_steps = train_cfg.get("accumulation_steps", 1)
    batches_per_iter = (num_train_groups * epochs_per_iter) // batch_size
    steps_per_iter = batches_per_iter // accumulation_steps
    num_iters = train_cfg["epochs"]  # Each "epoch" in config is one iteration
    total_training_steps = steps_per_iter * num_iters
    logger.info(f"Total training steps: {total_training_steps} ({steps_per_iter} steps/iter x {num_iters} iters)")
    logger.info(f"Gradient accumulation: {accumulation_steps} steps")
    logger.info(f"LR warmup: {lr_warmup_steps} steps, min_ratio: {lr_min_ratio}")

    # Optimizer setup
    optimizers = []
    optimizer_lrs = []

    if freeze_weights:
        raise ValueError("freeze_weights=True is not supported (puzzle embeddings disabled)")
    else:
        # AdamW + MuonClip for main model (proven stable)
        base_optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        main_optimizer = MuonClip(base_optimizer, model, MuonClipConfig())
        optimizers.append(main_optimizer)
        optimizer_lrs.append(lr)
        logger.info(f"Using AdamW + MuonClip for main model: base_lr={lr}, wd={wd}")

    # Primary optimizer reference (for checkpoint - use main optimizer, not SignSGD)
    # MuonClip is added last, so use the last optimizer
    optimizer = optimizers[-1] if optimizers else None

    # Resume from checkpoint if provided
    start_epoch = 1
    start_step = 0  # Global step counter for LR scheduling
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        logger.info(f"Resuming from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        model_state = checkpoint['model_state_dict']
        if hasattr(model, 'module'):
            model.module.load_state_dict(model_state, strict=False)
        else:
            model.load_state_dict(model_state, strict=False)
        # Skip optimizer state - incompatible between v1 (AdamW) and v3 (AdamAtan2)
        # Start fresh with new optimizer but keep model weights
        # Note: strict=False allows new heads (e.g., object_count_head) to be randomly initialized
        logger.info("Skipping optimizer state load (fresh optimizer for fine-tuning)")
        start_epoch = checkpoint['epoch'] + 1
        start_step = 0  # Reset step counter - fresh optimizer means fresh LR schedule
        logger.info(f"Loaded model from epoch {checkpoint['epoch']}, restarting training from epoch {start_epoch}")

    # Optional: Freeze canvas core for fine-tuning stability
    freeze_canvas = reg_cfg.get("freeze_canvas_core", False)
    if freeze_canvas:
        logger.info("Freezing Canvas Core (only Logic Core will be trained)")
        base_model = model.module if hasattr(model, 'module') else model
        # Freeze encoder and decoder (canvas processing)
        for param in base_model.encoder.parameters():
            param.requires_grad = False
        for param in base_model.decoder.parameters():
            param.requires_grad = False
        # Keep transformer (logic core) trainable
        trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in base_model.parameters() if not p.requires_grad)
        logger.info(f"Trainable params: {trainable_params:,}, Frozen params: {frozen_params:,}")

    # EMA of model weights
    use_ema = reg_cfg.get("ema", False)
    ema_decay = reg_cfg.get("ema_decay", 0.999)
    ema_model = None
    ema = None

    if use_ema:
        # Initialize EMA helper
        ema = EMAHelper(mu=ema_decay)
        ema.register(model)

        # Also create a separate EMA model for evaluation
        ema_model = TOPASDSPLModel(
            d_model=model_cfg["d_model"],
            n_heads=model_cfg["n_heads"],
            expansion=model_cfg.get("expansion", 4.0),
            dropout=0.0,  # No dropout for EMA model
            H_cycles=model_cfg.get("H_cycles", 3),
            L_cycles=model_cfg.get("L_cycles", 4),
            L_layers=model_cfg.get("L_layers", 2),
            puzzle_emb_ndim=puzzle_emb_ndim,
            num_tasks=num_total_tasks,
            halt_max_steps=model_cfg.get("halt_max_steps", 16),
            halt_exploration_prob=0.0,  # No exploration for EMA model
        ).to(device)
        # Copy state from main model (includes puzzle embeddings)
        ema_model.load_state_dict(
            model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        )
        for p in ema_model.parameters():
            p.requires_grad_(False)
        logger.info(f"EMA enabled: decay={ema_decay}")

    # Automatic Mixed Precision
    # Note: TPU uses bfloat16 natively, no scaler needed
    if use_tpu:
        use_amp = False  # TPU handles precision internally
        scaler = None
        logger.info("TPU mode: Using native bfloat16 precision")
    else:
        use_amp = train_cfg.get("amp", True) and torch.cuda.is_available()
        scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # Loss weights
    lw = config.get("loss_weights", {})
    w_primary = lw.get("primary", 1.0)
    w_deep = lw.get("deep_supervision", 1.0)
    w_logic = lw.get("logic_consistency", 0.1)
    w_halt = lw.get("halting", 0.1)
    w_comp = lw.get("complexity", 0.01)
    change_weight = lw.get("change_weight", 1.0)  # Weight multiplier for changed pixels (1.0 = disabled)
    w_object = lw.get("object_count", 0.5)  # Weight for auxiliary object count loss
    w_centroid = lw.get("centroid", 0.5)  # Weight for auxiliary centroid prediction loss
    w_q_halt = lw.get("q_halt", 0.1)  # Weight for Q-learning halting loss (TOPAS)

    num_epochs = train_cfg["epochs"]
    log_interval = train_cfg.get("log_interval", 50)
    vis_interval = log_cfg.get("vis_interval", 5)  # Visualize every N epochs

    # Gradient accumulation
    accumulation_steps = train_cfg.get("accumulation_steps", 1)
    effective_batch_size = train_cfg["batch_size"] * accumulation_steps * world_size

    logger.info(f"Training for {num_epochs} epochs")
    logger.info(f"Batch size: {train_cfg['batch_size']} x {accumulation_steps} accumulation x {world_size} GPUs = {effective_batch_size} effective")
    logger.info(f"Loss weights: primary={w_primary}, deep={w_deep}, logic={w_logic}, halt={w_halt}, comp={w_comp}, obj={w_object}, centroid={w_centroid}, change={change_weight}")

    best_nonbg_acc = 0.0
    global_step = start_step  # Initialize global step counter
    current_lr = 0.0  # Track current LR for logging
    model.train()

    for epoch in range(start_epoch, num_epochs + 1):
        # IterableDataset handles shuffling internally via seed + iteration counter

        # === FIX #3: Force Minimum Steps ===
        # Force 12 steps always - model was halting at 4 and plateauing
        current_min_steps = 12

        total_loss = 0.0
        epoch_losses = {'primary': 0, 'deep': 0, 'logic': 0, 'halt': 0, 'comp': 0, 'object': 0, 'centroid': 0, 'change_pct': 0}

        # Accumulation state tracking
        accum_step = 0
        accum_loss = 0.0
        accum_losses = {k: 0.0 for k in epoch_losses}
        accum_metrics = {'correct': 0, 'total_px': 0, 'solves': 0, 'count': 0}

        # Iterate over train_loader (ParallelLoader on TPU, direct on GPU)
        for batch_idx, batch in enumerate(train_loader):
            # NOTE: TOPAS-DSPL now uses v3 grid format (tuples), not sequence format (dicts)
            # The old sequence-based TOPAS mode is disabled - using unified V3 path with TOPAS detection
            is_topas_mode = False  # Disabled: now using v3 tuples with is_topas detection below

            if is_topas_mode:
                # Sequence batch: dict with 'inputs', 'labels', 'puzzle_id'
                inputs = batch['inputs'].to(device)
                labels = batch['labels'].to(device)
                puzzle_ids = batch['puzzle_id'].to(device) if 'puzzle_id' in batch else None

                if use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs, labels=labels, puzzle_identifiers=puzzle_ids, return_loss=True)
                        loss = outputs['loss']
                        q_halt = outputs.get('q_halt')
                        q_continue = outputs.get('q_continue')

                        # Add Q-learning loss if enabled
                        if w_q_halt > 0 and q_halt is not None:
                            # Q-learning: encourage halting when confident
                            q_loss = F.softplus(-q_halt).mean()  # Encourage q_halt > 0
                            loss = loss + w_q_halt * q_loss
                else:
                    outputs = model(inputs, labels=labels, puzzle_identifiers=puzzle_ids, return_loss=True)
                    loss = outputs['loss']
                    q_halt = outputs.get('q_halt')

                    if w_q_halt > 0 and q_halt is not None:
                        q_loss = F.softplus(-q_halt).mean()
                        loss = loss + w_q_halt * q_loss

                # Accumulate loss
                loss = loss / accumulation_steps
                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                accum_loss += loss.item()

                # Track metrics
                with torch.no_grad():
                    preds = outputs['logits'].argmax(dim=-1)
                    # Only count non-PAD tokens
                    valid_mask = labels != 0  # PAD=0
                    correct = ((preds == labels) & valid_mask).sum().item()
                    total = valid_mask.sum().item()
                    accum_metrics['correct'] += correct
                    accum_metrics['total_px'] += total
                    accum_metrics['count'] += inputs.size(0)

                    # Check for exact solves (all non-PAD tokens match)
                    for b in range(inputs.size(0)):
                        b_valid = valid_mask[b]
                        if ((preds[b] == labels[b]) & b_valid).sum() == b_valid.sum():
                            accum_metrics['solves'] += 1

                # Batch-level logging
                if (batch_idx + 1) % log_interval == 0:
                    batch_loss = loss.item() * accumulation_steps  # Undo scaling
                    batch_acc = accum_metrics['correct'] / max(accum_metrics['total_px'], 1)
                    batch_solves = accum_metrics['solves']
                    batch_count = accum_metrics['count']
                    logger.info(f"Epoch {epoch} Batch {batch_idx+1}/{batches_per_iter} - Loss: {batch_loss:.4f} ACC: {batch_acc*100:.1f}% solves {batch_solves}/{batch_count}")

                # Optimizer step
                if (batch_idx + 1) % accumulation_steps == 0:
                    if use_tpu:
                        # TPU: Use xm.optimizer_step for gradient sync across cores
                        xm.optimizer_step(optimizer)
                    elif use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()

                    global_step += 1
                    lr = cosine_schedule_with_warmup(global_step, base_lr, lr_warmup_steps, total_training_steps, lr_min_ratio)
                    for pg in optimizer.param_groups:
                        pg['lr'] = lr

                    if use_ema and ema is not None:
                        ema.update(model)

                    # Step-based evaluation (TOPAS mode)
                    eval_step_interval = train_cfg.get("eval_step_interval", 0)
                    first_eval_step = train_cfg.get("first_eval_step", 0)
                    should_eval = (first_eval_step > 0 and global_step == first_eval_step) or \
                                  (eval_step_interval > 0 and global_step % eval_step_interval == 0)
                    if should_eval and rank == 0:
                        logger.info(f"[Step {global_step}] Running evaluation...")
                        eval_model = ema_model if use_ema else model
                        if hasattr(eval_model, 'module'):
                            eval_model = eval_model.module
                        eval_model.eval()
                        step_eval_results = evaluate(eval_model, eval_data, device, min_steps=current_min_steps)
                        model.train()

                        step_acc = step_eval_results["accuracy"]
                        step_avg_steps = step_eval_results["avg_steps"]
                        step_pixel_acc = step_eval_results["pixel_acc"]
                        step_nonzero_acc = step_eval_results["nonzero_acc"]
                        step_mean_iou = step_eval_results["mean_iou"]
                        step_partial = step_eval_results["partial_matches"]
                        step_per_class = step_eval_results["per_class_acc"]

                        # Full eval logging
                        logger.info(
                            f"[Step {global_step}] Eval: {step_eval_results['solved']}/{step_eval_results['total']} solved "
                            f"({step_acc*100:.2f}%) | Pixel Acc: {step_pixel_acc*100:.2f}% | "
                            f"Non-BG Acc: {step_nonzero_acc*100:.2f}% | mIoU: {step_mean_iou*100:.2f}%"
                        )
                        logger.info(
                            f"[Step {global_step}] Partial matches: >90%: {step_partial['>90%']}, "
                            f">80%: {step_partial['>80%']}, >70%: {step_partial['>70%']}, >50%: {step_partial['>50%']} | "
                            f"Avg Steps: {step_avg_steps:.1f}"
                        )
                        class_strs = [f"C{i}:{v*100:.1f}%" for i, v in enumerate(step_per_class) if v is not None]
                        if class_strs:
                            logger.info(f"[Step {global_step}] Per-class Acc: {' '.join(class_strs)}")

                        task_nonbg_list = step_eval_results.get("task_nonbg_accs", [])
                        if task_nonbg_list:
                            sorted_tasks = sorted(enumerate(task_nonbg_list), key=lambda x: x[1], reverse=True)
                            top_10 = sorted_tasks[:10]
                            top_strs = [f"T{idx}:{acc*100:.1f}%" for idx, acc in top_10]
                            logger.info(f"[Step {global_step}] Top 10 tasks (Non-BG): {' '.join(top_strs)}")
                            best_idx, best_acc = top_10[0]
                            logger.info(f"[Step {global_step}] Best task: T{best_idx} at {best_acc*100:.2f}% Non-BG")

                        # TensorBoard logging
                        if logger.tb_writer:
                            logger.tb_writer.add_scalar("eval_step/accuracy", step_acc, global_step)
                            logger.tb_writer.add_scalar("eval_step/pixel_accuracy", step_pixel_acc, global_step)
                            logger.tb_writer.add_scalar("eval_step/nonzero_accuracy", step_nonzero_acc, global_step)
                            logger.tb_writer.add_scalar("eval_step/mean_iou", step_mean_iou, global_step)
                            logger.tb_writer.add_scalar("eval_step/avg_steps", step_avg_steps, global_step)

                        # Generate PDF visualization
                        try:
                            from visualization import create_eval_pdf_visualization
                            vis_results = create_eval_pdf_visualization(
                                eval_model, eval_data, device, global_step, logger.vis_dir,
                                tasks_per_page=10, pad_class=10, min_steps=current_min_steps,
                                filename_prefix=f"step_{global_step}"
                            )
                            logger.info(
                                f"[Step {global_step}] Generated PDF: {vis_results['pdf_path']} "
                                f"({vis_results['total_correct']}/{vis_results['total']} correct)"
                            )
                        except Exception as e:
                            logger.warning(f"Failed to generate PDF visualization: {e}")

                        # Save checkpoint after each step-based eval (crash protection)
                        logger.save_checkpoint(model, optimizer, epoch, best=False, suffix=f"_step{global_step}")
                        logger.info(f"[Step {global_step}] Checkpoint saved")

                    # Reset accumulation metrics after optimizer step
                    accum_loss = 0.0
                    accum_metrics = {'correct': 0, 'total_px': 0, 'solves': 0, 'count': 0}

                continue  # Skip V3-style processing below

            # V3/DSPL mode: tuple-style batch
            # Handle different tuple formats:
            # 8-tuple: with object mask (new format)
            # 7-tuple: with content mask (old format)
            # 6-tuple: legacy format
            if len(batch) == 8:
                train_in, train_out, test_in, target_out, demo_mask, task_ids, content_mask, test_in_obj_mask = batch
                content_mask = content_mask.to(device)
                test_in_obj_mask = test_in_obj_mask.to(device)
            elif len(batch) == 7:
                train_in, train_out, test_in, target_out, demo_mask, task_ids, content_mask = batch
                content_mask = content_mask.to(device)
                test_in_obj_mask = None
            else:
                train_in, train_out, test_in, target_out, demo_mask, task_ids = batch
                content_mask = None
                test_in_obj_mask = None

            train_in = train_in.to(device)
            train_out = train_out.to(device)
            test_in = test_in.to(device)
            target_out = target_out.to(device)
            demo_mask = demo_mask.to(device)
            task_ids = task_ids.to(device)

            # Compute target object counts and centroids for auxiliary loss
            target_object_counts, target_centroids = compute_batch_object_info(target_out) if (w_object > 0 or w_centroid > 0) else (None, None)

            # Forward pass with task_ids and object_mask for puzzle embeddings
            # Check if model is TOPAS (has q_head) vs V3 (has object_count_head)
            is_topas = hasattr(model, 'q_head')

            if use_amp:
                with torch.amp.autocast('cuda'):
                    result = model(
                        train_in, train_out, test_in, demo_mask,
                        task_ids=task_ids,
                        return_intermediate=True,
                        **({'return_logic_tokens': True, 'min_steps': current_min_steps, 'object_mask': test_in_obj_mask} if not is_topas else {})
                    )
                    final_logits, inter_logits_list, halting_list, logic_states, ponder_cost = result[:5]

                    # TOPAS returns q_logits in position 5, V3 returns object_count_logits
                    if is_topas:
                        q_logits = result[5]
                        object_count_logits, centroid_pred = None, None
                    else:
                        object_count_logits, centroid_pred = result[5], result[6]
                        q_logits = None

                    loss, loss_dict = compute_loss(
                        final_logits, inter_logits_list, halting_list, target_out,
                        logic_states, ponder_cost,
                        w_primary, w_deep, w_logic, w_halt, w_comp,
                        content_mask=content_mask, test_in=test_in, change_weight=change_weight,
                        object_count_logits=object_count_logits, target_object_counts=target_object_counts, w_object=w_object,
                        centroid_pred=centroid_pred, target_centroids=target_centroids, w_centroid=w_centroid,
                        q_logits=q_logits, w_q_halt=w_q_halt
                    )
            else:
                result = model(
                    train_in, train_out, test_in, demo_mask,
                    task_ids=task_ids,
                    return_intermediate=True,
                    **({'return_logic_tokens': True, 'min_steps': current_min_steps, 'object_mask': test_in_obj_mask} if not is_topas else {})
                )
                final_logits, inter_logits_list, halting_list, logic_states, ponder_cost = result[:5]

                if is_topas:
                    q_logits = result[5]
                    object_count_logits, centroid_pred = None, None
                else:
                    object_count_logits, centroid_pred = result[5], result[6]
                    q_logits = None

                loss, loss_dict = compute_loss(
                    final_logits, inter_logits_list, halting_list, target_out,
                    logic_states, ponder_cost,
                    w_primary, w_deep, w_logic, w_halt, w_comp,
                    content_mask=content_mask, test_in=test_in, change_weight=change_weight,
                    object_count_logits=object_count_logits, target_object_counts=target_object_counts, w_object=w_object,
                    centroid_pred=centroid_pred, target_centroids=target_centroids, w_centroid=w_centroid,
                    q_logits=q_logits, w_q_halt=w_q_halt
                )

            # Scale loss for gradient accumulation
            scaled_loss = loss / accumulation_steps

            # Backward pass (accumulates gradients)
            if use_amp:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            # Accumulate metrics for logging
            accum_loss += loss.item()
            for k in accum_losses:
                accum_losses[k] += loss_dict.get(k, 0)

            # Accumulate accuracy metrics
            with torch.no_grad():
                pred = final_logits.argmax(dim=1)
                target = target_out.argmax(dim=1)
                actual_batch_size = pred.shape[0]

                if content_mask is not None:
                    accum_metrics['correct'] += ((pred == target) & content_mask).sum().item()
                    accum_metrics['total_px'] += content_mask.sum().item()
                else:
                    accum_metrics['correct'] += (pred == target).sum().item()
                    accum_metrics['total_px'] += pred.numel()

                for b in range(actual_batch_size):
                    if content_mask is not None:
                        # All content pixels must be correct for a solve
                        mask_b = content_mask[b]
                        content_correct = ((pred[b] == target[b]) & mask_b).sum()
                        content_total = mask_b.sum()
                        if content_correct == content_total and content_total > 0:
                            accum_metrics['solves'] += 1
                    else:
                        if (pred[b] == target[b]).all():
                            accum_metrics['solves'] += 1
                accum_metrics['count'] += actual_batch_size

            accum_step += 1

            # Step optimizer after accumulation_steps batches
            if accum_step >= accumulation_steps:
                # === Per-step LR scheduling ===
                for opt, base_lr in zip(optimizers, optimizer_lrs):
                    current_lr = cosine_schedule_with_warmup(
                        current_step=global_step,
                        base_lr=base_lr,
                        num_warmup_steps=lr_warmup_steps,
                        num_training_steps=total_training_steps,
                        min_ratio=lr_min_ratio
                    )
                    for param_group in opt.param_groups:
                        param_group['lr'] = current_lr

                # Clip gradients and step optimizer
                if use_tpu:
                    # TPU: Use xm.optimizer_step for gradient sync across cores
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    for opt in optimizers:
                        xm.optimizer_step(opt)
                elif use_amp:
                    for opt in optimizers:
                        scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    for opt in optimizers:
                        scaler.step(opt)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    for opt in optimizers:
                        opt.step()

                # Zero gradients for next accumulation
                for opt in optimizers:
                    opt.zero_grad()

                # Increment global step
                global_step += 1

                # Step-based evaluation (if configured)
                eval_step_interval = train_cfg.get("eval_step_interval", 0)
                first_eval_step = train_cfg.get("first_eval_step", 0)
                should_eval = (first_eval_step > 0 and global_step == first_eval_step) or \
                              (eval_step_interval > 0 and global_step % eval_step_interval == 0)
                if should_eval and rank == 0:
                    logger.info(f"[Step {global_step}] Running step-based evaluation...")
                    eval_model = ema_model if use_ema else model
                    if hasattr(eval_model, 'module'):
                        eval_model = eval_model.module
                    eval_model.eval()
                    step_eval_results = evaluate(eval_model, eval_data, device, min_steps=current_min_steps)
                    model.train()

                    step_acc = step_eval_results["accuracy"]
                    step_avg_steps = step_eval_results["avg_steps"]
                    step_pixel_acc = step_eval_results["pixel_acc"]
                    step_nonzero_acc = step_eval_results["nonzero_acc"]
                    step_mean_iou = step_eval_results["mean_iou"]
                    step_partial = step_eval_results["partial_matches"]
                    step_per_class = step_eval_results["per_class_acc"]

                    # Full eval logging (matches epoch-end format)
                    logger.info(
                        f"[Step {global_step}] Eval: {step_eval_results['solved']}/{step_eval_results['total']} solved "
                        f"({step_acc*100:.2f}%) | Pixel Acc: {step_pixel_acc*100:.2f}% | "
                        f"Non-BG Acc: {step_nonzero_acc*100:.2f}% | mIoU: {step_mean_iou*100:.2f}%"
                    )
                    logger.info(
                        f"[Step {global_step}] Partial matches: >90%: {step_partial['>90%']}, "
                        f">80%: {step_partial['>80%']}, >70%: {step_partial['>70%']}, >50%: {step_partial['>50%']} | "
                        f"Avg Steps: {step_avg_steps:.1f}"
                    )
                    # Per-class accuracy
                    class_strs = [f"C{i}:{v*100:.1f}%" for i, v in enumerate(step_per_class) if v is not None]
                    if class_strs:
                        logger.info(f"[Step {global_step}] Per-class Acc: {' '.join(class_strs)}")

                    # Per-task non-BG accuracy - show top 10 performers
                    task_nonbg_list = step_eval_results.get("task_nonbg_accs", [])
                    if task_nonbg_list:
                        sorted_tasks = sorted(enumerate(task_nonbg_list), key=lambda x: x[1], reverse=True)
                        top_10 = sorted_tasks[:10]
                        top_strs = [f"T{idx}:{acc*100:.1f}%" for idx, acc in top_10]
                        logger.info(f"[Step {global_step}] Top 10 tasks (Non-BG): {' '.join(top_strs)}")
                        best_idx, best_acc = top_10[0]
                        logger.info(f"[Step {global_step}] Best task: T{best_idx} at {best_acc*100:.2f}% Non-BG")

                    # TensorBoard logging
                    if logger.tb_writer:
                        logger.tb_writer.add_scalar("eval_step/accuracy", step_acc, global_step)
                        logger.tb_writer.add_scalar("eval_step/pixel_accuracy", step_pixel_acc, global_step)
                        logger.tb_writer.add_scalar("eval_step/nonzero_accuracy", step_nonzero_acc, global_step)
                        logger.tb_writer.add_scalar("eval_step/mean_iou", step_mean_iou, global_step)
                        logger.tb_writer.add_scalar("eval_step/avg_steps", step_avg_steps, global_step)
                        logger.tb_writer.add_scalar("eval_step/partial_90", step_partial['>90%'], global_step)
                        logger.tb_writer.add_scalar("eval_step/partial_80", step_partial['>80%'], global_step)

                    # Generate PDF visualization for step-based evals
                    try:
                        from visualization import create_eval_pdf_visualization
                        vis_results = create_eval_pdf_visualization(
                            eval_model, eval_data, device, global_step, logger.vis_dir,
                            tasks_per_page=10, pad_class=10, min_steps=current_min_steps,
                            filename_prefix=f"step_{global_step}"
                        )
                        logger.info(
                            f"[Step {global_step}] Generated PDF visualization: {vis_results['pdf_path']} "
                            f"({vis_results['total_correct']}/{vis_results['total']} correct)"
                        )
                        logger.info(
                            f"[Step {global_step}] Errors: Wrong={vis_results['total_wrong_color']}, "
                            f"Missed={vis_results['total_missed']}, FP={vis_results['total_false_pos']}"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to generate PDF visualization: {e}")

                    # Save checkpoint after each step-based eval (crash protection)
                    logger.save_checkpoint(model, optimizer, epoch, best=False, suffix=f"_step{global_step}")
                    logger.info(f"[Step {global_step}] Checkpoint saved")

                # Update EMA model
                if use_ema:
                    with torch.no_grad():
                        m_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                        for k, v in ema_model.state_dict().items():
                            if k in m_state:
                                v.copy_(ema_decay * v + (1 - ema_decay) * m_state[k])

                # Logging (only after optimizer step)
                if rank == 0:
                    avg_loss = accum_loss / accumulation_steps
                    acc = accum_metrics['correct'] / accum_metrics['total_px'] if accum_metrics['total_px'] > 0 else 0.0
                    exact_acc = accum_metrics['solves'] / accum_metrics['count'] if accum_metrics['count'] > 0 else 0.0

                    # Log line (use logger.info to ensure it's captured)
                    logger.info(
                        f"[{global_step}/{total_training_steps}] "
                        f"Loss={avg_loss:.4f} Acc={acc*100:.1f}% "
                        f"Solves={accum_metrics['solves']}/{accum_metrics['count']} ({exact_acc*100:.1f}%)"
                    )

                    # TensorBoard logging
                    if logger.tb_writer:
                        logger.tb_writer.add_scalar("train/loss", avg_loss, global_step)
                        logger.tb_writer.add_scalar("train/accuracy", acc, global_step)
                        logger.tb_writer.add_scalar("train/exact_accuracy", exact_acc, global_step)
                        logger.tb_writer.add_scalar("train/count", accum_metrics['count'], global_step)
                        logger.tb_writer.add_scalar("train/lr", current_lr, global_step)
                        for key, value in accum_losses.items():
                            logger.tb_writer.add_scalar(f"train/loss_{key}", value / accumulation_steps, global_step)

                # Accumulate for epoch summary
                total_loss += accum_loss
                for k in epoch_losses:
                    epoch_losses[k] += accum_losses[k]

                # Reset accumulation state
                accum_step = 0
                accum_loss = 0.0
                accum_losses = {k: 0.0 for k in epoch_losses}
                accum_metrics = {'correct': 0, 'total_px': 0, 'solves': 0, 'count': 0}

        # Note: LR is now updated per-step, not per-epoch

        # Log epoch summary
        if rank == 0:
            # n_steps = number of optimizer steps this epoch (forward passes / accumulation_steps)
            n_forward = batch_idx + 1
            n_steps = n_forward // accumulation_steps
            if n_steps > 0:
                # Divide by total forward passes to get true per-batch average
                n_batches = n_steps * accumulation_steps
                avg_epoch_losses = {k: v / n_batches for k, v in epoch_losses.items()}
                logger.log_epoch(epoch, total_loss / n_batches, avg_epoch_losses, current_lr)

        # Epoch end: evaluate
        if rank == 0 and epoch % train_cfg.get("eval_interval", 1) == 0:
            eval_model = ema_model if use_ema else model
            if hasattr(eval_model, 'module'):
                eval_model = eval_model.module
            eval_results = evaluate(eval_model, eval_data, device, min_steps=current_min_steps)
            model.train()

            acc = eval_results["accuracy"]
            avg_steps = eval_results["avg_steps"]
            pixel_acc = eval_results["pixel_acc"]
            nonzero_acc = eval_results["nonzero_acc"]
            mean_iou = eval_results["mean_iou"]
            partial = eval_results["partial_matches"]
            per_class = eval_results["per_class_acc"]

            # Detailed eval logging
            logger.info(
                f"[Epoch {epoch}] Eval: {eval_results['solved']}/{eval_results['total']} solved "
                f"({acc*100:.2f}%) | Pixel Acc: {pixel_acc*100:.2f}% | "
                f"Non-BG Acc: {nonzero_acc*100:.2f}% | mIoU: {mean_iou*100:.2f}%"
            )
            logger.info(
                f"[Epoch {epoch}] Partial matches: >90%: {partial['>90%']}, "
                f">80%: {partial['>80%']}, >70%: {partial['>70%']}, >50%: {partial['>50%']} | "
                f"Avg Steps: {avg_steps:.1f}"
            )
            # Per-class accuracy (only show classes with data)
            class_strs = [f"C{i}:{v*100:.1f}%" for i, v in enumerate(per_class) if v is not None]
            if class_strs:
                logger.info(f"[Epoch {epoch}] Per-class Acc: {' '.join(class_strs)}")

            # Per-task non-BG accuracy - show top 10 performers (using non-BG acc, not pixel acc)
            task_nonbg_list = eval_results.get("task_nonbg_accs", [])
            if task_nonbg_list:
                # Sort tasks by non-BG accuracy (descending) with their indices
                sorted_tasks = sorted(enumerate(task_nonbg_list), key=lambda x: x[1], reverse=True)
                top_10 = sorted_tasks[:10]
                top_strs = [f"T{idx}:{acc*100:.1f}%" for idx, acc in top_10]
                logger.info(f"[Epoch {epoch}] Top 10 tasks (Non-BG): {' '.join(top_strs)}")
                # Also log the best accuracy
                best_idx, best_acc = top_10[0]
                logger.info(f"[Epoch {epoch}] Best task: T{best_idx} at {best_acc*100:.2f}% Non-BG")

            # Log to TensorBoard
            logger.log_eval(epoch, acc, avg_steps, eval_results['solved'], eval_results['total'])
            if logger.tb_writer:
                logger.tb_writer.add_scalar("eval/pixel_accuracy", pixel_acc, epoch)
                logger.tb_writer.add_scalar("eval/nonzero_accuracy", nonzero_acc, epoch)
                logger.tb_writer.add_scalar("eval/mean_iou", mean_iou, epoch)
                logger.tb_writer.add_scalar("eval/partial_90", partial['>90%'], epoch)
                logger.tb_writer.add_scalar("eval/partial_80", partial['>80%'], epoch)
                # Per-class to TensorBoard
                for i, v in enumerate(per_class):
                    if v is not None:
                        logger.tb_writer.add_scalar(f"eval/class_{i}_acc", v, epoch)

            # Generate multi-page PDF visualization with error mapping
            try:
                from visualization import create_eval_pdf_visualization
                vis_results = create_eval_pdf_visualization(
                    eval_model, eval_data, device, epoch, logger.vis_dir,
                    tasks_per_page=10, pad_class=10, min_steps=current_min_steps
                )
                logger.info(
                    f"[Epoch {epoch}] Generated PDF visualization: {vis_results['pdf_path']} "
                    f"({vis_results['total_correct']}/{vis_results['total']} correct)"
                )
                logger.info(
                    f"[Epoch {epoch}] Errors: Wrong={vis_results['total_wrong_color']}, "
                    f"Missed={vis_results['total_missed']}, FP={vis_results['total_false_pos']}"
                )
            except Exception as e:
                logger.warning(f"Failed to generate PDF visualization: {e}")

            # Log halting probability distribution (from first task)
            # Skip for TOPAS mode - halting logging requires V3 grid format
            if len(eval_data) > 0 and not train_cfg.get("_topas_mode", False):
                with torch.no_grad():
                    test_sample = eval_data[0]
                    t_in = test_sample[0].unsqueeze(0).to(device)
                    t_out = test_sample[1].unsqueeze(0).to(device)
                    test_in = test_sample[2].unsqueeze(0).to(device)
                    target = test_sample[3].unsqueeze(0).to(device)
                    mask = test_sample[4].unsqueeze(0).to(device)
                    result = eval_model(
                        t_in, t_out, test_in, mask, return_intermediate=True
                    )
                    _, inter_logits, halt_probs = result[0], result[1], result[2]
                    logger.log_halting_probs(epoch, halt_probs)

            # Save best model based on Non-BG accuracy
            if nonzero_acc > best_nonbg_acc:
                best_nonbg_acc = nonzero_acc
                logger.save_checkpoint(model, optimizer, epoch, best=True, suffix="_nonbg")
                logger.info(f"New best model saved (Non-BG Acc): {nonzero_acc*100:.2f}%")

    # Save final model and close logger
    if rank == 0:
        logger.save_checkpoint(model, optimizer, num_epochs, best=False)
        logger.info(f"Best Non-BG accuracy achieved: {best_nonbg_acc*100:.2f}%")
        logger.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train DSPL model')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file (default: config.yaml)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (overrides config)')
    # TPU support
    parser.add_argument('--tpu', action='store_true',
                        help='Use TPU instead of GPU (requires torch_xla)')
    parser.add_argument('--tpu-cores', type=int, default=8,
                        help='Number of TPU cores (default: 8 for v5e-8)')
    args = parser.parse_args()
    train(config_path=args.config, resume_checkpoint=args.resume,
          use_tpu=args.tpu, tpu_cores=args.tpu_cores)
