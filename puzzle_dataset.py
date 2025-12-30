"""
PuzzleDataset for TOPAS model.

Data loading strategy:
1. IterableDataset with memory-mapped arrays
2. Epoch-based group shuffling
3. Hierarchical sampling: groups → puzzles → examples
4. Returns data in one-hot grid format
"""
import os
import json
from typing import List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info


def _sample_puzzles_for_batch(
    rng: np.random.Generator,
    group_order: np.ndarray,
    puzzle_indices: np.ndarray,
    group_indices: np.ndarray,
    start_index: int,
    batch_size: int
) -> Tuple[int, List[int]]:
    """
    Sample puzzles for a batch - one puzzle per batch item.

    Uses group shuffling for task-based batching.
    Each batch item is one full puzzle (with demos and test).
    """
    puzzle_ids = []

    while (start_index < group_order.size) and (len(puzzle_ids) < batch_size):
        # Pick a group and a random puzzle (augmentation) from that group
        group_id = group_order[start_index]
        puzzle_id = rng.integers(group_indices[group_id], group_indices[group_id + 1])
        start_index += 1
        puzzle_ids.append(puzzle_id)

    return start_index, puzzle_ids


class PuzzleDataset(IterableDataset):
    """
    IterableDataset for TOPAS model.

    Key features:
    - Memory-mapped arrays for efficiency with 1M+ examples
    - Epoch-based group shuffling (reproducible with seed)
    - Hierarchical sampling: groups → puzzles → examples
    - Lazy loading on first iteration

    Output format (depends on output_mode):
    - "v3": Converts 900-length sequences to 30x30 one-hot grids
            Returns (train_in, train_out, test_in, test_out, demo_mask, task_id, content_mask)
    - "sequence": Returns token sequences directly
            Returns dict with 'inputs', 'labels', 'puzzle_id' keys
    """

    def __init__(
        self,
        dataset_paths: List[str],
        split: str = "train",
        seed: int = 42,
        global_batch_size: int = 64,
        epochs_per_iter: int = 1,
        rank: int = 0,
        num_replicas: int = 1,
        max_demos: int = 3,
        img_size: int = 30,
        num_colors: int = 11,
        test_set_mode: bool = False,
        output_mode: str = "v3"
    ):
        """
        Args:
            dataset_paths: List of paths to dataset directories
            split: "train" or "test"
            seed: Random seed for reproducibility
            global_batch_size: Total batch size across all replicas
            epochs_per_iter: Number of epochs to batch together (reduces overhead)
            rank: Current replica rank (for distributed training)
            num_replicas: Total number of replicas
            max_demos: Maximum number of demonstrations per task
            img_size: Grid size (30x30)
            num_colors: Number of color classes (11 = 10 colors + PAD)
            test_set_mode: If True, iterate sequentially (for eval)
            output_mode: "v3" for one-hot grids, "sequence" for token sequences
        """
        super().__init__()
        self.dataset_paths = dataset_paths
        self.split = split
        self.seed = seed
        self.global_batch_size = global_batch_size
        self.epochs_per_iter = epochs_per_iter
        self.rank = rank
        self.num_replicas = num_replicas
        self.max_demos = max_demos
        self.img_size = img_size
        self.num_colors = num_colors
        self.test_set_mode = test_set_mode
        self.output_mode = output_mode

        # V3 constants
        self.PAD_CLASS = 10

        # Load and merge metadata from all paths
        self._load_metadata()

        # Validate batch size
        assert self.global_batch_size % self.num_replicas == 0, \
            f"Global batch size {self.global_batch_size} must be divisible by num_replicas {self.num_replicas}"
        self.local_batch_size = self.global_batch_size // self.num_replicas

        # State
        self._data = None
        self._iters = 0

    def _load_metadata(self):
        """Load and merge metadata from all dataset paths."""
        self.metadata = None
        total_groups = 0
        total_puzzles = 0
        total_examples = 0
        num_identifiers = 0

        for path in self.dataset_paths:
            meta_path = os.path.join(path, self.split, "dataset.json")
            with open(meta_path, 'r') as f:
                meta = json.load(f)

            if self.metadata is None:
                self.metadata = meta.copy()
            else:
                # Validate compatibility
                assert meta['seq_len'] == self.metadata['seq_len']
                assert meta['vocab_size'] == self.metadata['vocab_size']

            total_groups += meta.get('total_groups', 0)
            total_puzzles += meta.get('total_puzzles', 0)
            num_identifiers += meta.get('num_puzzle_identifiers', 0)

        self.metadata['total_groups'] = total_groups
        self.metadata['total_puzzles'] = total_puzzles
        self.metadata['num_puzzle_identifiers'] = num_identifiers

    def _lazy_load_dataset(self):
        """Lazy load arrays on first iteration (memory-mapped for efficiency)."""
        if self._data is not None:
            return

        field_mmap_modes = {
            "inputs": "r",      # Memory-mapped (3.6GB)
            "labels": "r",      # Memory-mapped (3.6GB)
            "puzzle_identifiers": None,  # Keep in RAM (small)
            "puzzle_indices": None,      # Keep in RAM (small)
            "group_indices": None        # Keep in RAM (small)
        }

        self._data = {}
        sets = self.metadata.get('sets', ['train'])

        for set_name in sets:
            for i, dataset_path in enumerate(self.dataset_paths):
                set_key = set_name if i == 0 else f"{set_name}{i}"
                split_dir = os.path.join(dataset_path, self.split)

                self._data[set_key] = {
                    field_name: np.load(
                        os.path.join(split_dir, f"{set_name}__{field_name}.npy"),
                        mmap_mode=mmap_mode
                    )
                    for field_name, mmap_mode in field_mmap_modes.items()
                }

    def _seq_to_grid(self, seq: np.ndarray) -> Tuple[np.ndarray, int, int, np.ndarray]:
        """
        Convert 900-length sequence to 30x30 grid.

        Sequence format: PAD=0, EOS=1, colors 0-9 = tokens 2-11
        Grid format: colors 0-9, PAD=10 (one-hot encoded)

        Returns:
            grid: [H, W] class indices (0-9 colors, 10=PAD)
            H: actual content height
            W: actual content width
            content_mask: [H, W] boolean mask (True for content)
        """
        seq = seq.reshape(self.img_size, self.img_size)

        # Initialize grid with PAD class (10)
        grid = np.full((self.img_size, self.img_size), self.PAD_CLASS, dtype=np.int64)

        # Find content region (non-PAD, non-EOS)
        # EOS=1 marks boundaries, PAD=0 is padding
        content_mask = (seq >= 2) & (seq <= 11)  # Actual color tokens

        # Convert colors: tokens 2-11 → colors 0-9
        grid[content_mask] = seq[content_mask] - 2

        # Find actual dimensions
        H, W = self.img_size, self.img_size
        for h in range(self.img_size):
            if not content_mask[h, :].any():
                H = h
                break
        for w in range(self.img_size):
            if not content_mask[:, w].any():
                W = w
                break

        # Content mask for loss computation
        out_mask = np.zeros((self.img_size, self.img_size), dtype=bool)
        out_mask[:H, :W] = True

        return grid, H, W, out_mask

    def _prepare_v3_sample(
        self,
        inputs: np.ndarray,
        labels: np.ndarray,
        puzzle_id: int
    ) -> Tuple[torch.Tensor, ...]:
        """
        Convert a puzzle's examples to V3 format.

        Args:
            inputs: [num_examples, 900] input sequences
            labels: [num_examples, 900] label sequences
            puzzle_id: Task identifier

        Returns:
            V3-format tuple: (train_in, train_out, test_in, test_out, mask, task_id, content_mask)
        """
        num_examples = len(inputs)
        n_demos = min(num_examples - 1, self.max_demos)

        # Initialize demo tensors with PAD class
        train_in = torch.zeros(
            (self.max_demos, self.num_colors, self.img_size, self.img_size),
            dtype=torch.float32
        )
        train_out = torch.zeros_like(train_in)
        train_in[:, self.PAD_CLASS, :, :] = 1.0
        train_out[:, self.PAD_CLASS, :, :] = 1.0

        # Fill in demos
        for i in range(n_demos):
            inp_grid, H, W, _ = self._seq_to_grid(inputs[i])
            out_grid, oH, oW, _ = self._seq_to_grid(labels[i])

            # One-hot encode
            inp_onehot = torch.eye(self.num_colors)[torch.from_numpy(inp_grid)].permute(2, 0, 1)
            out_onehot = torch.eye(self.num_colors)[torch.from_numpy(out_grid)].permute(2, 0, 1)

            train_in[i] = inp_onehot
            train_out[i] = out_onehot

        # Test example (last one)
        test_inp_grid, H, W, _ = self._seq_to_grid(inputs[-1])
        test_out_grid, oH, oW, test_content_mask = self._seq_to_grid(labels[-1])

        test_in = torch.eye(self.num_colors)[torch.from_numpy(test_inp_grid)].permute(2, 0, 1)
        test_out = torch.eye(self.num_colors)[torch.from_numpy(test_out_grid)].permute(2, 0, 1)

        # Demo mask (True for padded positions)
        mask = torch.zeros(self.max_demos, dtype=torch.bool)
        if n_demos < self.max_demos:
            mask[n_demos:] = True

        return (
            train_in.float(),
            train_out.float(),
            test_in.float(),
            test_out.float(),
            mask,
            torch.tensor(puzzle_id, dtype=torch.int32),
            torch.from_numpy(test_content_mask)
        )

    def _collate_puzzles(
        self,
        dataset: dict,
        puzzle_ids: List[int]
    ) -> Tuple[torch.Tensor, ...]:
        """
        Collate a batch of puzzles into V3 format.

        Each puzzle becomes one batch item with its demos and test example.
        """
        batch_items = []

        for puzzle_id in puzzle_ids:
            # Get example range for this puzzle
            ex_start = dataset["puzzle_indices"][puzzle_id]
            ex_end = dataset["puzzle_indices"][puzzle_id + 1]

            # Load examples for this puzzle
            inputs = dataset["inputs"][ex_start:ex_end]
            labels = dataset["labels"][ex_start:ex_end]
            # puzzle_identifiers is indexed by example, not puzzle
            # All examples in a puzzle share the same identifier
            task_id = dataset["puzzle_identifiers"][ex_start]

            item = self._prepare_v3_sample(inputs, labels, task_id)
            batch_items.append(item)

        # Stack into batch
        if not batch_items:
            return None

        return tuple(torch.stack([item[i] for item in batch_items]) for i in range(7))

    def _collate_seq_puzzles(
        self,
        dataset: dict,
        puzzle_ids: List[int]
    ) -> dict:
        """
        Collate a batch of puzzles into sequence format.

        Each puzzle becomes one batch item with test example's input/label sequences.

        Returns:
            dict with 'inputs', 'labels', 'puzzle_id' tensors
        """
        inputs_list = []
        labels_list = []
        task_ids_list = []

        for puzzle_id in puzzle_ids:
            # Get example range for this puzzle
            ex_start = dataset["puzzle_indices"][puzzle_id]
            ex_end = dataset["puzzle_indices"][puzzle_id + 1]

            # Use test example (last one in puzzle)
            test_idx = ex_end - 1

            # Load sequences directly (no conversion to grid)
            inputs = torch.from_numpy(dataset["inputs"][test_idx].astype(np.int64))
            labels = torch.from_numpy(dataset["labels"][test_idx].astype(np.int64))
            task_id = dataset["puzzle_identifiers"][ex_start]

            inputs_list.append(inputs)
            labels_list.append(labels)
            task_ids_list.append(task_id)

        if not inputs_list:
            return None

        return {
            'inputs': torch.stack(inputs_list),      # [B, 900]
            'labels': torch.stack(labels_list),      # [B, 900]
            'puzzle_id': torch.tensor(task_ids_list, dtype=torch.long),  # [B]
        }

    def _iter_train(self):
        """Training iteration with group shuffling."""
        for set_name, dataset in self._data.items():
            self._iters += 1

            # Reproducible shuffle based on seed + iteration
            rng = np.random.Generator(np.random.Philox(seed=self.seed + self._iters))

            # Shuffle groups for this epoch(s)
            # Each epoch goes through all groups once with random augmentation selection
            group_order = np.concatenate([
                rng.permutation(dataset["group_indices"].size - 1)
                for _ in range(self.epochs_per_iter)
            ])

            start_index = 0
            while start_index < group_order.size:
                # Sample puzzles for this batch (one puzzle per batch item)
                start_index, puzzle_ids = _sample_puzzles_for_batch(
                    rng,
                    group_order=group_order,
                    puzzle_indices=dataset["puzzle_indices"],
                    group_indices=dataset["group_indices"],
                    start_index=start_index,
                    batch_size=self.global_batch_size,
                )

                # Skip incomplete batches
                if len(puzzle_ids) < self.global_batch_size:
                    break

                # Select current rank's portion of puzzles
                local_start = self.rank * self.local_batch_size
                local_end = (self.rank + 1) * self.local_batch_size
                local_puzzle_ids = puzzle_ids[local_start:local_end]

                # Collate based on output mode
                if self.output_mode == "sequence":
                    batch = self._collate_seq_puzzles(dataset, local_puzzle_ids)
                else:
                    batch = self._collate_puzzles(dataset, local_puzzle_ids)
                if batch is not None:
                    yield batch

    def _iter_test(self):
        """Sequential iteration for evaluation."""
        for set_name, dataset in self._data.items():
            num_puzzles = dataset["puzzle_indices"].size - 1

            for puzzle_id in range(num_puzzles):
                ex_start = dataset["puzzle_indices"][puzzle_id]
                ex_end = dataset["puzzle_indices"][puzzle_id + 1]

                if self.output_mode == "sequence":
                    # Use test example (last one in puzzle)
                    test_idx = ex_end - 1
                    inputs = torch.from_numpy(dataset["inputs"][test_idx].astype(np.int64))
                    labels = torch.from_numpy(dataset["labels"][test_idx].astype(np.int64))
                    task_id = dataset["puzzle_identifiers"][ex_start]
                    # Return as single-item batch dict
                    yield {
                        'inputs': inputs.unsqueeze(0),      # [1, 900]
                        'labels': labels.unsqueeze(0),      # [1, 900]
                        'puzzle_id': torch.tensor([task_id], dtype=torch.long),  # [1]
                    }
                else:
                    inputs = dataset["inputs"][ex_start:ex_end]
                    labels = dataset["labels"][ex_start:ex_end]
                    # puzzle_identifiers indexed by example
                    task_id = dataset["puzzle_identifiers"][ex_start]

                    item = self._prepare_v3_sample(inputs, labels, task_id)
                    # Return as single-item batch
                    yield tuple(t.unsqueeze(0) for t in item)

    def __iter__(self):
        worker_info = get_worker_info()
        assert worker_info is None or worker_info.num_workers == 1, \
            "Multi-worker data loading not supported"

        self._lazy_load_dataset()

        if self.test_set_mode:
            yield from self._iter_test()
        else:
            yield from self._iter_train()


def create_dataloader(
    data_dir: str,
    split: str = "train",
    batch_size: int = 64,
    seed: int = 42,
    max_demos: int = 3,
    test_mode: bool = False
) -> PuzzleDataset:
    """
    Convenience function to create a dataloader.

    Args:
        data_dir: Path to dataset directory
        split: "train" or "test"
        batch_size: Global batch size
        seed: Random seed
        max_demos: Max demonstrations per task
        test_mode: If True, iterate sequentially for eval

    Returns:
        PuzzleDataset instance (use directly with DataLoader or iterate)
    """
    return PuzzleDataset(
        dataset_paths=[data_dir],
        split=split,
        seed=seed,
        global_batch_size=batch_size,
        max_demos=max_demos,
        test_set_mode=test_mode
    )
