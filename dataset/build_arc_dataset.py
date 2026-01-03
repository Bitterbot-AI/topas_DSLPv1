#!/usr/bin/env python3
"""Build augmented ARC dataset for TOPAS-DSPL training.

Ported from TinyRecursiveModels (https://github.com/alexjm/TinyRecursiveModels)
Simplified to use standard argparse instead of argdantic.

Usage:
    python dataset/build_arc_dataset.py \
        --input-prefix data/arc-agi \
        --output-dir data \
        --subsets training \
        --test-set evaluation \
        --num-aug 1000
"""
import os
import json
import hashlib
import argparse
from typing import List, Tuple, Dict
from dataclasses import dataclass

import numpy as np

from dataset.common import PuzzleDatasetMetadata, dihedral_transform, inverse_dihedral_transform


ARC_MAX_GRID_SIZE = 30
ARC_AUGMENT_RETRIES_FACTOR = 5
PUZZLE_ID_SEPARATOR = "|||"


@dataclass
class ARCPuzzle:
    id: str
    examples: List[Tuple[np.ndarray, np.ndarray]]


def arc_grid_to_np(grid: List[List[int]]) -> np.ndarray:
    """Convert ARC grid to numpy array with validation."""
    arr = np.array(grid)
    assert arr.ndim == 2
    assert arr.shape[0] <= ARC_MAX_GRID_SIZE and arr.shape[1] <= ARC_MAX_GRID_SIZE
    assert np.all((arr >= 0) & (arr <= 9))
    return arr.astype(np.uint8)


def np_grid_to_seq_translational_augment(inp: np.ndarray, out: np.ndarray, do_translation: bool):
    """Convert grids to flattened sequences with optional translational augmentation.

    Token mapping: PAD=0, EOS=1, colors=2-11
    """
    if do_translation:
        pad_r = np.random.randint(0, ARC_MAX_GRID_SIZE - max(inp.shape[0], out.shape[0]) + 1)
        pad_c = np.random.randint(0, ARC_MAX_GRID_SIZE - max(inp.shape[1], out.shape[1]) + 1)
    else:
        pad_r = pad_c = 0

    result = []
    for grid in [inp, out]:
        nrow, ncol = grid.shape
        grid = np.pad(
            grid + 2,
            ((pad_r, ARC_MAX_GRID_SIZE - pad_r - nrow), (pad_c, ARC_MAX_GRID_SIZE - pad_c - ncol)),
            constant_values=0
        )

        # Add EOS markers
        eos_row, eos_col = pad_r + nrow, pad_c + ncol
        if eos_row < ARC_MAX_GRID_SIZE:
            grid[eos_row, pad_c:eos_col] = 1
        if eos_col < ARC_MAX_GRID_SIZE:
            grid[pad_r:eos_row, eos_col] = 1

        result.append(grid.flatten())

    return result


def grid_hash(grid: np.ndarray) -> str:
    """Compute hash of a grid for deduplication."""
    assert grid.ndim == 2 and grid.dtype == np.uint8
    buffer = [x.to_bytes(1, byteorder='big') for x in grid.shape]
    buffer.append(grid.tobytes())
    return hashlib.sha256(b"".join(buffer)).hexdigest()


def puzzle_hash(puzzle: dict) -> str:
    """Compute hash of a puzzle for deduplication."""
    hashes = []
    for example_type, example in puzzle.items():
        for input_grid, label_grid in example.examples:
            hashes.append(f"{grid_hash(input_grid)}|{grid_hash(label_grid)}")
    hashes.sort()
    return hashlib.sha256("|".join(hashes).encode()).hexdigest()


def augment(name: str):
    """Create augmentation function with random dihedral transform + color permutation."""
    trans_id = np.random.randint(0, 8)
    # Permute colors 1-9, keep 0 (black/background) fixed
    mapping = np.concatenate([
        np.arange(0, 1, dtype=np.uint8),
        np.random.permutation(np.arange(1, 10, dtype=np.uint8))
    ])

    name_with_aug = f"{name}{PUZZLE_ID_SEPARATOR}t{trans_id}{PUZZLE_ID_SEPARATOR}{''.join(str(x) for x in mapping)}"

    def _map_grid(grid: np.ndarray):
        return dihedral_transform(mapping[grid], trans_id)

    return name_with_aug, _map_grid


def inverse_augment(name: str):
    """Inverse the augmentation for a puzzle name."""
    if PUZZLE_ID_SEPARATOR not in name:
        return name, lambda x: x

    trans_id, perm = name.split(PUZZLE_ID_SEPARATOR)[-2:]
    trans_id = int(trans_id[1:])
    inv_perm = np.argsort(list(perm)).astype(np.uint8)

    def _map_grid(grid: np.ndarray):
        return inv_perm[inverse_dihedral_transform(grid, trans_id)]

    return name.split(PUZZLE_ID_SEPARATOR)[0], _map_grid


def convert_single_arc_puzzle(
    results: dict,
    name: str,
    puzzle: dict,
    aug_count: int,
    dest_mapping: Dict[str, Tuple[str, str]]
):
    """Convert a single ARC puzzle with augmentations."""
    dests = set(dest_mapping.values())
    converted = {dest: ARCPuzzle(name, []) for dest in dests}

    for example_type, examples in puzzle.items():
        dest = dest_mapping[example_type]
        converted[dest].examples.extend([
            (arc_grid_to_np(example["input"]), arc_grid_to_np(example["output"]))
            for example in examples
        ])

    group = [converted]

    # Generate augmentations
    if aug_count > 0:
        hashes = {puzzle_hash(converted)}

        for _trial in range(ARC_AUGMENT_RETRIES_FACTOR * aug_count):
            aug_name, _map_grid = augment(name)

            augmented = {
                dest: ARCPuzzle(aug_name, [(_map_grid(inp), _map_grid(lbl)) for (inp, lbl) in puz.examples])
                for dest, puz in converted.items()
            }
            h = puzzle_hash(augmented)
            if h not in hashes:
                hashes.add(h)
                group.append(augmented)

            if len(group) >= aug_count + 1:
                break

        if len(group) < aug_count + 1:
            print(f"[Puzzle {name}] augmentation not full, only {len(group)}")

    # Append to results
    for dest in dests:
        dest_split, dest_set = dest
        results.setdefault(dest_split, {})
        results[dest_split].setdefault(dest_set, [])
        results[dest_split][dest_set].append([converted[dest] for converted in group])


def load_puzzles_arcagi(
    input_prefix: str,
    subsets: List[str],
    test_set_name: str,
    test_set_name2: str,
    num_aug: int
):
    """Load and process ARC-AGI puzzles."""
    # Destination format: (split_dir, file_prefix)
    # Using "train" prefix to match puzzle_dataset.py expectations
    train_examples_dest = ("train", "train")
    test_examples_map = {
        test_set_name: [(1.0, ("test", "test"))],
        test_set_name2: [(1.0, ("test", "test"))],
        "_default": [(1.0, ("train", "train"))]
    }

    results = {}
    total_puzzles = 0

    for subset_name in subsets:
        challenges_file = f"{input_prefix}_{subset_name}_challenges.json"
        solutions_file = f"{input_prefix}_{subset_name}_solutions.json"

        with open(challenges_file, "r") as f:
            puzzles = json.load(f)

        if os.path.isfile(solutions_file):
            with open(solutions_file, "r") as f:
                sols = json.load(f)
                for puzzle_id in puzzles.keys():
                    for idx, sol_grid in enumerate(sols[puzzle_id]):
                        puzzles[puzzle_id]["test"][idx]["output"] = sol_grid
        else:
            print(f"{subset_name} solutions not found, filling with dummy")
            for puzzle_id, puzzle in puzzles.items():
                for example in puzzle["test"]:
                    example.setdefault("output", [[0]])

        # Shuffle puzzles
        puzzles = list(puzzles.items())
        np.random.shuffle(puzzles)

        for idx, (name, puzzle) in enumerate(puzzles):
            fraction = idx / len(puzzles)
            test_examples_dest = None
            for f, dest in test_examples_map.get(subset_name, test_examples_map["_default"]):
                if fraction < f:
                    test_examples_dest = dest
                    break

            assert test_examples_dest is not None

            convert_single_arc_puzzle(results, name, puzzle, num_aug, {
                "train": train_examples_dest,
                "test": test_examples_dest
            })
            total_puzzles += 1

    print(f"Total puzzles: {total_puzzles}")
    return results


def convert_dataset(
    input_prefix: str,
    output_dir: str,
    subsets: List[str],
    test_set_name: str,
    test_set_name2: str,
    num_aug: int,
    seed: int,
    puzzle_identifiers_start: int
):
    """Main conversion function."""
    np.random.seed(seed)

    data = load_puzzles_arcagi(
        input_prefix, subsets, test_set_name, test_set_name2, num_aug
    )

    # Map global puzzle identifiers
    num_identifiers = puzzle_identifiers_start
    identifier_map = {}
    for split_name, split in data.items():
        for subset_name, subset in split.items():
            for group in subset:
                for puzzle in group:
                    if puzzle.id not in identifier_map:
                        identifier_map[puzzle.id] = num_identifiers
                        num_identifiers += 1
    print(f"Total puzzle IDs (including <blank>): {num_identifiers}")

    # Save splits
    for split_name, split in data.items():
        os.makedirs(os.path.join(output_dir, split_name), exist_ok=True)

        enable_translational_augment = split_name == "train"
        total_examples = 0
        total_puzzles = 0
        total_groups = 0

        for subset_name, subset in split.items():
            results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
            results["puzzle_indices"].append(0)
            results["group_indices"].append(0)

            example_id = 0
            puzzle_id = 0

            for group in subset:
                for puzzle in group:
                    no_aug_id = np.random.randint(0, len(puzzle.examples))
                    puzzle_identifier = identifier_map[puzzle.id]
                    for _idx_ex, (inp, out) in enumerate(puzzle.examples):
                        inp, out = np_grid_to_seq_translational_augment(
                            inp, out,
                            do_translation=enable_translational_augment and _idx_ex != no_aug_id
                        )
                        results["inputs"].append(inp)
                        results["labels"].append(out)
                        # Store identifier per example (indexed by example in puzzle_dataset.py)
                        results["puzzle_identifiers"].append(puzzle_identifier)
                        example_id += 1
                        total_examples += 1

                    results["puzzle_indices"].append(example_id)
                    puzzle_id += 1
                    total_puzzles += 1

                results["group_indices"].append(puzzle_id)
                total_groups += 1

            for k, v in results.items():
                if k in {"inputs", "labels"}:
                    v = np.stack(v, 0)
                else:
                    v = np.array(v, dtype=np.int32)
                np.save(os.path.join(output_dir, split_name, f"{subset_name}__{k}.npy"), v)

        # Save metadata
        metadata = PuzzleDatasetMetadata(
            seq_len=ARC_MAX_GRID_SIZE * ARC_MAX_GRID_SIZE,
            vocab_size=10 + 2,  # PAD + EOS + colors 0-9
            pad_id=0,
            ignore_label_id=0,
            blank_identifier_id=0,
            num_puzzle_identifiers=num_identifiers,
            total_groups=total_groups,
            mean_puzzle_examples=total_examples / total_puzzles,
            total_puzzles=total_puzzles,
            sets=list(split.keys())
        )
        with open(os.path.join(output_dir, split_name, "dataset.json"), "w") as f:
            json.dump(metadata.to_dict(), f)

    print(f"Dataset saved to {output_dir}")
    print(f"  Train: {total_groups} groups, {total_puzzles} puzzles, {total_examples} examples")


def main():
    parser = argparse.ArgumentParser(
        description="Build augmented ARC dataset for TOPAS-DSPL training"
    )
    parser.add_argument(
        "--input-prefix", required=True,
        help="Prefix for input files (e.g., 'data/arc-agi' for 'data/arc-agi_training_challenges.json')"
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory for processed dataset"
    )
    parser.add_argument(
        "--subsets", nargs="+", default=["training"],
        help="Subset names to process (default: training)"
    )
    parser.add_argument(
        "--test-set", default="evaluation",
        help="Name of test set subset (default: evaluation)"
    )
    parser.add_argument(
        "--test-set2", default="your_test_set",
        help="Name of secondary test set (default: your_test_set)"
    )
    parser.add_argument(
        "--num-aug", type=int, default=1000,
        help="Number of augmentations per puzzle (default: 1000)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--id-start", type=int, default=1,
        help="Starting puzzle identifier (default: 1)"
    )

    args = parser.parse_args()

    convert_dataset(
        input_prefix=args.input_prefix,
        output_dir=args.output_dir,
        subsets=args.subsets,
        test_set_name=args.test_set,
        test_set_name2=args.test_set2,
        num_aug=args.num_aug,
        seed=args.seed,
        puzzle_identifiers_start=args.id_start
    )


if __name__ == "__main__":
    main()
