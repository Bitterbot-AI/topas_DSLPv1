"""Common utilities for dataset building.

Ported from TinyRecursiveModels (https://github.com/alexjm/TinyRecursiveModels)
"""
from typing import List, Optional
from dataclasses import dataclass, asdict
import numpy as np


# Dihedral group D8 inverse mapping
DIHEDRAL_INVERSE = [0, 3, 2, 1, 4, 5, 6, 7]


@dataclass
class PuzzleDatasetMetadata:
    """Metadata for serialized puzzle dataset."""
    pad_id: int
    ignore_label_id: Optional[int]
    blank_identifier_id: int
    vocab_size: int
    seq_len: int
    num_puzzle_identifiers: int
    total_groups: int
    mean_puzzle_examples: float
    total_puzzles: int
    sets: List[str]

    def to_dict(self):
        return asdict(self)


def dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    """Apply one of 8 dihedral symmetry transforms."""
    if tid == 0:
        return arr
    elif tid == 1:
        return np.rot90(arr, k=1)
    elif tid == 2:
        return np.rot90(arr, k=2)
    elif tid == 3:
        return np.rot90(arr, k=3)
    elif tid == 4:
        return np.fliplr(arr)
    elif tid == 5:
        return np.flipud(arr)
    elif tid == 6:
        return arr.T
    elif tid == 7:
        return np.fliplr(np.rot90(arr, k=1))
    else:
        return arr


def inverse_dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    """Apply inverse dihedral transform."""
    return dihedral_transform(arr, DIHEDRAL_INVERSE[tid])
