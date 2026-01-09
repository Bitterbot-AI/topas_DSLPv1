from typing import Union, List

import torch
from torch import nn
import torch.distributed as dist

from common import trunc_normal_init_


class CastedSparseEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, batch_size: int, init_std: float, cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to

        # Real Weights
        # Truncated LeCun normal init
        self.weights = nn.Buffer(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std), persistent=True
        )

        # Local weights and IDs
        # Local embeddings, with gradient, not persistent
        self.local_weights = nn.Buffer(torch.zeros(batch_size, embedding_dim, requires_grad=True), persistent=False)
        # Local embedding IDs, not persistent
        self.local_ids = nn.Buffer(torch.zeros(batch_size, dtype=torch.int32), persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.training:
            # Test mode, no gradient
            return self.weights[inputs].to(self.cast_to)

        # Training mode: create fresh tensor that's part of computation graph
        # We gather from weights (detached), but return a grad-enabled tensor
        with torch.no_grad():
            self.local_ids.copy_(inputs)

        # Create a new tensor with gradients enabled in the computation graph
        self.local_weights = self.weights[inputs].clone().requires_grad_(True)
        self.local_weights.retain_grad()  # Retain grad for non-leaf tensor

        return self.local_weights.to(self.cast_to)


class CastedSparseEmbeddingSignSGD_Distributed:
    """
    Custom optimizer for CastedSparseEmbedding using SignSGD.

    This is NOT a standard PyTorch optimizer - it takes the embedding module
    directly and updates weights based on local_weights gradients.

    SignSGD with high learning rate enables instant memorization of puzzle patterns.
    """

    def __init__(
        self,
        embedding_modules: List[CastedSparseEmbedding],
        world_size: int,
        lr: float = 0.01,
        weight_decay: float = 0.01,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        self.embedding_modules = embedding_modules
        self.lr = lr
        self.weight_decay = weight_decay
        self.world_size = world_size

        # PyTorch optimizer compatibility - param_groups for LR scheduling
        self.param_groups = [{'lr': lr, 'weight_decay': weight_decay}]

    def zero_grad(self):
        """Zero gradients on local_weights for all embedding modules."""
        for emb in self.embedding_modules:
            if emb.local_weights.grad is not None:
                emb.local_weights.grad.zero_()

    @torch.no_grad()
    def step(self, closure=None):
        """Update weights based on local_weights gradients using SignSGD."""
        # Get LR from param_groups (allows external LR scheduling)
        lr = self.param_groups[0]['lr']
        wd = self.param_groups[0]['weight_decay']

        for emb in self.embedding_modules:
            local_weights = emb.local_weights
            local_ids = emb.local_ids
            weights = emb.weights

            if local_weights.grad is None:
                continue

            _sparse_emb_signsgd_dist(
                local_weights.grad,
                local_ids,
                weights,
                lr=lr,
                weight_decay=wd,
                world_size=self.world_size
            )

    def state_dict(self):
        """Return optimizer state (just hyperparameters, no momentum state)."""
        return {
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'world_size': self.world_size
        }

    def load_state_dict(self, state_dict):
        """Load optimizer state."""
        self.lr = state_dict.get('lr', self.lr)
        self.weight_decay = state_dict.get('weight_decay', self.weight_decay)
        self.world_size = state_dict.get('world_size', self.world_size)


def _sparse_emb_signsgd_dist(
    local_weights_grad: torch.Tensor,
    local_ids: torch.Tensor,
    weights: torch.Tensor,

    lr: float,
    weight_decay: float,
    world_size: int
) -> None:
    N, D = local_weights_grad.shape

    # All-gather
    all_weights_grad = local_weights_grad
    all_ids = local_ids

    if world_size > 1:
        all_weights_grad = torch.empty((world_size * N, D), dtype=local_weights_grad.dtype, device=local_weights_grad.device)
        all_ids = torch.empty(world_size * N,               dtype=local_ids.dtype,          device=local_ids.device)

        dist.all_gather_into_tensor(all_weights_grad, local_weights_grad)
        dist.all_gather_into_tensor(all_ids,          local_ids)

    # Unique
    grad_ids, inv = all_ids.unique(return_inverse=True)

    grad = torch.zeros((grad_ids.shape[0], D), dtype=all_weights_grad.dtype, device=all_weights_grad.device)
    grad.scatter_add_(0, inv.unsqueeze(-1).expand(-1, D), all_weights_grad)

    # SignSGD with decoupled weight decay
    p = weights[grad_ids]

    p.mul_(1.0 - lr * weight_decay).add_(torch.sign(grad), alpha=-lr)

    # Write updated slices back
    weights[grad_ids] = p
