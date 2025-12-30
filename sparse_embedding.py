"""
Sparse Embedding module for task-specific learned embeddings.

Key features:
- CastedSparseEmbedding: Embeddings that only update accessed rows (sparse updates)
- CastedSparseEmbeddingSignSGD: Optimizer using sign of gradient for sparse embeddings
- Efficient for task-specific learning where only one task's embedding updates per batch
"""

from typing import Union
import math

import torch
from torch import nn
from torch.optim.optimizer import Optimizer


def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0) -> torch.Tensor:
    """Truncated normal initialization (values outside 2*std are resampled)."""
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std)
    return tensor


class CastedSparseEmbedding(nn.Module):
    """
    Sparse embedding layer with efficient gradient handling for task-specific embeddings.

    In training:
    - Copies relevant embedding rows to local_weights (with gradient tracking)
    - Returns casted local_weights
    - Gradients only flow through accessed embeddings

    In eval:
    - Direct lookup from weights (no gradient)

    Args:
        num_embeddings: Number of unique task embeddings
        embedding_dim: Dimension of each embedding
        batch_size: Maximum batch size (for local buffer allocation)
        init_std: Standard deviation for truncated normal initialization (0 = zero init)
        cast_to: Dtype to cast outputs to (e.g., torch.bfloat16)
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        batch_size: int,
        init_std: float,
        cast_to: torch.dtype
    ):
        super().__init__()
        self.cast_to = cast_to
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Main weight buffer - persistent, stores all task embeddings
        if init_std > 0:
            weights = trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std)
        else:
            weights = torch.zeros((num_embeddings, embedding_dim))
        self.weights = nn.Buffer(weights, persistent=True)

        # Local weights buffer - non-persistent, used for gradient computation
        # local_weights has requires_grad=True to capture gradients
        self.local_weights = nn.Buffer(
            torch.zeros(batch_size, embedding_dim, requires_grad=True),
            persistent=False
        )
        # Local IDs buffer - tracks which embeddings are in local_weights
        self.local_ids = nn.Buffer(
            torch.zeros(batch_size, dtype=torch.int32),
            persistent=False
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Look up embeddings for given task IDs.

        Args:
            inputs: [B] tensor of task IDs

        Returns:
            [B, embedding_dim] tensor of task embeddings
        """
        if not self.training:
            # Inference: direct lookup, no gradient
            return self.weights[inputs].to(self.cast_to)

        # Training: copy to local buffer for gradient tracking
        with torch.no_grad():
            # Resize local buffers if needed
            batch_size = inputs.shape[0]
            if batch_size > self.local_weights.shape[0]:
                device = self.local_weights.device
                self.local_weights = nn.Buffer(
                    torch.zeros(batch_size, self.embedding_dim, requires_grad=True, device=device),
                    persistent=False
                )
                self.local_ids = nn.Buffer(
                    torch.zeros(batch_size, dtype=torch.int32, device=device),
                    persistent=False
                )

            # Copy relevant embeddings to local buffer
            self.local_weights[:batch_size].copy_(self.weights[inputs])
            self.local_ids[:batch_size].copy_(inputs)

        return self.local_weights[:batch_size].to(self.cast_to)


class CastedSparseEmbeddingSignSGD(Optimizer):
    """
    SignSGD optimizer for sparse embeddings.

    Uses sign(gradient) for updates, which is equivalent to Adam when gradients
    are very sparse (only a few rows updated per step).

    Includes decoupled weight decay.

    Args:
        params: Parameters from CastedSparseEmbedding.buffers()
        lr: Learning rate
        weight_decay: Decoupled weight decay coefficient
    """

    def __init__(
        self,
        params,
        lr: Union[float, torch.Tensor] = 1e-3,
        weight_decay: float = 1e-2,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, weight_decay=weight_decay)
        # Store buffers directly - don't use parent's param validation
        # because nn.Buffer with requires_grad=True isn't a leaf tensor
        self.defaults = defaults
        self._buffers = list(params)
        self.param_groups = [{"params": self._buffers, **defaults}]
        self.state = {}
        # Initialize hooks that PyTorch expects
        self._optimizer_step_pre_hooks = {}
        self._optimizer_step_post_hooks = {}
        self._optimizer_state_dict_pre_hooks = {}
        self._optimizer_state_dict_post_hooks = {}
        self._optimizer_load_state_dict_pre_hooks = {}
        self._optimizer_load_state_dict_post_hooks = {}

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        for group in self.param_groups:
            # Find the sparse embedding components
            local_weights_grad = None
            local_ids = None
            weights = None

            # Expect 3 buffers: weights, local_weights, local_ids
            for p in group["params"]:
                if p.requires_grad and p.grad is not None:
                    local_weights_grad = p.grad
                elif p.ndim == 1:  # local_ids is 1D
                    local_ids = p
                elif p.ndim == 2 and not p.requires_grad:  # weights is 2D, no grad
                    weights = p

            if local_ids is None or weights is None:
                continue

            # Apply SignSGD with decoupled weight decay
            if local_weights_grad is not None:
                _sparse_emb_signsgd(
                    local_weights_grad,
                    local_ids,
                    weights,
                    lr=group["lr"],
                    weight_decay=group["weight_decay"]
                )


def _sparse_emb_signsgd(
    local_weights_grad: torch.Tensor,
    local_ids: torch.Tensor,
    weights: torch.Tensor,
    lr: float,
    weight_decay: float
) -> None:
    """
    SignSGD update for sparse embeddings.

    1. Gather unique IDs (handles duplicate IDs in batch via scatter_add)
    2. Apply decoupled weight decay
    3. Update using sign(aggregated_gradient)
    """
    N, D = local_weights_grad.shape

    # Get unique IDs and aggregate gradients for duplicates
    # (If same task appears multiple times in batch, sum their gradients)
    grad_ids, inv = local_ids[:N].unique(return_inverse=True)

    # Aggregate gradients by task ID
    grad = torch.zeros((grad_ids.shape[0], D), dtype=local_weights_grad.dtype, device=local_weights_grad.device)
    grad.scatter_add_(0, inv.unsqueeze(-1).expand(-1, D), local_weights_grad)

    # Get current embedding values for updated tasks
    p = weights[grad_ids]

    # Decoupled weight decay + SignSGD update
    # p = p * (1 - lr * wd) - lr * sign(grad)
    p.mul_(1.0 - lr * weight_decay).add_(torch.sign(grad), alpha=-lr)

    # Write updated embeddings back
    weights[grad_ids] = p
