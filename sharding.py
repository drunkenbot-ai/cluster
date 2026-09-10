"""Data sharding and deterministic dataset slicing for distributed cluster workers."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def compute_shard_boundaries(
    total_tokens: int,
    shard_index: int,
    total_shards: int,
    context_length: int = 512,
) -> tuple[int, int]:
    """Compute aligned non-overlapping token slice boundaries for a worker shard.

    Ensures boundaries are multiples of ``context_length`` to prevent fractional windows.

    Args:
        total_tokens: Total token count in train_tokens.npy.
        shard_index: 0-indexed worker slot.
        total_shards: Total active worker count.
        context_length: Transformer context window size.

    Returns:
        (start_token_idx, end_token_idx).
    """
    if total_shards <= 1:
        return 0, total_tokens

    # Total usable complete windows
    total_windows = total_tokens // context_length
    windows_per_shard = total_windows // total_shards

    if windows_per_shard == 0:
        # Dataset is smaller than total_shards * context_length
        return 0, total_tokens

    start_token = shard_index * windows_per_shard * context_length
    if shard_index == total_shards - 1:
        end_token = total_tokens  # Last worker covers remaining tokens
    else:
        end_token = (shard_index + 1) * windows_per_shard * context_length

    return start_token, end_token


class ShardedTokenDataset(Dataset):
    """Memory-mapped token dataset that slices a disjoint partition for a worker node."""

    def __init__(
        self,
        token_array: np.ndarray,
        context_length: int,
        shard_index: int = 0,
        total_shards: int = 1,
        stride: Optional[int] = None,
        target_array: Optional[np.ndarray] = None,
        vocab_size: Optional[int] = None,
    ) -> None:
        """Create a sharded token dataset.

        Args:
            token_array: Memory-mapped or loaded token array.
            context_length: Model context length.
            shard_index: Current worker shard index (0 to total_shards - 1).
            total_shards: Total number of participating workers.
            stride: Token step stride between windows. Defaults to context_length.
            target_array: Optional target token array for loss masking.
            vocab_size: Optional maximum vocabulary size to clamp out-of-range token IDs.
        """
        self.context_length = context_length
        self.stride = stride or context_length
        self.has_targets = target_array is not None
        self.vocab_size = vocab_size

        # Compute worker slice boundaries
        total_tokens = len(token_array)
        start_idx, end_idx = compute_shard_boundaries(
            total_tokens, shard_index, total_shards, context_length=context_length
        )

        self.tokens = token_array[start_idx:end_idx]
        self.targets = target_array[start_idx:end_idx] if self.has_targets else None

        # Number of samples in this shard
        usable = len(self.tokens) - self.context_length
        self.sample_count = max(0, (usable // self.stride) + 1) if usable >= 0 else 0

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        end = start + self.context_length

        input_chunk = self.tokens[start:end]
        x = torch.from_numpy(input_chunk.astype(np.int64))

        if self.has_targets and self.targets is not None:
            target_chunk = self.targets[start:end]
            y = torch.from_numpy(target_chunk.astype(np.int64))
        else:
            # Autoregressive next-token target: shift right by 1
            if end < len(self.tokens):
                target_chunk = self.tokens[start + 1 : end + 1]
                y = torch.from_numpy(target_chunk.astype(np.int64))
            else:
                y = x.clone()

        if self.vocab_size is not None and self.vocab_size > 0:
            x = torch.clamp(x, 0, self.vocab_size - 1)
            y = torch.clamp(y, 0, self.vocab_size - 1)

        return x, y
