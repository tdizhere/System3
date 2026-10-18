from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TokenChunkDataset(Dataset):
    """Fixed-length next-token chunks read on demand from a uint16 token shard.

    Item i is (x, y) with x = tokens[start : start+seq_len] and
    y = tokens[start+1 : start+seq_len+1]. So position t of y is the token that
    follows position t of x, which matches how System3Model lines its logits up
    with `labels` (it does not shift internally).
    """

    def __init__(self, path, seq_len, stride=None, offset=0, length=None):
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.stride = stride or seq_len
        self.offset = offset
        available = len(self.tokens) - offset
        self.length = available if length is None else min(length, available)
        self.n_chunks = max(0, (self.length - self.seq_len - 1) // self.stride + 1)

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        start = self.offset + idx * self.stride
        chunk = np.asarray(self.tokens[start : start + self.seq_len + 1], dtype=np.int64)
        return torch.from_numpy(chunk[:-1]), torch.from_numpy(chunk[1:])


def build_dataloaders(cfg):
    """Build (train_loader, val_loader) from a uint16 shard described by `cfg`.

    Expects `cfg` to expose data_path, seq_len, batch_size, val_fraction and device.
    """
    n_tokens = len(np.memmap(cfg.data_path, dtype=np.uint16, mode="r"))
    n_val = int(n_tokens * cfg.val_fraction)
    n_train = n_tokens - n_val

    train_set = TokenChunkDataset(cfg.data_path, cfg.seq_len, offset=0, length=n_train)
    val_set = TokenChunkDataset(cfg.data_path, cfg.seq_len, offset=n_train, length=n_val + 1)

    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
        pin_memory=cfg.device.startswith("cuda"),
    )
    # The held-out tail is small, so validate with smaller batched passes over all of it.
    val_loader = DataLoader(val_set, batch_size=min(32, cfg.batch_size), shuffle=False)

    print(
        f"{n_tokens:,} tokens -> {len(train_set):,} train chunks ({len(train_loader)} batches/epoch)"
        f" + {len(val_set):,} val chunks ({len(val_loader)} batches)"
    )
    return train_loader, val_loader
