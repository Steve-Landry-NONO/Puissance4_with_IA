from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class Batch:
    s: np.ndarray       # (B,6,7,2)
    a: np.ndarray       # (B,)
    r: np.ndarray       # (B,)
    s2: np.ndarray      # (B,6,7,2)
    done: np.ndarray    # (B,)
    mask2: np.ndarray   # (B,7) bool


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000) -> None:
        self.capacity = int(capacity)
        self._ptr = 0
        self._size = 0

        self.s = np.zeros((capacity, 6, 7, 2), dtype=np.float32)
        self.a = np.zeros((capacity,), dtype=np.int64)
        self.r = np.zeros((capacity,), dtype=np.float32)
        self.s2 = np.zeros((capacity, 6, 7, 2), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)
        self.mask2 = np.zeros((capacity, 7), dtype=np.bool_)

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        s: np.ndarray,
        a: int,
        r: float,
        s2: np.ndarray,
        done: bool,
        mask2: np.ndarray,
    ) -> None:
        i = self._ptr
        self.s[i] = s
        self.a[i] = a
        self.r[i] = r
        self.s2[i] = s2
        self.done[i] = 1.0 if done else 0.0
        self.mask2[i] = mask2

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int = 64) -> Batch:
        idx = np.random.randint(0, self._size, size=batch_size)
        return Batch(
            s=self.s[idx],
            a=self.a[idx],
            r=self.r[idx],
            s2=self.s2[idx],
            done=self.done[idx],
            mask2=self.mask2[idx],
        )
#buffer

