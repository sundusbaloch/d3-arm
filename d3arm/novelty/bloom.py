"""A Bloom filter over 64-bit hash pairs, vectorised with numpy.

Spark computes two independent 64-bit hashes per item (xxhash64 with two
different seeds), and we use double hashing for the k positions:
    position_i = (h1 + i * h2) mod m,   i = 0..k-1
Sizing for n items and false-positive rate p:
    m = -n ln p / (ln 2)^2 bits,   k = (m / n) ln 2
Expected false-positive rate after inserting n items: (1 - e^(-k n / m))^k
"""
import math

import numpy as np


def bloom_size(n, p):
    m = math.ceil(-n * math.log(p) / math.log(2) ** 2)
    k = max(1, round(m / n * math.log(2)))
    return m, k


def expected_fp(n, m, k):
    return (1 - math.exp(-k * n / m)) ** k


class Bloom:
    def __init__(self, m, k):
        self.m, self.k = m, k
        self.bits = np.zeros(m, dtype=bool)          # 1 byte per bit for speed; the real size is m/8 bytes
        self.inserted = 0

    @staticmethod
    def _u64(h):
        h = np.atleast_1d(np.asarray(h))
        return h if h.dtype == np.uint64 else h.astype(np.int64).view(np.uint64)   # Spark gives signed longs

    def _positions(self, h1, h2):
        h1 = self._u64(h1)
        h2 = self._u64(h2) | np.uint64(1)                            # odd step, so the k positions differ
        i = np.arange(self.k, dtype=np.uint64)[:, None]
        with np.errstate(over="ignore"):                             # wrap-around is intended
            return ((h1[None, :] + i * h2[None, :]) % np.uint64(self.m)).astype(np.int64)

    def contains(self, h1, h2):
        return self.bits[self._positions(h1, h2)].all(axis=0)

    def add(self, h1, h2):
        self.bits[self._positions(h1, h2).ravel()] = True
        self.inserted += len(np.atleast_1d(h1))

    @property
    def size_bytes(self):
        return math.ceil(self.m / 8)
