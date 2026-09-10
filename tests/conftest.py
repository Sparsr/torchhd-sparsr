import pytest
import torch
import torchhd

import torchhd_sparsr  # noqa: F401  (registers the "sparsr" device)

# Dense (sparsity=0.5, torchhd's default) hypervectors overflow Sparsr's CMEM
# LIL-32b compression codec, so tests use a low density that reliably fits --
# see torchhd_sparsr/csrc/sparsr_backend.cpp's check_fits_lil_compression.
SPARSITY = 0.998


@pytest.fixture
def random_pair():
    def _make(dimensions=4096, sparsity=SPARSITY):
        a = torchhd.random(1, dimensions, vsa="BSC", sparsity=sparsity).squeeze()
        b = torchhd.random(1, dimensions, vsa="BSC", sparsity=sparsity).squeeze()
        return a, b

    return _make


@pytest.fixture
def random_hypervector():
    def _make(dimensions=4096, sparsity=SPARSITY):
        return torchhd.random(1, dimensions, vsa="BSC", sparsity=sparsity).squeeze()

    return _make
