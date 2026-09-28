"""Declarations for the compiled extension ``torchhd_sparsr._C``.

The extension is built from ``csrc/sparsr_backend.cpp``. Most of what it does is
registered with PyTorch's dispatcher and reached through ordinary tensor
operations on the ``"sparsr"`` device, so it has no Python name. These are the
few things it exports by name, and this file is what lets a type checker see
them: the package carries ``py.typed``, and a compiled module has no
annotations of its own.
"""

import torch

LIL_MAX_NONZERO_CHUNKS: int
"""How many non-zero 32-bit lanes a WMEM row can hold under the LIL-32b codec."""

LIL_CHUNK_COUNT: int
"""How many 32-bit lanes a 4096-bit hypervector has."""

def init_native() -> None:
    """Initialise libsparsr_hdc and load its device kernels. Idempotent."""

def bundle_kernel(a: torch.Tensor, b: torch.Tensor, tiebreak: torch.Tensor) -> torch.Tensor:
    """torchhd's BSC bundle, as a majority vote over the two operands and the tiebreak."""

def similarity(a: torch.Tensor, b: torch.Tensor) -> tuple[int, int, int]:
    """Overlap, left weight and right weight of two ``"sparsr"`` hypervectors."""
