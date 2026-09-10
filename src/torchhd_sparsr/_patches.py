"""Sparsr-aware overrides for the handful of Torchhd BSC operations Sparsr
hardware can support (bundle, similarity), and a clear error for the one it
can't yet (permute).

`bind()` needs none of this: it's `self.logical_xor(other)`, a single real
aten op, so registering a PrivateUse1 kernel for `logical_xor` (see
csrc/sparsr_backend.cpp) is enough for it to dispatch to Sparsr hardware
transparently. `bundle()`/`permute()`/`*_similarity()` are each composed of
several aten ops in torchhd (`eq`, `where`, `bernoulli_`, `roll`, `matmul`,
...) that don't individually map onto Sparsr's bitwise instructions -- most
fundamentally, `eq`'s complement is dense and dense data can't survive
CMEM's sparse compression codec (see csrc/sparsr_backend.cpp's
check_fits_device) -- so those are patched at the Python method level
instead of the aten level.

`multibundle()` (N-way majority vote; needs the wide population-count/
threshold-accumulate unit Sparsr hardware doesn't have yet) isn't patched at
all: it needs a batch dimension, and the "sparsr" device
only supports single hypervectors (see csrc/sparsr_backend.cpp) -- there's no
way for a batched tensor to reach this device in the first place, so a device
check here would be unreachable code.

`bundle()` is the operation this module has to refuse, and why is worth
stating in full. torchhd's BSC bundle is
`where(a == b, a, tiebreak)` with `tiebreak` drawn from a fair coin. A fair
coin flip over 4096 bits is dense, and dense data does not fit CMEM, so the
tiebreak below is drawn at `_TIEBREAK_SPARSITY` instead. That makes the coin
enormously biased towards 0, so every position where the two operands
disagree is resolved to 0 and the bundle of two sparse hypervectors comes
back as the all-zero hypervector -- a valid-looking tensor carrying no
information. `_patched_bundle` therefore refuses whenever the operands
disagree anywhere, rather than returning that. An uncompressed CMEM path
would lift the density ceiling and make a fair tiebreak storable; it is
planned but not built.
"""

from __future__ import annotations

import torch
from torchhd.tensors.bsc import BSCTensor

from . import _C

# Density for bundle()'s random tiebreak coin flips. Must be low enough that
# the tiebreak tensor reliably fits Sparsr's CMEM LIL-32b compression codec;
# torchhd's own default (dense, sparsity=0.5) tiebreak cannot be represented
# on Sparsr hardware at all. See check_fits_device in
# csrc/sparsr_backend.cpp for the exact limit this is chosen to comfortably
# clear.
#
# Being this far from a fair coin is exactly why _patched_bundle refuses any
# pair that disagrees anywhere. The value still matters for the pair that is
# allowed through -- operands that agree everywhere -- where the two operands
# outvote the tiebreak at every position, so it only has to be storable, not
# fair.
_TIEBREAK_SPARSITY = 0.998

_ORIGINAL_BUNDLE = BSCTensor.bundle
_ORIGINAL_PERMUTE = BSCTensor.permute
_ORIGINAL_DOT_SIMILARITY = BSCTensor.dot_similarity
_ORIGINAL_COSINE_SIMILARITY = BSCTensor.cosine_similarity

_PERMUTE_HARDWARE_GAP = (
    "torchhd_sparsr: permute() needs a wide bit-rotate instruction Sparsr "
    "hardware doesn't have yet (WAND/WOR/WXOR/WL/WS can't express a "
    "rotation). This is a known hardware limit, not a bug in your code: move "
    "to CPU first (`.to('cpu')`). A wide bit-rotate instruction that would "
    "close this gap is planned but not built."
)

_BATCHED_SIMILARITY_HARDWARE_GAP = (
    "torchhd_sparsr: dot_similarity() against a batch of stored hypervectors "
    "isn't implemented yet; compare one pair at a time. Doing it on-device "
    "needs the wide population-count / threshold-accumulate unit Sparsr "
    "hardware doesn't have yet."
)


def _bundle_tiebreak_gap(disagreements: int, dimensions: int) -> str:
    return (
        "torchhd_sparsr: bundle() cannot be computed correctly on the 'sparsr' "
        "device for these two hypervectors. torchhd resolves every position "
        "where the operands disagree with a fair coin flip, and these disagree "
        f"in {disagreements} of {dimensions} positions. A fair coin flip is "
        "dense, and dense data does not fit Sparsr's CMEM: the LIL-32b codec "
        f"has room for only {_C.LIL_MAX_NONZERO_CHUNKS} of "
        f"{_C.LIL_CHUNK_COUNT} 32-bit chunks. Sparsr therefore draws its "
        "tiebreak at the low density CMEM can store, which leaves essentially "
        "every disagreeing position at 0 -- so the result would come back "
        "biased towards the all-zero hypervector instead of the bundle you "
        "asked for. This is a known hardware limit, not a bug in your code: "
        "bundle on the host instead (`.to('cpu')`). An uncompressed CMEM path "
        "that would lift this density ceiling is planned but not built."
    )


def _on_sparsr(*tensors: torch.Tensor) -> bool:
    return all(t.device.type == "sparsr" for t in tensors)


def _patched_bundle(self: BSCTensor, other: BSCTensor, *, generator: torch.Generator = None) -> BSCTensor:
    if not _on_sparsr(self, other):
        return _ORIGINAL_BUNDLE(self, other, generator=generator)

    # Refuse before computing anything: the tiebreak below is too sparse to
    # resolve a disagreement, so any disagreeing position would silently come
    # back as 0. Agreeing operands consult no tiebreak at all and are exact,
    # so they are still computed on-device.
    disagreements = _hamming_distance(self, other)
    if disagreements > 0:
        raise RuntimeError(_bundle_tiebreak_gap(disagreements, self.shape[-1]))

    tiebreaker = torch.empty(self.shape[-1], dtype=torch.bool)
    tiebreaker.bernoulli_(1.0 - _TIEBREAK_SPARSITY, generator=generator)

    result = _C.bundle_kernel(self, other, tiebreaker.to(self.device))
    return result.as_subclass(type(self))


def _patched_permute(self: BSCTensor, shifts: int = 1) -> BSCTensor:
    if self.device.type == "sparsr":
        raise NotImplementedError(_PERMUTE_HARDWARE_GAP)
    return _ORIGINAL_PERMUTE(self, shifts)


def _hamming_distance(a: torch.Tensor, b: torch.Tensor) -> int:
    """How many positions the two hypervectors differ in -- the Hamming distance.

    libsparsr_hdc reports an intersection rather than a difference: the overlap
    and each operand's own weight. The Hamming distance follows from those three
    counts, since a position differs exactly when it is set in one operand and
    not in the intersection: |a| + |b| - 2 * |a AND b|.

    The intersection and its count are one wide instruction on the device.
    Asking libsparsr_hdc for the counts rather than computing them here is what
    let that land for every caller of the library at once, which is the point of
    the layering: this file did not change when it did.

    The two operand weights are host counts of host-owned data, on purpose --
    see hdc_similarity() in libsparsr_hdc.
    """
    overlap, left_weight, right_weight = _C.similarity(a, b)
    return left_weight + right_weight - 2 * overlap


def _patched_dot_similarity(self: BSCTensor, others: BSCTensor, *, dtype=None) -> torch.Tensor:
    if not _on_sparsr(self, others):
        return _ORIGINAL_DOT_SIMILARITY(self, others, dtype=dtype)
    if others.dim() >= 2:
        raise NotImplementedError(_BATCHED_SIMILARITY_HARDWARE_GAP)

    if dtype is None:
        dtype = torch.get_default_dtype()

    # Bipolar dot product of two {-1,+1}-mapped hypervectors equals
    # dimension - 2 * hamming_distance(a, b).
    dimension = self.shape[-1]
    return torch.tensor(dimension - 2 * _hamming_distance(self, others), dtype=dtype)


def _patched_cosine_similarity(self: BSCTensor, others: BSCTensor, *, dtype=None) -> torch.Tensor:
    if not _on_sparsr(self, others):
        return _ORIGINAL_COSINE_SIMILARITY(self, others, dtype=dtype)
    dimension = self.shape[-1]
    return _patched_dot_similarity(self, others, dtype=dtype) / dimension


def install() -> None:
    BSCTensor.bundle = _patched_bundle
    BSCTensor.permute = _patched_permute
    BSCTensor.dot_similarity = _patched_dot_similarity
    BSCTensor.cosine_similarity = _patched_cosine_similarity
