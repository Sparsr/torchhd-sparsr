"""Every path torchhd-sparsr cannot compute correctly must raise, not return a
plausible-looking wrong answer.

The dangerous one is `bundle()`. torchhd's BSC bundle resolves each position
where the two operands disagree with a fair coin flip; a fair coin flip is
dense, and dense data cannot be stored in Sparsr's CMEM (the LIL-32b codec has
room for 48 of 128 32-bit chunks), so torchhd_sparsr draws its tiebreak at the
low density CMEM can store. That leaves essentially every disagreeing position
at 0, and the bundle of two sparse hypervectors comes back as the *zero*
hypervector -- a valid-looking tensor carrying no information at all.

Measured on this repo before the fix, at torchhd's own `sparsity=0.998`:

    |a|=10  |b|=7  |a AND b|=0  |a XOR b|=17   CPU bundle: 12 set bits
                                               Sparsr bundle: 0 set bits

The remaining gaps (permute, batched similarity search, multibundle) already
fail loudly; what their messages lacked was a pointer to the hardware ticket
that would close them, so a researcher can tell a known limit from a bug in
their own code.
"""

import pytest
import torch
import torchhd


def test_bundle_raises_rather_than_returning_zeros(random_pair):
    """Two independently drawn sparse hypervectors disagree in ~|a|+|b|
    positions, none of which Sparsr's sparse tiebreak can resolve."""
    a, b = random_pair()
    with pytest.raises(RuntimeError):
        torchhd.bundle(a.to("sparsr"), b.to("sparsr"))


def test_bundle_never_returns_a_zero_hypervector(random_pair):
    """The exact regression, stated as the property that must hold forever --
    including after the uncompressed CMEM path lifts the density ceiling and bundle() starts
    succeeding: whatever bundle() does, it must never hand back a tensor with
    no set bits when its operands had some."""
    raised = 0
    for seed in range(20):
        torch.manual_seed(seed)
        a, b = random_pair()
        a_sparsr, b_sparsr = a.to("sparsr"), b.to("sparsr")
        try:
            result = torchhd.bundle(a_sparsr, b_sparsr).to("cpu")
        except RuntimeError:
            raised += 1
            continue
        assert result.sum() > 0, f"seed {seed}: bundle() returned an all-zero hypervector"

    # Today every one of these pairs disagrees somewhere, so every one must
    # refuse. Asserting that keeps this test from passing vacuously if the
    # bundle path ever stops being exercised at all.
    assert raised == 20


def test_bundle_error_names_the_limit_and_the_way_out(random_pair):
    a, b = random_pair()
    with pytest.raises(RuntimeError) as excinfo:
        torchhd.bundle(a.to("sparsr"), b.to("sparsr"))

    message = str(excinfo.value)
    assert "48" in message, "the error must name the LIL-32b 48-chunk ceiling it hit"
    assert "cpu" in message, "the error must name the workaround the caller can apply today"


def test_bundle_of_identical_hypervectors_still_works(random_hypervector):
    """No disagreeing positions means no tiebreak is ever consulted, so this
    case is computed exactly and must not be swept up by the check above."""
    a = random_hypervector()
    a_sparsr = a.to("sparsr")

    result = torchhd.bundle(a_sparsr, a_sparsr).to("cpu")

    assert torch.equal(result, a)


def test_permute_error_names_the_missing_instruction(random_hypervector):
    a = random_hypervector().to("sparsr")
    with pytest.raises(NotImplementedError, match="bit-rotate"):
        torchhd.permute(a)


def test_hardware_gap_messages_name_the_missing_hardware():
    """The batched-similarity gap can't be reached through a real call today --
    a batched tensor cannot live on the "sparsr" device at all, so
    `.to("sparsr")` rejects it first (see
    test_device.py::test_to_sparsr_rejects_batched_tensors). Its message still
    has to name the hardware that would close it, so assert on the message
    constant directly rather than writing an unreachable call."""
    from torchhd_sparsr import _patches

    assert "bit-rotate" in _patches._PERMUTE_HARDWARE_GAP
    assert "population-count" in _patches._BATCHED_SIMILARITY_HARDWARE_GAP


def test_dense_rejection_names_the_density_limit():
    torch.manual_seed(0)
    dense = torchhd.random(1, 4096, vsa="BSC", sparsity=0.5).squeeze()
    with pytest.raises(RuntimeError, match="too dense for Sparsr's CMEM"):
        dense.to("sparsr")
