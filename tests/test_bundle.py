import torch
import torchhd
from torchhd.tensors.bsc import BSCTensor


def _hypervector_from_positions(positions, dimensions=4096):
    v = torch.zeros(dimensions, dtype=torch.bool)
    for p in positions:
        v[p] = True
    return v.as_subclass(BSCTensor)


def test_bundle_agreement_and_disagreement_resolution():
    """torchhd's own bundle() is where(a == b, a, tiebreak). Sparsr computes it
    as a majority vote over the two operands and the tiebreak, which is the same
    function: where a and b agree they outvote the tiebreak two to one, and where
    they disagree the tiebreak decides. That is libsparsr_hdc's
    hdc_bundle_majority() over three members -- see csrc/sparsr_backend.cpp.
    Exercise both branches with hand-picked, non-overlapping bit positions so the
    test is exact rather than depending on a random pair happening to
    agree/disagree."""
    a = _hypervector_from_positions([0, 1, 10, 21])
    b = _hypervector_from_positions([0, 1, 11, 20])
    tiebreak_positions = {10, 11, 99}  # 99 is irrelevant (not a disagreement position)

    a_sparsr = a.to("sparsr")
    b_sparsr = b.to("sparsr")

    # Bypass the random tiebreak generation to get a fully deterministic
    # result: patch torch.empty(...).bernoulli_ isn't practical here, so
    # instead verify the *pure* device-resident kernel via the exposed hook.
    from torchhd_sparsr import _C

    tiebreak = _hypervector_from_positions(tiebreak_positions).to("sparsr")
    result = _C.bundle_kernel(a_sparsr, b_sparsr, tiebreak).to("cpu")

    expected = _hypervector_from_positions({0, 1, 10, 11})
    assert torch.equal(result, expected)


def test_bundle_kernel_preserves_agreement(random_pair):
    """Property that must hold regardless of the tiebreak: wherever a and b
    agree, the bundled result must equal them.

    This goes through the raw device kernel rather than `torchhd.bundle()`,
    because the latter now refuses any pair that disagrees anywhere (see
    test_fail_loudly.py). Note also that asserting this through
    `torchhd.bundle()` used to pass *vacuously*: two sparse hypervectors agree
    almost everywhere by both being 0 there, and the wrong all-zero result
    satisfied the assertion perfectly.

    The tiebreak here is a fair coin masked down to the disagreeing positions,
    which is semantically identical to torchhd's unmasked fair coin (the
    kernel ANDs it with `a XOR b` anyway) but sparse enough to store.
    """
    a, b = random_pair()
    a_sparsr, b_sparsr = a.to("sparsr"), b.to("sparsr")

    disagreements = a ^ b
    tiebreak = torch.empty(a.shape[-1], dtype=torch.bool).bernoulli_(0.5) & disagreements

    from torchhd_sparsr import _C

    result = _C.bundle_kernel(a_sparsr, b_sparsr, tiebreak.to("sparsr")).to("cpu")

    agree_mask = a == b
    assert torch.equal(result[agree_mask], a[agree_mask])
    assert torch.equal(result[disagreements], tiebreak[disagreements])


def test_bundle_preserves_bsc_tensor_type(random_hypervector):
    """Bundling a hypervector with itself consults no tiebreak, so it is the
    one pair `torchhd.bundle()` still computes on-device."""
    a = random_hypervector()
    a_sparsr = a.to("sparsr")
    result = torchhd.bundle(a_sparsr, a_sparsr)
    assert isinstance(result, type(a))
    assert result.device.type == "sparsr"
