import torch
import torchhd


def test_bind_matches_torchhd_cpu_xor(random_pair):
    a, b = random_pair()
    expected = torchhd.bind(a, b)

    result = torchhd.bind(a.to("sparsr"), b.to("sparsr"))

    assert result.device.type == "sparsr"
    assert torch.equal(expected, result.to("cpu"))


def test_bind_preserves_bsc_tensor_type(random_pair):
    a, b = random_pair()
    result = torchhd.bind(a.to("sparsr"), b.to("sparsr"))
    assert isinstance(result, type(a))


def test_bind_is_self_inverse(random_pair):
    a, b = random_pair()
    a_sparsr, b_sparsr = a.to("sparsr"), b.to("sparsr")

    bound = torchhd.bind(a_sparsr, b_sparsr)
    unbound = torchhd.bind(bound, b_sparsr)

    assert torch.equal(unbound.to("cpu"), a)


def test_bind_matches_cpu_across_many_random_pairs(random_pair):
    """bind() is deterministic (XOR), so this should match bit-for-bit every
    time a pair of hypervectors is sparse enough to fit WMEM."""
    mismatches = 0
    trials = 30
    for seed in range(trials):
        torch.manual_seed(seed)
        a, b = random_pair()
        expected = torchhd.bind(a, b)
        result = torchhd.bind(a.to("sparsr"), b.to("sparsr")).to("cpu")
        if not torch.equal(expected, result):
            mismatches += 1
    assert mismatches == 0
