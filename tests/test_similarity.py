import torch
import torchhd


def test_dot_similarity_matches_cpu(random_pair):
    a, b = random_pair()
    expected = torchhd.dot_similarity(a, b)
    result = torchhd.dot_similarity(a.to("sparsr"), b.to("sparsr"))
    assert result.item() == expected.item()


def test_cosine_similarity_matches_cpu(random_pair):
    a, b = random_pair()
    expected = torchhd.cosine_similarity(a, b)
    result = torchhd.cosine_similarity(a.to("sparsr"), b.to("sparsr"))
    assert abs(result.item() - expected.item()) < 1e-6


def test_dot_similarity_self_is_dimension(random_hypervector):
    """A hypervector's bipolar dot product with itself is exactly D (every
    bit agrees), regardless of density."""
    a = random_hypervector()
    result = torchhd.dot_similarity(a.to("sparsr"), a.to("sparsr"))
    assert result.item() == a.shape[-1]


def test_similarity_matches_cpu_across_many_random_pairs():
    mismatches = 0
    trials = 20
    for seed in range(trials):
        torch.manual_seed(seed)
        a = torchhd.random(1, 4096, vsa="BSC", sparsity=0.998).squeeze()
        b = torchhd.random(1, 4096, vsa="BSC", sparsity=0.998).squeeze()
        expected = torchhd.dot_similarity(a, b).item()
        result = torchhd.dot_similarity(a.to("sparsr"), b.to("sparsr")).item()
        if expected != result:
            mismatches += 1
    assert mismatches == 0
