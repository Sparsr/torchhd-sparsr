import pytest
import torch


def test_to_sparsr_and_back_round_trips(random_hypervector):
    a = random_hypervector()
    a_sparsr = a.to("sparsr")
    assert a_sparsr.device.type == "sparsr"
    assert torch.equal(a, a_sparsr.to("cpu"))


def test_to_sparsr_preserves_bsc_tensor_type(random_hypervector):
    a = random_hypervector()
    a_sparsr = a.to("sparsr")
    assert isinstance(a_sparsr, type(a))
    assert isinstance(a_sparsr.to("cpu"), type(a))


def test_to_sparsr_rejects_non_4096_dimension(random_hypervector):
    a = random_hypervector(dimensions=1024)
    with pytest.raises(RuntimeError):
        a.to("sparsr")


def test_to_sparsr_rejects_batched_tensors():
    import torchhd

    batch = torchhd.random(3, 4096, vsa="BSC", sparsity=0.998)
    with pytest.raises(RuntimeError):
        batch.to("sparsr")


def test_to_sparsr_rejects_dense_hypervectors():
    import torchhd

    torch.manual_seed(0)
    dense = torchhd.random(1, 4096, vsa="BSC", sparsity=0.5).squeeze()
    with pytest.raises(RuntimeError, match="too dense"):
        dense.to("sparsr")


def test_to_sparsr_rejects_non_bool_dtype():
    with pytest.raises(RuntimeError):
        torch.zeros(4096, dtype=torch.float32).to("sparsr")
