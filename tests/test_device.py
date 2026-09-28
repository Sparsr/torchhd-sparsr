import warnings

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


def test_manual_seed_does_not_warn_about_the_sparsr_device():
    """torch.manual_seed() seeds every registered device module, and it warns
    about one that does not offer the two methods it looks for. The warning is
    noise here, because nothing on the device draws random numbers: every
    hypervector is generated on the host and sent over. It showed up in this
    package's own test output, and in any user script that seeds torch.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.manual_seed(1234)

    about_sparsr = [str(warning.message) for warning in caught if "sparsr" in str(warning.message)]
    assert not about_sparsr, about_sparsr


def test_the_device_module_reports_a_deterministic_fork_state():
    """`torch.manual_seed` asks the device module whether the process is in a
    bad fork before seeding it. There is no device state to lose across a
    fork, so the answer is always no."""
    assert torch.sparsr._is_in_bad_fork() is False
