"""permute() needs a wide instruction Sparsr hardware doesn't have yet (a bit
rotate -- see torchhd_sparsr/_patches.py). It should fail
loudly on the "sparsr" device rather than silently falling back to the CPU or
producing a wrong result.

multibundle() (N-way majority vote) has no equivalent test here: it needs a
batch dimension, and the "sparsr" device only supports single hypervectors,
so a batched tensor can never reach the device in the first place --
test_device.py::test_to_sparsr_rejects_batched_tensors already covers that.
"""

import pytest
import torchhd


def test_permute_raises_on_sparsr_device(random_hypervector):
    a = random_hypervector().to("sparsr")
    with pytest.raises(NotImplementedError):
        torchhd.permute(a)


def test_permute_still_works_on_cpu(random_hypervector):
    a = random_hypervector()
    torchhd.permute(a)  # must not raise
