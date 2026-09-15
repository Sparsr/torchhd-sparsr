"""torchhd-sparsr: a Sparsr backend for Torchhd (https://github.com/hyperdimensional-computing/torchhd).

Importing this package registers "sparsr" as a real PyTorch device (backed by
a PrivateUse1 C++ extension), loads the bundled Sparsr host runtime and the
Sparsr HDC/VSA library that runs every operation on the device
(``libsparsr_hdc``), and patches in Sparsr-accelerated implementations
of the Torchhd BSC operations Sparsr's hardware can support. Standard Torchhd
code then runs on Sparsr the same way it would run on CUDA -- no call-site
changes needed beyond `.to("sparsr")`::

    import torch
    import torchhd
    import torchhd_sparsr

    a = torchhd.random(1, 4096, vsa="BSC", sparsity=0.998).squeeze().to("sparsr")
    b = torchhd.random(1, 4096, vsa="BSC", sparsity=0.998).squeeze().to("sparsr")

    result = torchhd.bind(a, b)  # dispatches to Sparsr's WXOR instruction

Hypervectors must be sparse to fit Sparsr's WMEM (see the README); dense
(sparsity=0.5, the Torchhd default) hypervectors will raise a RuntimeError
when moved to "sparsr". Only single hypervectors are supported, not batches
(see the README).

Operations Sparsr cannot compute correctly raise rather than returning a
plausible-looking wrong answer. `bundle()` raises a RuntimeError for any pair
of hypervectors that disagree anywhere: torchhd resolves those positions with
a fair coin flip, and a fair coin flip is too dense for WMEM to store.
`permute()` raises NotImplementedError -- it needs a wide bit-rotate
instruction Sparsr hardware doesn't have yet.
"""

from __future__ import annotations

import os
import pathlib

# libsparsr_hdc's kernels are RV32I, so they run on the Sparsr VM. `softemu`, the loader's
# default, executes MIPS words and would read the same images as something else entirely --
# the two backends are not interchangeable, which is why they have separate names. Setting
# this before `_C` is imported is what makes it take effect: libsparsr_host resolves the
# backend once, on first use.
#
# `setdefault`, not an assignment: `SPARSR_BACKEND=fpgaf2` has to keep working, since the
# whole point of the loader's indirection is that this package is agnostic about which
# Sparsr device it runs against.
os.environ.setdefault("SPARSR_BACKEND", "vm")

import torch  # noqa: E402  (must follow the backend selection above)

from . import _C  # noqa: E402

_NATIVE_DIR = pathlib.Path(__file__).parent / "_native"

# Claims the device for libsparsr_hdc: loads its kernels into instruction memory and
# reserves the WMEM rows they use. Nothing arbitrates those rows, so this package holds
# none of its own -- see the note at the top of csrc/sparsr_backend.cpp.
_C.init_native()
torch.utils.rename_privateuse1_backend("sparsr")


class _SparsrDeviceModule:
    """Minimal backend module torch's device-string plumbing expects to find
    as `torch.sparsr` once the PrivateUse1 backend is renamed."""

    @staticmethod
    def is_available() -> bool:
        return True

    # torch.manual_seed() seeds every registered device module, and warns about
    # one that does not offer these two. Nothing on the device draws random
    # numbers: a hypervector is generated on the host and sent over, so there
    # is no per-device generator to seed and no fork state to lose. These
    # answer the question rather than do anything.

    @staticmethod
    def manual_seed(seed: int) -> None:
        return None

    @staticmethod
    def manual_seed_all(seed: int) -> None:
        return None

    @staticmethod
    def _is_in_bad_fork() -> bool:
        return False


torch._register_device_module("sparsr", _SparsrDeviceModule)
torch.utils.generate_methods_for_privateuse1_backend(for_storage=False)

from . import _patches  # noqa: E402  (must run after rename_privateuse1_backend)

_patches.install()

__all__: list[str] = []
