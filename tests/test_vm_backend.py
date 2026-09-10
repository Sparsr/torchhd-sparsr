"""The package runs its operations on the Sparsr VM, and says so.

The loader picks a backend from `SPARSR_BACKEND` and defaults to `softemu`. That default is
wrong for this package: libsparsr_hdc's kernels are RV32I and softemu executes MIPS words,
so the same image means two different things and only one of them is this package's.

That failure used to be silent -- softemu would read the image, execute whatever the words
happen to decode to, and return an untouched result row. It is loud now: a similarity is
a population-count reduce, and `hdc_init()` probes the device with a known
pair rather than assuming it can run one. A backend that cannot answer is refused at import
instead of answering every similarity with whatever data memory held.

So these tests pin three things: that the package selects the VM, that it still lets a
caller choose a different Sparsr device, and that choosing one which cannot run the kernels
fails where it can be seen.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import torchhd_sparsr  # noqa: F401  (importing is what sets the backend)

REPO_PYTHON = sys.executable


def test_package_selects_the_vm_backend_by_default() -> None:
    assert os.environ["SPARSR_BACKEND"] == "vm"


# Imports the package under whatever backend the caller asked for, and reports both what
# the variable ended up holding and how the import went. The two are printed rather than
# returned because this has to happen in a fresh interpreter: the backend is resolved once,
# on first use, so a process that has already imported the package cannot re-choose.
# `!r` on the error keeps the outcome to one line whatever the message contains, which is
# what lets the two values be read off the end of stdout. A backend that cannot serve a call
# is chatty on the way down -- the loader announces each fall-back and softemu narrates every
# instruction -- so the last two lines are taken rather than the only two.
_REPORT_BACKEND_AND_OUTCOME = (
    "import os\n"
    "outcome = 'initialised'\n"
    "try:\n"
    "    import torchhd_sparsr\n"
    "except RuntimeError as error:\n"
    "    outcome = f'refused: {error!r}'\n"
    "print(os.environ['SPARSR_BACKEND'])\n"
    "print(outcome)\n"
)


def _import_under_backend(backend: str) -> tuple[str, str, str]:
    result = subprocess.run(
        [REPO_PYTHON, "-c", _REPORT_BACKEND_AND_OUTCOME],
        env={**os.environ, "SPARSR_BACKEND": backend},
        capture_output=True,
        text=True,
    )
    lines = result.stdout.strip().splitlines()
    assert len(lines) >= 2, f"stdout was {result.stdout!r}, stderr {result.stderr!r}"
    return lines[-2], lines[-1], result.stderr


def test_an_explicit_backend_choice_is_left_alone() -> None:
    """`.to("sparsr")` has to keep meaning "whichever Sparsr device you asked for".

    The choice is made before the device is touched, so it survives whether or not the
    device turns out to be able to run the kernels. That is the point: the package must not
    quietly substitute a backend it prefers.
    """
    selected, _outcome, stderr = _import_under_backend("fpgasim")
    assert selected == "fpgasim", stderr


def test_a_backend_that_cannot_run_the_kernels_is_refused() -> None:
    """The other half of leaving the choice alone: saying so when the choice cannot work.

    `fpgasim` falls back to softemu for the calls it does not provide, and softemu executes
    MIPS words -- so libsparsr_hdc's RV32I kernels decode to something else entirely and its
    population-count reduce never runs. `hdc_init()` probes for exactly that and raises here,
    where previously the import succeeded and every similarity afterwards reported
    whatever data memory happened to hold.
    """
    _selected, outcome, stderr = _import_under_backend("fpgasim")
    assert outcome.startswith("refused: "), f"outcome was {outcome!r}, stderr {stderr!r}"
    assert "HDC_ERROR_DEVICE" in outcome, outcome


def test_the_bundled_native_libraries_are_present() -> None:
    """These are build artefacts, so a stale checkout should fail here, not at run time.

    `libsparsr_hdc.so` is where every HDC operation and every device kernel now lives --
    this package carries none of its own. It used to assert a `torchhd_ops.spex`
    image beside it, which was the second, independent copy of bind and bundle that the
    rewrite deleted.
    """
    native_dir = pathlib.Path(torchhd_sparsr.__file__).parent / "_native"
    assert (native_dir / "libsparsr_hdc.so").is_file()
    assert (native_dir / "libsparsr_vm.so").is_file()
