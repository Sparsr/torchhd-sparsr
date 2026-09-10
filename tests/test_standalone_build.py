"""The package builds from its own sources plus an unpacked Sparsr HDC tarball, and nothing else.

A clone of this repository has the Python sources, the C++ extension source, and a
downloaded HDC tarball. That tarball already holds everything the build needs: the five native libraries the wheel
bundles and ``sparsr_hdc.h`` that ``_C`` compiles against. So ``SPARSR_HDC_ROOT`` naming
an unpacked HDC tarball is the whole standalone requirement.

The test simulates the clone: it copies the package's sources into a temporary directory,
so nothing outside them can be found by accident, and runs ``pip wheel`` there with
``SPARSR_HDC_ROOT`` set. It is skipped when that variable is not set, because building
needs the tarball, and says so, because a skip that reads as a pass is the failure mode
this suite most needs to avoid.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys
import zipfile

import pytest

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent

# What the public repository will contain. Not packaging/, not dist/, not build output.
SOURCE_TREE = ["pyproject.toml", "setup.py", "README.md", "LICENSE", "LICENSE-RUNTIME", "src", "csrc"]

BUNDLED_LIBRARIES = {
    "libsparsr_hdc.so",
    "libsparsr_host.so",
    "libsparsr_vm.so",
    "libsparsr_vmproc.so",
    "libsparsr_softemu.so",
}


@pytest.mark.skipif(not os.environ.get("SPARSR_HDC_ROOT"), reason="SPARSR_HDC_ROOT is not set: no unpacked HDC tarball to build against")
def test_the_package_builds_from_its_own_sources_and_an_hdc_tarball(tmp_path: pathlib.Path):
    hdc_root = pathlib.Path(os.environ["SPARSR_HDC_ROOT"]).resolve()
    assert (hdc_root / "include" / "sparsr_hdc.h").is_file(), f"{hdc_root} is not an unpacked HDC tarball"

    clone = tmp_path / "torchhd-sparsr"
    clone.mkdir()
    for entry in SOURCE_TREE:
        source = PACKAGE_ROOT / entry
        if source.is_dir():
            shutil.copytree(source, clone / entry, ignore=shutil.ignore_patterns("_native", "__pycache__", "*.so", "*.egg-info"))
        else:
            shutil.copy2(source, clone / entry)

    wheelhouse = tmp_path / "wheelhouse"
    env = {**os.environ, "SPARSR_HDC_ROOT": str(hdc_root)}
    result = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "-w", str(wheelhouse), str(clone)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    print(result.stdout)
    print(result.stderr, file=sys.stderr)
    assert result.returncode == 0, f"pip wheel failed from the copied sources:\n{result.stderr[-3000:]}"

    wheels = list(wheelhouse.glob("torchhd_sparsr-*.whl"))
    assert len(wheels) == 1, f"expected one wheel, found {wheels}"

    with zipfile.ZipFile(wheels[0]) as wheel:
        names = set(wheel.namelist())
    bundled = {pathlib.Path(n).name for n in names if "/_native/" in n and n.endswith(".so")}
    assert bundled == BUNDLED_LIBRARIES, f"the wheel bundles {sorted(bundled)}, expected {sorted(BUNDLED_LIBRARIES)}"
    assert any(n.startswith("torchhd_sparsr/_C.") for n in names), "the compiled extension is missing from the wheel"


def test_the_build_refuses_without_an_hdc_tarball_and_names_the_variable(tmp_path: pathlib.Path):
    """With no tarball named, the build says so and stops.

    There is one way to build, and the message names the variable and the download.
    """
    clone = tmp_path / "torchhd-sparsr"
    clone.mkdir()
    for entry in SOURCE_TREE:
        source = PACKAGE_ROOT / entry
        if source.is_dir():
            shutil.copytree(source, clone / entry, ignore=shutil.ignore_patterns("_native", "__pycache__", "*.so", "*.egg-info"))
        else:
            shutil.copy2(source, clone / entry)

    env = {k: v for k, v in os.environ.items() if k != "SPARSR_HDC_ROOT"}
    result = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "-w", str(tmp_path / "wheelhouse"), str(clone)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0, "a build with no HDC tarball named must not succeed"
    assert "SPARSR_HDC_ROOT" in result.stdout + result.stderr, "the failure must name the variable to set"
