"""Build script for the torchhd-sparsr package.

The package bundles the Sparsr runtime and the Sparsr HDC library as binaries and
compiles one PyTorch C++ extension, `torchhd_sparsr._C`, against the HDC library's
header. All of that input comes from one place: an unpacked Sparsr HDC tarball,
named by SPARSR_HDC_ROOT. The tarball is the free download from the Sparsr
Developer Zone, and it holds exactly what this build needs:

    include/sparsr_hdc.h        the header _C compiles against
    lib/libsparsr_hdc.so        the HDC library
    lib/libsparsr_host.so       the Sparsr runtime, and the three backends
    lib/libsparsr_vm.so         libsparsr_host.so records as DT_NEEDED entries
    lib/libsparsr_vmproc.so
    lib/libsparsr_softemu.so

Nothing else is needed: no RISC-V toolchain, no other SDK, no other checkout. With
SPARSR_HDC_ROOT unset the build stops and says so.
"""

import os
import pathlib
import shutil
import subprocess

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

PACKAGE_ROOT = pathlib.Path(__file__).parent.resolve()
NATIVE_DIR = PACKAGE_ROOT / "src" / "torchhd_sparsr" / "_native"

# libsparsr_host.so links against the first three, so all of them have to travel with it.
NATIVE_LIBS = (
    "libsparsr_softemu.so",
    # The out-of-process backend, reached with SPARSR_BACKEND=vmproc. It has to ship
    # whether or not anything here uses it: libsparsr_host.so records it as a DT_NEEDED
    # entry, so a wheel without it fails to import at all rather than merely missing one
    # backend. It costs the wheel about 20 KB and imposes nothing -- it is plain C and
    # links against libc alone.
    "libsparsr_vmproc.so",
    "libsparsr_vm.so",
    "libsparsr_host.so",
    # The HDC/VSA library: every operation this package performs is a call into it, and
    # `_C` records it as a DT_NEEDED entry, so the same rule applies.
    "libsparsr_hdc.so",
)

def _hdc_root() -> pathlib.Path:
    """The unpacked HDC tarball, or an error that says what to set and where to get it."""
    configured = os.environ.get("SPARSR_HDC_ROOT")
    if not configured:
        raise RuntimeError(
            "SPARSR_HDC_ROOT is not set. torchhd-sparsr builds against an unpacked Sparsr HDC "
            "tarball: download it from the Sparsr Developer Zone, unpack it, and set "
            "SPARSR_HDC_ROOT to that directory."
        )
    root = pathlib.Path(configured).resolve()
    missing = [name for name in NATIVE_LIBS if not (root / "lib" / name).exists()]
    if not (root / "include" / "sparsr_hdc.h").exists():
        missing.append("include/sparsr_hdc.h")
    if missing:
        raise RuntimeError(
            f"SPARSR_HDC_ROOT={root} does not look like an unpacked Sparsr HDC tarball: "
            f"missing {', '.join(missing)}."
        )
    return root


def _strip(binary: pathlib.Path) -> None:
    """Remove the symbol table and every debug section from a binary the wheel carries.

    Four of the bundled libraries are proprietary, and their licence says nobody may
    reverse engineer them. The published 0.2.0 wheel carried three of them with full DWARF
    debug information: source file names, line tables, the location of every local
    variable. That hands over most of what disassembly would cost. The compiled extension
    is the other reason: it came out at 13.1 MB, of which 12.8 MB was `.debug_*` sections.

    `strip` is required rather than optional on purpose. A build that quietly skipped it
    would produce a wheel that looks right and is wrong in exactly the way the last one
    was, so a machine without binutils fails here and says so. Every machine that can
    compile the extension has binutils.
    """
    strip = shutil.which("strip")
    if strip is None:
        raise RuntimeError(
            f"`strip` is not on PATH, so {binary.name} cannot be stripped. torchhd-sparsr "
            "refuses to build a wheel that carries debug information in the bundled Sparsr "
            "runtime. Install binutils and build again."
        )
    subprocess.run([strip, str(binary)], check=True)


def _stage_native(root: pathlib.Path) -> None:
    """Copy the five libraries into the package so the wheel carries them, stripped."""
    if NATIVE_DIR.exists():
        shutil.rmtree(NATIVE_DIR)
    NATIVE_DIR.mkdir(parents=True)
    for lib_name in NATIVE_LIBS:
        staged = NATIVE_DIR / lib_name
        shutil.copy2(root / "lib" / lib_name, staged)
        _strip(staged)


# The libraries must be in place before the C++ extension below is compiled (it links
# against libsparsr_hdc.so), so this runs at setup.py import time rather than in a
# build_ext hook, which guarantees it happens before any setuptools command that needs it.
_HDC_ROOT = _hdc_root()
_stage_native(_HDC_ROOT)

def _torch_requirement() -> str:
    """The PyTorch minor version that compiled this wheel, as a range over its patch releases.

    `_C` uses libtorch's C++ ABI directly, and PyTorch does not keep that ABI
    stable between minor versions, so a wheel works against the torch minor
    that built it and no other. A range from 2.0 upwards would let pip install
    it against any torch and crash on import.

    The range is one minor version wide, `torch>=2.14,<2.15`, rather than the
    exact `torch==2.14.0` that 0.2.0 declared. PyTorch's patch releases are bug
    fixes on the same ABI, and an exact pin cost more than it protected: it
    refused `2.14.1`, and it conflicted outright with anything else in the same
    environment that pinned torch. The lower bound is the minor, not the exact
    build version, for the same reason.

    Building against an older torch does not widen the window, because the break
    is not one-directional: an extension compiled against 2.0 fails to load on
    2.14 just as the reverse does. It only moves the single supported minor
    somewhere less useful.

    Nor can wheel tags help. A tag carries the Python version, the ABI and the
    platform -- nothing about a dependency -- so two wheels for two torch
    minors cannot coexist under one package version. Supporting another torch
    means another release of this package, which is what torchvision does too.

    Derived from the torch that built the artifact rather than written by hand,
    the same rule the manylinux tag follows: hard-coding it would let a change
    to the build quietly promise compatibility the binary no longer has.
    """
    major, minor = (int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
    return f"torch>={major}.{minor},<{major}.{minor + 1}"


class _StrippedBuildExtension(BuildExtension):
    """torch's build_ext, plus a strip of every extension it produces.

    The staged runtime libraries are stripped as they are copied in; the extension
    does not exist until here, so this is where it gets the same treatment.
    """

    def build_extension(self, ext) -> None:
        super().build_extension(ext)
        _strip(pathlib.Path(self.get_ext_fullpath(ext.name)))


ext_modules = [
    CppExtension(
        name="torchhd_sparsr._C",
        sources=["csrc/sparsr_backend.cpp"],
        include_dirs=[str(_HDC_ROOT / "include")],
        library_dirs=[str(NATIVE_DIR)],
        libraries=["sparsr_hdc"],
        extra_link_args=["-Wl,-rpath,$ORIGIN/_native"],
    )
]

setup(
    install_requires=[_torch_requirement(), "torch-hd>=5.0", "numpy"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": _StrippedBuildExtension},
)
