"""Nothing internal to Sparsr may ship inside the published package.

This package is meant to be installed with `pip install torchhd-sparsr` by
people outside the company. Everything in the wheel therefore reaches a
customer: the Python sources verbatim, the compiled `_C` extension's string
literals, and the bundled native libraries. A ticket key, a link to a private
repository, or an internal file path in any of them tells an outside reader
about our tracker and our source layout, and helps them not at all -- "see
SPR-nnn" points at a tracker they cannot open.

The wheel is the unit under test, not the source tree. Comments in `csrc/`
and in `tests/` are internal and stay internal, because neither directory
ships: the wheel carries `src/torchhd_sparsr` only. What `csrc/` contributes
to the wheel is the compiled `_C` extension, so its *string literals* are in
scope here while its comments are not -- which is why this scans binaries
rather than reading the C++ source.

The rules below are the two a reader needs no context to understand. The
fuller policy, which also covers names and source paths, runs over this tree
from the build that publishes the wheel, so nothing is lost by keeping this
copy short.
"""

from __future__ import annotations

import pathlib
import re

import pytest

import torchhd_sparsr

PACKAGE_DIR = pathlib.Path(torchhd_sparsr.__file__).parent

# `scan_binaries` says whether a rule is also applied to the compiled
# extension's printable strings. Both of these are: an issue key or a hostname
# does not occur inside unrelated byte sequences by accident.
RULES: list[tuple[str, re.Pattern[str], bool]] = [
    ("issue key", re.compile(r"\bSPR-\d+\b"), True),
    # docs.sparsr.com, developers.sparsr.com and www.sparsr.com are the public
    # sites and pointing at them is the correct thing to do. Any other
    # subdomain is somewhere a reader cannot go.
    ("non-public site link", re.compile(r"\b(?!docs\.|developers\.|www\.)[a-z0-9-]+\.sparsr\.com\b"), True),
]

# Printable runs of 6+ characters, matching what `strings` reports. Comparing
# against these rather than the whole blob stops a pattern matching across
# unrelated data and keeps the reported text readable.
_PRINTABLE_RUN = re.compile(rb"[\x20-\x7e]{6,}")


def _is_binary(data: bytes) -> bool:
    return b"\x00" in data[:8192]


def _display_name(path: pathlib.Path) -> str:
    """Name a scanned file relative to site-packages, so the package's own
    files and the sibling dist-info both read sensibly in a failure."""
    return str(path.relative_to(PACKAGE_DIR.parent))


def _violations(path: pathlib.Path) -> list[str]:
    data = path.read_bytes()
    relative = _display_name(path)
    found: list[str] = []

    if _is_binary(data):
        haystack = [run.decode("ascii") for run in _PRINTABLE_RUN.findall(data)]
        for name, pattern, scan_binaries in RULES:
            if not scan_binaries:
                continue
            for run in haystack:
                for match in pattern.findall(run):
                    found.append(f"{relative}: {name} -- {match!r} (inside a binary)")
    else:
        for number, line in enumerate(data.decode("utf-8", "replace").splitlines(), start=1):
            for name, pattern, _ in RULES:
                for match in pattern.findall(line):
                    found.append(f"{relative}:{number}: {name} -- {match!r}")

    return found


def _metadata_file() -> pathlib.Path:
    """The wheel's METADATA, which carries README.md as the long description.

    That text becomes the project page on PyPI, so it reaches more people than
    any file in the package does -- a reader sees it before deciding whether to
    install anything. Scanning it here rather than scanning README.md directly
    is deliberate: it checks what was actually packaged, so renaming the readme
    or changing which file `pyproject.toml` points at cannot quietly drop the
    check.
    """
    candidates = sorted(PACKAGE_DIR.parent.glob("torchhd_sparsr-*.dist-info/METADATA"))
    assert candidates, f"no installed dist-info found beside {PACKAGE_DIR}"
    return candidates[-1]


def _shipped_files() -> list[pathlib.Path]:
    package_files = [
        path
        for path in PACKAGE_DIR.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    ]
    return sorted(package_files) + [_metadata_file()]


def test_the_package_ships_something_to_scan():
    """A scan that silently found no files would pass forever. The wheel
    always carries the Python sources, the `_C` extension and the bundled
    native libraries, so anything less means this test stopped testing."""
    shipped = _shipped_files()
    names = {path.name for path in shipped}

    assert len(shipped) >= 6, f"only found {len(shipped)} files under {PACKAGE_DIR}"
    assert "__init__.py" in names
    assert any(name.startswith("_C.") and name.endswith(".so") for name in names)
    assert "libsparsr_hdc.so" in names
    assert "libsparsr_vm.so" in names


def test_the_project_page_has_no_relative_links():
    """A relative link works in the repository and nowhere else.

    The long description is rendered on a project page that has no repository
    around it, so `[sparsr.h](../../runtime/loader/include/sparsr.h)` renders as
    a dead link *and* publishes our internal source layout in its link text.
    Every link on that page has to be absolute, or be no link at all.
    """
    metadata = _metadata_file().read_text(encoding="utf-8", errors="replace")

    # Markdown inline links whose target is neither absolute nor a fragment.
    relative = re.findall(r"\[[^\]]*\]\((?!https?://|mailto:|#)([^)]+)\)", metadata)

    assert not relative, (
        "the project page carries "
        f"{len(relative)} relative link(s), which resolve nowhere once published:\n  "
        + "\n  ".join(sorted(set(relative)))
        + "\n\nLink to a public URL, or name the component in plain text instead."
    )


@pytest.mark.parametrize("path", _shipped_files(), ids=_display_name)
def test_no_internal_references_ship(path: pathlib.Path):
    found = _violations(path)
    assert not found, (
        f"{len(found)} internal reference(s) would ship to customers:\n  "
        + "\n  ".join(found)
        + "\n\nIssue keys, internal links, personal names and internal file paths belong in "
        "the repository and the tracker, not in a file a customer downloads. Say what the "
        "reader needs in order to use the package, and keep the history internal."
    )


def test_the_metadata_pins_the_torch_it_was_built_against():
    """A wheel is valid only against the PyTorch that compiled it, so say so.

    `_C` links libtorch's C++ ABI directly -- `DeviceGuardImplInterface`,
    `Allocator`, `TORCH_LIBRARY_IMPL` -- and PyTorch does not keep that ABI
    stable across minor versions. A range like `torch>=2.0` therefore promises
    compatibility the binary does not have: pip installs the wheel happily and
    the import crashes. torchvision has the identical constraint and answers it
    the identical way, pinning `torch==2.14.0` for 0.29.0 and one torch release
    per torchvision release.

    Wheel tags cannot express this. They carry the Python version, the ABI and
    the platform, and nothing about a dependency -- so two wheels for two torch
    versions cannot coexist under one package version, because pip has no way to
    choose between them. One package release per torch release is the only
    mechanism there is.

    The pin is derived from the torch that built the artifact rather than
    written by hand, the same rule the manylinux tag follows.
    """
    import torch

    metadata = _metadata_file().read_text(encoding="utf-8", errors="replace")
    requirements = re.findall(r"^Requires-Dist:\s*(.+)$", metadata, re.MULTILINE)
    torch_pins = [r for r in requirements if re.match(r"^torch\s*(?:[=<>!~]|$)", r)]

    assert torch_pins, f"the wheel declares no torch requirement at all: {requirements}"

    expected = torch.__version__.split("+")[0]
    assert torch_pins == [f"torch=={expected}"], (
        f"expected an exact pin on the torch that built this wheel "
        f"(torch=={expected}), found {torch_pins}. A range here is a promise "
        f"the compiled extension cannot keep."
    )


def test_the_wheel_declares_its_split_licence():
    """MIT for the sources, proprietary for the bundled runtime, both files in the wheel.

    The wheel bundles five native libraries. One, ``libsparsr_hdc.so``, is MIT
    like the package. The other four are the Sparsr host runtime and device
    model, and they are proprietary. A bare ``License: MIT`` over that mix would
    grant redistribution and reverse-engineering rights the runtime does not
    come with, so the metadata carries a licence expression that says both, and
    the two licence files travel inside the wheel where a reader of ``dist-info``
    can find them.
    """
    metadata = _metadata_file().read_text(encoding="utf-8", errors="replace")
    expression = re.findall(r"^License-Expression:\s*(.+)$", metadata, re.MULTILINE)
    assert expression == ["MIT AND LicenseRef-Proprietary"], (
        f"expected a split licence expression, found {expression or 'none'}: "
        f"a bare MIT over proprietary binaries is a grant nobody decided to make"
    )
    assert not re.search(r"^License:\s", metadata, re.MULTILINE), (
        "the legacy License field must not sit beside the expression and contradict it"
    )

    licences_dir = _metadata_file().parent / "licenses"
    shipped = sorted(p.name for p in licences_dir.iterdir()) if licences_dir.is_dir() else []
    assert shipped == ["LICENSE", "LICENSE-RUNTIME"], f"licence files inside the wheel: {shipped}"

    runtime_terms = (licences_dir / "LICENSE-RUNTIME").read_text(encoding="utf-8")
    for lib in ("libsparsr_host.so", "libsparsr_vm.so", "libsparsr_vmproc.so", "libsparsr_softemu.so"):
        assert lib in runtime_terms, f"{lib} is bundled but LICENSE-RUNTIME does not name it"
    assert "libsparsr_hdc.so" in runtime_terms, (
        "LICENSE-RUNTIME should say in so many words that the HDC library is not covered by it"
    )
