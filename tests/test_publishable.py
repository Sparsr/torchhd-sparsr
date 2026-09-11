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
import struct

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


def test_the_metadata_requires_the_torch_minor_it_was_built_against():
    """A wheel is valid only against the PyTorch minor that compiled it, so say so.

    `_C` links libtorch's C++ ABI directly -- `DeviceGuardImplInterface`,
    `Allocator`, `TORCH_LIBRARY_IMPL` -- and PyTorch does not keep that ABI
    stable across minor versions. A range like `torch>=2.0` therefore promises
    compatibility the binary does not have: pip installs the wheel happily and
    the import crashes.

    The requirement is one minor version wide, `torch>=2.14,<2.15`, and not the
    exact `torch==2.14.0` that 0.2.0 declared. Patch releases keep the ABI, and
    the exact pin refused `2.14.1` and conflicted with anything else in the same
    environment that pinned torch.

    Wheel tags cannot express this. They carry the Python version, the ABI and
    the platform, and nothing about a dependency -- so two wheels for two torch
    minors cannot coexist under one package version, because pip has no way to
    choose between them. One package release per torch minor is the only
    mechanism there is.

    The requirement is derived from the torch that built the artifact rather
    than written by hand, the same rule the manylinux tag follows.
    """
    import torch

    metadata = _metadata_file().read_text(encoding="utf-8", errors="replace")
    requirements = re.findall(r"^Requires-Dist:\s*(.+)$", metadata, re.MULTILINE)
    torch_pins = [r for r in requirements if re.match(r"^torch\s*(?:[=<>!~]|$)", r)]

    assert torch_pins, f"the wheel declares no torch requirement at all: {requirements}"

    # Compare specifier sets, not strings: setuptools writes the clauses in its own order.
    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet

    major, minor = (int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
    expected = SpecifierSet(f">={major}.{minor},<{major}.{minor + 1}")
    assert len(torch_pins) == 1 and Requirement(torch_pins[0]).specifier == expected, (
        f"expected the torch minor that built this wheel (torch{expected}), found {torch_pins}. "
        f"Wider is a promise the compiled extension cannot keep; an exact pin refuses patch "
        f"releases that keep the ABI."
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
    assert shipped == ["LICENSE", "LICENSE-RUNTIME", "THIRD-PARTY-NOTICES"], f"licence files inside the wheel: {shipped}"

    runtime_terms = (licences_dir / "LICENSE-RUNTIME").read_text(encoding="utf-8")
    for lib in ("libsparsr_host.so", "libsparsr_vm.so", "libsparsr_vmproc.so", "libsparsr_softemu.so"):
        assert lib in runtime_terms, f"{lib} is bundled but LICENSE-RUNTIME does not name it"
    assert "libsparsr_hdc.so" in runtime_terms, (
        "LICENSE-RUNTIME should say in so many words that the HDC library is not covered by it"
    )


def test_the_wheel_carries_the_dotnet_runtime_notice():
    """``libsparsr_vm.so`` is a Native AOT binary, so the .NET runtime is inside it.

    Native AOT links the runtime into the binary rather than loading it at start,
    and the runtime is MIT: its one condition is that the copyright notice and the
    permission notice be included with every copy. The 0.2.0 wheel had neither.
    ``THIRD-PARTY-NOTICES`` is the .NET Foundation's MIT notice plus the notices of
    the components inside the runtime, as the ILCompiler package distributes them,
    and ``LICENSE-RUNTIME`` says where to find it.
    """
    licences_dir = _metadata_file().parent / "licenses"
    notices = (licences_dir / "THIRD-PARTY-NOTICES").read_text(encoding="utf-8")
    assert ".NET Foundation and Contributors" in notices, "the .NET runtime's MIT notice is missing"
    assert "License notice for" in notices, "the .NET runtime's own third-party notices are missing"
    assert "libsparsr_vm.so" in notices, "the notice file should say which binary it is for"

    runtime_terms = (licences_dir / "LICENSE-RUNTIME").read_text(encoding="utf-8")
    assert "THIRD-PARTY-NOTICES" in runtime_terms, "LICENSE-RUNTIME should point at the notices"


# --- the shape of the binaries ---------------------------------------------------------
#
# Read with the standard library rather than `readelf`, so the check runs wherever the
# tests run, including a container that has no binutils. A shared library is a 64-bit
# little-endian ELF here, and its section headers and dynamic section are a few structs.


def _elf_sections(data: bytes) -> dict[str, tuple[int, int]]:
    """Section name -> (file offset, size) for a 64-bit little-endian ELF."""
    assert data[:4] == b"\x7fELF" and data[4] == 2 and data[5] == 1, "not a 64-bit little-endian ELF"
    (e_shoff,) = struct.unpack_from("<Q", data, 0x28)
    e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", data, 0x3A)

    def header(index: int) -> tuple:
        # sh_name, sh_type, sh_flags, sh_addr, sh_offset, sh_size, sh_link, sh_info, sh_addralign, sh_entsize
        return struct.unpack_from("<IIQQQQIIQQ", data, e_shoff + index * e_shentsize)

    names_offset = header(e_shstrndx)[4]
    sections: dict[str, tuple[int, int]] = {}
    for index in range(e_shnum):
        sh_name, _, _, _, sh_offset, sh_size, *_ = header(index)
        end = data.index(b"\0", names_offset + sh_name)
        sections[data[names_offset + sh_name : end].decode("ascii")] = (sh_offset, sh_size)
    return sections


def _elf_runpaths(data: bytes) -> list[str]:
    """Every DT_RPATH and DT_RUNPATH string in the dynamic section."""
    sections = _elf_sections(data)
    dynamic_offset, dynamic_size = sections[".dynamic"]
    strtab_offset, _ = sections[".dynstr"]
    found: list[str] = []
    for offset in range(dynamic_offset, dynamic_offset + dynamic_size, 16):
        tag, value = struct.unpack_from("<qQ", data, offset)
        if tag == 0:  # DT_NULL
            break
        if tag in (15, 29):  # DT_RPATH, DT_RUNPATH
            end = data.index(b"\0", strtab_offset + value)
            found.append(data[strtab_offset + value : end].decode("ascii"))
    return found


def _shipped_binaries() -> list[pathlib.Path]:
    return [path for path in _shipped_files() if path.suffix == ".so"]


def test_the_package_ships_the_binaries_these_checks_expect():
    """Six ELF files: the extension and the five runtime libraries. Fewer means the
    checks below stopped checking something."""
    assert len(_shipped_binaries()) == 6, [p.name for p in _shipped_binaries()]


@pytest.mark.parametrize("path", _shipped_binaries(), ids=_display_name)
def test_every_binary_is_stripped(path: pathlib.Path):
    """No debug information in anything the wheel carries.

    Four of the bundled libraries are proprietary, and ``LICENSE-RUNTIME`` says
    nobody may reverse engineer them. The 0.2.0 wheel carried three of them with
    full DWARF: source file names, line tables, the location of every local
    variable. The extension is the other reason: 12.8 MB of a 13.1 MB file was
    ``.debug_*`` sections. ``setup.py`` strips every one of them, and this is the
    check that it did.
    """
    sections = _elf_sections(path.read_bytes())
    debug = sorted(name for name in sections if name.startswith(".debug"))
    assert not debug, f"{_display_name(path)} carries debug sections: {debug}"
    assert ".symtab" not in sections, f"{_display_name(path)} still has its symbol table"


@pytest.mark.parametrize("path", _shipped_binaries(), ids=_display_name)
def test_no_binary_searches_an_absolute_runpath(path: pathlib.Path):
    """Every library finds its dependencies beside itself, and nowhere else.

    An absolute path in a RUNPATH is a path from the machine that built the
    wheel. It is printed by ``readelf`` to anyone who asks, so it leaks the build
    layout, and the loader searches it on every machine that imports the package,
    so a directory of that name on a user's machine would be consulted first. The
    0.2.0 wheel's ``libsparsr_hdc.so`` carried ``/repo/software/build/sdk-root/lib``
    from the build container.

    One entry is tolerated, on the extension alone: the building Python's own
    library directory. Some Python builds (GitHub's hosted toolcache is one)
    put ``-Wl,-rpath,<LIBDIR>`` in the flags every extension is linked with, so
    a test build on such a machine carries it whatever ``setup.py`` does. The
    manylinux build that produces the published wheel does not, and
    ``packaging/verify_wheel.sh`` checks that artifact with no tolerance at all.
    """
    import sysconfig

    tolerated = {sysconfig.get_config_var("LIBDIR")} if path.name.startswith("_C.") else set()
    for runpath in _elf_runpaths(path.read_bytes()):
        for entry in runpath.split(":"):
            assert entry.startswith("$ORIGIN") or entry in tolerated, (
                f"{_display_name(path)} searches {entry!r} (RUNPATH {runpath!r})"
            )


def test_the_package_is_typed():
    """PEP 561: ``py.typed`` says the annotations are meant to be read, and a
    stub declares the compiled extension, which has none of its own. Without the
    marker a type checker ignores every annotation in the package; with the
    marker and no stub it finds ``_C`` and cannot read it."""
    assert (PACKAGE_DIR / "py.typed").is_file(), "py.typed is not in the wheel"
    stub = PACKAGE_DIR / "_C.pyi"
    assert stub.is_file(), "_C.pyi is not in the wheel"

    declared = stub.read_text(encoding="utf-8")
    for name in ("init_native", "bundle_kernel", "similarity", "LIL_MAX_NONZERO_CHUNKS", "LIL_CHUNK_COUNT"):
        assert name in declared, f"_C.pyi does not declare {name}"
        assert hasattr(torchhd_sparsr._C, name), f"_C.pyi declares {name}, which the extension does not export"

    metadata = _metadata_file().read_text(encoding="utf-8", errors="replace")
    assert "Classifier: Typing :: Typed" in metadata


def test_the_project_page_links_somewhere():
    """The 0.2.0 page on PyPI had no links, no classifiers and no author, so it
    pointed a reader nowhere and appeared under no classifier search."""
    metadata = _metadata_file().read_text(encoding="utf-8", errors="replace")
    urls = dict(re.findall(r"^Project-URL:\s*([^,]+),\s*(\S+)$", metadata, re.MULTILINE))
    for label in ("Homepage", "Documentation", "Source", "Issues"):
        assert label in urls, f"the project page has no {label} link: {urls}"
    assert urls["Source"].startswith("https://github.com/"), urls["Source"]

    classifiers = re.findall(r"^Classifier:\s*(.+)$", metadata, re.MULTILINE)
    assert any(c.startswith("Programming Language :: Python :: 3.") for c in classifiers), classifiers
    assert "Operating System :: POSIX :: Linux" in classifiers, classifiers
    assert not any(c.startswith("License ::") for c in classifiers), (
        "a License classifier beside License-Expression is what PEP 639 retires"
    )
    assert re.search(r"^Author:\s*Sparsr$", metadata, re.MULTILINE), "the page should say who publishes it"
