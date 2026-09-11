# torchhd-sparsr

A [Torchhd](https://github.com/hyperdimensional-computing/torchhd) backend that runs Vector Symbolic
Architecture / Hyperdimensional Computing operations on the Sparsr processor instead of the host CPU.

Importing the package registers `"sparsr"` as a real PyTorch device (via
[`PrivateUse1`](https://pytorch.org/tutorials/advanced/privateuseone.html)) and loads the bundled
Sparsr host runtime automatically. From there, standard Torchhd code runs on Sparsr the same way it would
run on CUDA -- no `torchhd_sparsr`-specific API, just `.to("sparsr")`:

```python
import torch
import torchhd
import torchhd_sparsr

a = torchhd.random(1, 4096, vsa="BSC", sparsity=0.998).squeeze().to("sparsr")
b = torchhd.random(1, 4096, vsa="BSC", sparsity=0.998).squeeze().to("sparsr")

result = torchhd.bind(a, b)              # Sparsr's WXOR instruction
similarity = torchhd.cosine_similarity(a, b)  # intersection on Sparsr, counting on the host
bundled = torchhd.bundle(a, b)           # raises: see "Why bundle() refuses" below
```

By default this targets the **Sparsr VM** (`SPARSR_BACKEND=vm`), the software device model published
as `libsparsr_vm.so`. Set `SPARSR_BACKEND` to run against real Sparsr FPGA hardware instead -- the
host library's own header, `sparsr.h`, lists the supported backend names.

**Not `softemu`, and the difference is not cosmetic.** The two backends execute different
instruction sets: softemu runs MIPS words, the VM runs RV32I, and the kernels behind these operations
are RV32I. The same image on softemu decodes to something else entirely, so the package selects `vm`
at import time rather than accepting the loader's default. It only fills the variable in when it is
unset, so an explicit choice still wins.

## What this package computes: nothing

Every HDC operation here is a call into `libsparsr_hdc`, the Sparsr HDC/VSA library, which is
bundled in this package. That library owns the algorithms and the device kernels that run them;
this package owns the PyTorch side alone -- registering the device, converting tensors, and
dispatching.

It used to own kernels too, a bind and a bundle written independently of the library's. Two
implementations of one operation drift apart, and the first sign of it would have been `torchhd` and
`libsparsr_hdc` disagreeing about what `bind` means. So the dependency runs one way: torchhd is a
helper library *for* HDC, and the HDC operations are the HDC library's.

The mapping is direct in every case but one:

| This package | libsparsr_hdc |
| --- | --- |
| `torchhd.bind()` | `hdc_bind()` -- one `WXOR` |
| `torchhd.bundle()` | `hdc_bundle_majority()` over three members |
| `dot_similarity()` / `cosine_similarity()` | `hdc_similarity()` -- one `WAND`, then counts |

The bundle is the one worth explaining. torchhd defines the BSC bundle as
`where(a == b, a, tiebreak)`, and that is exactly a majority vote over the three of them: where `a`
and `b` agree they outvote the tiebreak two to one, and where they disagree the tiebreak decides. So
it needs no kernel of its own. A test in the HDC library's own suite pins that identity, because the
library's threshold could change to "at least half" and still pass every other majority test it
has -- while silently turning this package's bundle into a union.

## What's supported

| Torchhd call             | Runs on Sparsr hardware?                                                   |
|---------------------------|-----------------------------------------------------------------------------|
| `torchhd.bind()`          | Yes -- a single `WXOR` instruction, via genuine `logical_xor` op dispatch. |
| `torchhd.bundle()`        | No -- raises `RuntimeError` whenever the two operands disagree anywhere, which is every real pair. Sparsr cannot store the fair-coin tiebreak torchhd's semantics need; see below. |
| `torchhd.dot_similarity()` / `cosine_similarity()` | Partially -- the intersection is a `WAND` on Sparsr; counting its bits runs on the host, inside `libsparsr_hdc` (see below). Only single pairs, not batches of stored class vectors. |
| `torchhd.permute()`       | No -- raises `NotImplementedError`. Needs a wide bit-rotate instruction Sparsr doesn't have yet. |
| `torchhd.multiset()` / `multibundle()` | No -- needs a batch of hypervectors, and the `"sparsr"` device only supports single hypervectors (see below), so it can never be reached on this device. The device-side half is no longer the blocker: `libsparsr_hdc` bundles any number of members today, with a bit-plane carry-save adder. |

Everything else (arbitrary tensor ops, printing/repr, arithmetic) isn't implemented for the `"sparsr"`
device -- move a tensor back with `.to("cpu")` first.

## What fits in CMEM, and what does not

Sparsr's CMEM always transfers data through its native LIL-32b sparse compression codec. That codec
splits each 512-byte (4096-bit) block into 128 four-byte lanes and stores one entry per **non-zero
lane** -- a one-byte lane index plus the whole four-byte lane value -- with room for 48 entries in a
240-byte row. Moving a tensor to `"sparsr"` checks that and raises a clear `RuntimeError` rather than
letting data corrupt silently.

**The ceiling counts non-zero lanes, not set bits.** A lane is stored whole, so once a lane is occupied
the bits inside it are free. That gives two quite different ways to fit, and only one of them is
"sparse":

- **Bits spread across all 4096 positions** have to be genuinely sparse. Every set bit tends to occupy
  a fresh lane, so the practical ceiling is around 1.5% density -- e.g.
  `torchhd.random(..., vsa="BSC", sparsity=0.998)`. This is the Sparse Distributed Representation (SDR)
  style of VSA that Sparsr's compression hardware is built for, and it is what this package's own
  examples and tests use.
- **Bits confined to 48 of the 128 lanes** can be at any density at all, including 50%. That is a
  fully dense hypervector 1536 bits wide, and it always fits.

So a full-width dense 4096-bit BSC hypervector does not fit -- essentially all 128 lanes are occupied --
but a dense 1536-bit one does. Measured on MNIST, that narrower dense code is worth about 65% accuracy
against about 80% at the full 4096 bits, so the ceiling costs real accuracy rather than blocking dense
codes outright. An uncompressed CMEM path would lift it; it is planned but not built.

## Why `bundle()` refuses

torchhd's BSC `bundle(a, b)` is `where(a == b, a, tiebreak)`: keep the shared value wherever the two
hypervectors agree, and resolve every position where they disagree with a **fair** coin flip. That
tiebreak vector is dense by construction, and by the section above dense data cannot be stored in CMEM
at all.

Sparsr therefore has no way to compute a faithful bundle today. Drawing the tiebreak at the low density
CMEM *can* store makes the coin overwhelmingly biased towards 0, so every disagreeing position resolves
to 0 and the bundle of two sparse hypervectors comes back as the **all-zero hypervector** -- a
valid-looking tensor carrying no information at all. Measured at `sparsity=0.998`:

| Tiebreak | set bits in `a` | set bits in `b` | agreeing set bits | disagreeing positions | bundled set bits |
| --- | --- | --- | --- | --- | --- |
| CPU, fair coin | 10 | 7 | 0 | 17 | 12 |
| Sparsr, sparse coin | 10 | 7 | 0 | 17 | **0** |

So `bundle()` raises a `RuntimeError` naming the limit it hit, instead of returning that.
Two operands that agree everywhere consult no tiebreak, so that case is exact and still runs on Sparsr.
Bundle on the host (`.to("cpu")`) in the meantime.

## Where a tensor's bits live

In host memory. A `"sparsr"` tensor holds its 4096 bits on the host, and each operation sends its
operands to the device and reads the result back.

They used to be resident: a tensor's storage pointer encoded a CMEM row, and the bits stayed on the
device between operations. That could not survive moving onto `libsparsr_hdc`, and it should not have.
The library reserves CMEM rows 0 to 31 -- every row there is -- from `hdc_init()` onwards, and nothing
on a Sparsr device arbitrates who owns a row. Residency here meant two libraries writing the same rows
with no error on either side, which is the collision the HDC library's own device memory layout
names as the one it cannot prevent. One
owner of CMEM is the only arrangement that works today.

What it costs is a host round trip per operation, which matters for chained work: `bind()` then
`bundle()` no longer keeps the intermediate on the device. A host-side memory manager would let
residency come back, for both libraries at once; it is planned but not built.

Independently of that, `.to("sparsr")` supports exactly one 4096-bit hypervector per tensor -- a batch
has to be moved one at a time. Giving CMEM real capacity is planned but not built.

## Installing

```sh
pip install torchhd-sparsr
```

That is the whole install step. The wheel already contains:

- the Sparsr host runtime,
- the Sparsr VM, which executes the operations,
- the HDC library and its device kernels,
- the compiled PyTorch extension that registers the `"sparsr"` device.

Operations run on the software emulator by default, so Sparsr hardware is optional. You do not
need a RISC-V toolchain, a compiler, or any other SDK. A wheel is published per CPython version,
for Linux on x86-64.

**Each wheel is built against one PyTorch minor version** and uses libtorch's C++ ABI directly, the
same constraint torchvision has. Install the wheel that matches the PyTorch you run.

There is no source distribution on PyPI, because a build needs the Sparsr runtime as binaries. To
build the package yourself, unpack the Sparsr HDC tarball from the Developer Zone and point the build
at it. It holds the runtime and `libsparsr_hdc`, which is all the build takes from outside:

```sh
SPARSR_HDC_ROOT=/path/to/sparsr-hdc pip wheel --no-deps --no-build-isolation .
```

That needs a C++ compiler and the PyTorch you intend to run against, and nothing else.

## Licence

The sources of this package are MIT, and so is `libsparsr_hdc`, the library every operation calls
into. The wheel also bundles four proprietary binaries, the Sparsr host runtime and the device
model: `libsparsr_host.so`, `libsparsr_vm.so`, `libsparsr_vmproc.so` and `libsparsr_softemu.so`.
Their terms are in `LICENSE-RUNTIME`, which is packaged inside the wheel beside `LICENSE`. The
package metadata says the same thing in one line: `MIT AND LicenseRef-Proprietary`.
