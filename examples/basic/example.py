import shutil

import torch
import torchhd

import torchhd_sparsr  # noqa: F401  (registers the "sparsr" device)


def active_bit_positions(tensor):
    indices = tensor.squeeze().nonzero().flatten().tolist()
    return f"{len(indices)} active bits at indices {indices}"


def check(name, cpu_result, sparsr_result):
    match = torch.equal(cpu_result, sparsr_result.to("cpu"))
    status = "\u2714\ufe0f " if match else "\u274c"
    print(f"{status} {name}: Sparsr result {'matches' if match else 'DOES NOT MATCH'} the CPU result.")
    if not match:
        raise AssertionError(f"{name}: Sparsr result does not match the CPU result.")


# Create 4096-bit hypervectors. Sparsr's CMEM transfers data through a sparse
# compression codec, so hypervectors need a low density of active bits to fit
# -- see torchhd_sparsr's README for details.
a = torchhd.random(1, 4096, vsa="BSC", sparsity=0.998).squeeze()
b = torchhd.random(1, 4096, vsa="BSC", sparsity=0.998).squeeze()

print("--- Active Bits (these hypervectors are >99.8% sparse) ---")
print(f"Hypervector a:\n\t{active_bit_positions(a)}")
print(f"Hypervector b:\n\t{active_bit_positions(b)}")

# Move both hypervectors to the Sparsr processor (the software emulator by
# default; set SPARSR_BACKEND=fpgaf2 to run on real Sparsr FPGA hardware).
# From here on, standard Torchhd calls transparently dispatch to Sparsr
# hardware -- no torchhd_sparsr-specific API needed, same as moving tensors
# to "cuda".
a_sparsr = a.to("sparsr")
b_sparsr = b.to("sparsr")

# --- bind (VSA XOR): a single Sparsr WXOR instruction ---
bind_result = torchhd.bind(a_sparsr, b_sparsr)
print(f"\nResult of bind (Sparsr):\n\t{active_bit_positions(bind_result.to('cpu'))}")
check("bind", torchhd.bind(a, b), bind_result)

# --- bundle (VSA majority vote): refused, rather than answered wrongly ---
# torchhd resolves every position where the two operands disagree with a fair
# coin flip, and a fair coin flip is far too dense to store in Sparsr's CMEM.
# Sparsr would have to use a sparse tiebreak, which resolves those positions
# to 0 and hands back the all-zero hypervector. It raises instead; the
# uncompressed CMEM path is the hardware change that would lift the ceiling.
try:
    torchhd.bundle(a_sparsr, b_sparsr)
    raise AssertionError("bundle() should have refused these operands.")
except RuntimeError as error:
    print(f"\nAs expected, bundle() refuses rather than answering wrongly:\n\t{error}")

# Bundling a hypervector with itself consults no tiebreak at all, so it is
# exact and still runs on Sparsr hardware (AND/XOR/AND/OR).
check("bundle(a, a)", a, torchhd.bundle(a_sparsr, a_sparsr))

# --- similarity: XOR on Sparsr hardware, popcount on the host ---
cosine = torchhd.cosine_similarity(a_sparsr, b_sparsr).item()
cosine_cpu = torchhd.cosine_similarity(a, b).item()
print(f"\nCosine similarity: Sparsr={cosine:.6f} CPU={cosine_cpu:.6f}")
if abs(cosine - cosine_cpu) > 1e-6:
    raise AssertionError("Sparsr cosine_similarity does not match the CPU result.")
print("\u2714\ufe0f  cosine_similarity: Sparsr result matches the CPU result.")

# --- permute: needs a wide rotate, which Sparsr doesn't have yet ---
try:
    torchhd.permute(a_sparsr)
except NotImplementedError as error:
    print(f"\nAs expected, permute() is not available on Sparsr yet:\n\t{error}")
