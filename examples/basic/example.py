import os

# Where these operations run. "vm" is the Sparsr VM: a model of the Sparsr
# processor, in software, that ships inside this package. So this example needs
# no Sparsr hardware, and every number it prints below comes from that model.
#
# Switching to hardware is this one string, and nothing else in this file. When
# Sparsr hardware support is published, "fpgaf2" runs the same code on a real
# Sparsr card. That backend is not in the package yet, so choosing it today
# fails when this file imports torchhd_sparsr, and the host library says on
# stderr that the backend provides none of the operations.
#
# The README's backend table says what each name needs from this package.
#
# It has to be set before torchhd_sparsr is imported: the Sparsr host library
# reads the choice once, when the first operation runs. The package fills the
# same value in when nothing has set it, so this line changes no behaviour
# today. It is here to say out loud what is running.
SPARSR_BACKEND = "vm"
os.environ["SPARSR_BACKEND"] = SPARSR_BACKEND

import torch  # noqa: E402  (both imports must follow the backend selection above)
import torchhd  # noqa: E402

import torchhd_sparsr  # noqa: E402,F401  (registers the "sparsr" device)


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

print(f"--- Running on the Sparsr '{SPARSR_BACKEND}' backend ---")
print("--- Active Bits (these hypervectors are >99.8% sparse) ---")
print(f"Hypervector a:\n\t{active_bit_positions(a)}")
print(f"Hypervector b:\n\t{active_bit_positions(b)}")

# Move both hypervectors to the Sparsr processor, which is the VM selected at
# the top of this file. From here on, standard Torchhd calls dispatch to Sparsr
# on their own -- no torchhd_sparsr-specific API needed, same as moving tensors
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
