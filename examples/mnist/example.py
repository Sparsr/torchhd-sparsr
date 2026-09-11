"""Recognise handwritten digits on Sparsr, with standard Torchhd code.

This is hyperdimensional computing (HDC), also called a vector symbolic
architecture (VSA): an image becomes one 4096-bit hypervector, a digit class
becomes one 4096-bit hypervector, and recognising a digit means finding the
class hypervector the image hypervector resembles most. There are no
gradients, no training loop and no floating-point arithmetic.

Run it with no arguments to train on all 60,000 MNIST training images and
classify all 10,000 test images. It takes about half a minute and reports
about 79% accuracy.

What runs where matters, and the report at the end says it in numbers. The
comparison against every class prototype runs on Sparsr: each score is one
wide AND instruction plus its population count. Encoding and training run on
the host with plain Torchhd, because the two operations they need are the two
this package cannot run on Sparsr yet. `torchhd.bundle()` refuses on the
"sparsr" device -- the fair coin it needs for a tie is too dense to store in
CMEM -- and `torchhd.multiset()` needs a batch of hypervectors, which the
device does not take. Both are hardware limits with planned fixes, and the
README says more about each.
"""

import os

# Where the comparisons run. "vm" is the Sparsr VM: a model of the Sparsr
# processor, in software, that ships inside this package. So this example needs
# no Sparsr hardware, and every number it prints comes from that model.
#
# Switching to hardware is this one string, and nothing else in this file. When
# Sparsr hardware support is published, "fpgaf2" runs the same code on a real
# Sparsr card. That backend is not in the package yet, so choosing it today
# fails when this file imports torchhd_sparsr, and the host library says on
# stderr that the backend provides none of the operations.
#
# It has to be set before torchhd_sparsr is imported: the Sparsr host library
# reads the choice once, when the first operation runs.
SPARSR_BACKEND = "vm"
os.environ["SPARSR_BACKEND"] = SPARSR_BACKEND

import argparse  # noqa: E402  (everything below must follow the backend selection above)
import sys  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
import torchhd  # noqa: E402

import torchhd_sparsr  # noqa: E402,F401  (registers the "sparsr" device)

# A Sparsr hypervector is 4096 bits, and CMEM stores it as 128 lanes of 32
# bits. A row holds at most 48 non-zero lanes, so a hypervector whose set bits
# all live inside 48 lanes always fits, at any density -- see the README.
DIMENSIONS = 4096
LANE_BITS = 32
MAX_LANES = 48

PIXELS = 28 * 28
INK_THRESHOLD = 127
CLASSES = 10


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="./data", help="Where MNIST is stored. Downloaded here on first run.")
    parser.add_argument("--train", type=int, default=0, help="Training images to learn from. 0 means all of them.")
    parser.add_argument("--test", type=int, default=0, help="Test images to classify. 0 means all of them.")
    parser.add_argument("--lanes", type=int, default=MAX_LANES, help=f"Width of the code, 1 to {MAX_LANES} lanes.")
    parser.add_argument("--seed", type=int, default=1, help="Seed for the item memory.")
    arguments = parser.parse_args(argv)
    if not 1 <= arguments.lanes <= MAX_LANES:
        parser.error(f"--lanes must be between 1 and {MAX_LANES}: a CMEM row holds no more than {MAX_LANES} lanes.")
    return arguments


def load_mnist(directory):
    """Load MNIST with torchvision, downloading it on first run."""
    try:
        from torchvision.datasets import MNIST
    except ImportError:
        print(
            "This example reads MNIST with torchvision, which is not installed.\n"
            "    pip install torchvision",
            file=sys.stderr,
        )
        raise SystemExit(2)

    train = MNIST(directory, train=True, download=True)
    test = MNIST(directory, train=False, download=True)
    return train, test


def random_item_memory(lanes, seed):
    """One hypervector per pixel position, drawn once and never changed.

    Two positions get unrelated hypervectors, which is what makes the encoding
    below say *which* pixels were lit.

    Every set bit lives inside the first `lanes` lanes, and the bits inside
    those lanes are at 50% density. That is a dense code roughly `lanes * 32`
    bits wide, and it is the whole reason this example fits on the device:
    Sparsr's limit counts occupied lanes, not set bits. A code of the same
    weight scattered across all 4096 positions would occupy far more than 48
    lanes and be refused.

    Density is what accuracy comes from here. At the full 48 lanes the code
    carries about 1,536 bits of signal; at 8 lanes it carries 256, and accuracy
    drops with it.
    """
    generator = torch.Generator().manual_seed(seed)
    bits = torch.zeros(PIXELS, DIMENSIONS, dtype=torch.bool)
    width = lanes * LANE_BITS
    bits[:, :width] = torch.rand(PIXELS, width, generator=generator) < 0.5
    return bits.as_subclass(torchhd.BSCTensor)


def encode(image, item_memory, generator):
    """Turn one 28x28 image into one hypervector: the majority vote of the
    hypervectors of its lit pixels.

    A bit of the result is set where more than half of those pixels' vectors
    had it set, so two images of the same digit encode to similar
    hypervectors. A position no member set stays 0, so the result stays inside
    the item memory's lanes and stays storable on the device.

    This vote runs on the host. It needs a batch of hypervectors, and the
    "sparsr" device takes one hypervector at a time, so it cannot be
    dispatched there today.

    `torchhd.multiset(item_memory[lit])` is the same call with one difference:
    an image with an even number of lit pixels has tied positions, Torchhd
    breaks a tie with a coin, and `multiset` draws that coin from the global
    generator. Passing one keeps a run repeatable.
    """
    lit = image.reshape(-1) > INK_THRESHOLD
    return item_memory[lit].multibundle(generator=generator)


def train(images, labels, item_memory, generator):
    """One prototype hypervector per digit: the majority vote of every training
    image of that digit.

    This is the same vote as `encode`, over thousands of members instead of
    about 150, and it is one pass over the data. It counts set bits per
    position as it goes rather than stacking every encoded image and calling
    `torchhd.multiset` once, which is the same arithmetic with memory that
    stays flat instead of growing to 60,000 hypervectors.
    """
    counts = torch.zeros(CLASSES, DIMENSIONS, dtype=torch.int32)
    members = torch.zeros(CLASSES, dtype=torch.int32)

    for image, label in zip(images, labels):
        digit = int(label)
        counts[digit] += encode(image, item_memory, generator).int()
        members[digit] += 1

    # Strictly more than half, so a tie resolves to 0. Torchhd breaks a tie
    # with a fair coin, which is fine on the host but would put bits outside
    # the item memory's lanes and make the prototype unstorable on the device.
    prototypes = (counts * 2 > members.unsqueeze(1)).as_subclass(torchhd.BSCTensor)
    return prototypes, members


def classify(query, prototypes_on_device):
    """Score one encoded image against all ten prototypes on Sparsr.

    This is the part that runs on the device. Each `cosine_similarity` call
    intersects the two hypervectors with one wide AND instruction and counts
    the bits of the result, and the count is a mode on the same instruction:
    nothing reads a 4096-bit row back to the host to count it.

    One pair at a time, because the device takes one hypervector per tensor.
    """
    query_on_device = query.to("sparsr")
    scores = [torchhd.cosine_similarity(query_on_device, prototype).item() for prototype in prototypes_on_device]
    return max(range(CLASSES), key=scores.__getitem__)


def check_device_matches_host(query, prototypes, prototypes_on_device):
    """Confirm the device's ten scores are the numbers the host would compute.

    Worth one image of work: the scores decide every answer below, and a
    device that returned plausible-looking wrong ones would show up only as
    slightly worse accuracy.
    """
    for digit in range(CLASSES):
        on_device = torchhd.cosine_similarity(query.to("sparsr"), prototypes_on_device[digit]).item()
        on_host = torchhd.cosine_similarity(query, prototypes[digit]).item()
        if abs(on_device - on_host) > 1e-6:
            raise AssertionError(
                f"Sparsr scored digit {digit} at {on_device:.6f} and the host at {on_host:.6f}."
            )
    print("Checked: Sparsr's ten scores for the first test image match the host's.")


def report(members, tested, correct, device_operations, seconds, lanes):
    total_tested = int(tested.sum())
    total_correct = int(correct.sum())
    accuracy = 100.0 * total_correct / total_tested

    print(f"\nAccuracy: {accuracy:.2f}%  ({total_correct} of {total_tested} correct)\n")
    print("  digit   trained on   tested   correct   accuracy")
    for digit in range(CLASSES):
        per_digit = 100.0 * int(correct[digit]) / max(int(tested[digit]), 1)
        print(f"      {digit}   {int(members[digit]):10}   {int(tested[digit]):6}   {int(correct[digit]):7}   {per_digit:7.1f}%")

    print(f"\nCode width: {lanes} lanes, about {lanes * LANE_BITS} bits of signal per hypervector.")
    print(f"Ran in {seconds:.1f} s, of which {device_operations} comparisons ran on the Sparsr '{SPARSR_BACKEND}' backend.")
    print("Encoding and training ran on the host: see the note at the top of this file for why.")


def main(argv=None):
    arguments = parse_arguments(argv)
    train_set, test_set = load_mnist(arguments.data)

    train_count = arguments.train or len(train_set.data)
    test_count = arguments.test or len(test_set.data)
    print(f"Training on {train_count} images, testing on {test_count}, on the Sparsr '{SPARSR_BACKEND}' backend.")

    started = time.time()
    item_memory = random_item_memory(arguments.lanes, arguments.seed)

    # The coin that breaks a tied vote, seeded so a run repeats exactly. See
    # encode() for what it is for.
    generator = torch.Generator().manual_seed(arguments.seed)

    prototypes, members = train(train_set.data[:train_count], train_set.targets[:train_count], item_memory, generator)

    # The prototypes move to the device once and stay there for every test
    # image. A "sparsr" tensor holds its bits in host memory today, so this
    # saves the conversion rather than a transfer, but it is the call site that
    # device residency would make free.
    prototypes_on_device = [prototypes[digit].to("sparsr") for digit in range(CLASSES)]

    check_device_matches_host(encode(test_set.data[0], item_memory, generator), prototypes, prototypes_on_device)

    tested = torch.zeros(CLASSES, dtype=torch.int32)
    correct = torch.zeros(CLASSES, dtype=torch.int32)
    device_operations = CLASSES

    for image, label in zip(test_set.data[:test_count], test_set.targets[:test_count]):
        digit = int(label)
        predicted = classify(encode(image, item_memory, generator), prototypes_on_device)
        device_operations += CLASSES
        tested[digit] += 1
        if predicted == digit:
            correct[digit] += 1

    report(members, tested, correct, device_operations, time.time() - started, arguments.lanes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
