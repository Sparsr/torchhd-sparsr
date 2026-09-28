# Recognising handwritten digits on Sparsr

This example classifies MNIST digits with **78.8% accuracy**, trained on all 60,000 training
images and tested on all 10,000 test images. It is one file of ordinary Torchhd code, and the
comparison that decides every answer runs on Sparsr.

Training is one pass over the data, and it uses integer bit operations only. This is
*hyperdimensional computing* (HDC), also called a *vector symbolic architecture* (VSA). An image
becomes one 4096-bit hypervector, and so does a digit class. Recognising a digit means finding
the class hypervector that the image hypervector resembles most.

## Run it

```sh
pip install torchhd-sparsr torchvision
python example.py
```

MNIST is about 11 MB and torchvision downloads it into `./data` on first run. A full run takes
about half a minute. For a quicker look, use a slice of the data:

```sh
python example.py --train 6000 --test 1000
```

It runs on the **Sparsr VM**, a model of the Sparsr processor in software that is included in
the package, so it does not require Sparsr hardware. The first lines of `example.py` name that
choice in one string, and say what changing it would do.

## What it prints

```
Training on 60000 images, testing on 10000, on the Sparsr 'vm' backend.
Checked: Sparsr's ten scores for the first test image match the host's.

Accuracy: 78.79%  (7879 of 10000 correct)

  digit   trained on   tested   correct   accuracy
      0         5923      980       888      90.6%
      1         6742     1135      1026      90.4%
      2         5958     1032       788      76.4%
      3         6131     1010       819      81.1%
      4         5842      982       743      75.7%
      5         5421      892       576      64.6%
      6         5918      958       779      81.3%
      7         6265     1028       824      80.2%
      8         5851      974       706      72.5%
      9         5949     1009       730      72.3%

Code width: 48 lanes, about 1536 bits of signal per hypervector.
Ran in 32.8 s, of which 100010 comparisons ran on the Sparsr 'vm' backend.
```

Digit 1 is the easiest and digit 5 is the hardest, which is what an HDC classifier usually
finds on MNIST.

## The algorithm, in four steps

**1. Item memory.** Give each of the 784 pixel positions a fixed random hypervector, drawn once
and never changed. Any two positions get unrelated hypervectors, which is what makes the
encoding below say *which* pixels were lit.

**2. Encode.** An image is the **majority vote** of the hypervectors of its lit pixels. A bit of
the result is set where more than half of those pixels' vectors had it set. Images of the same
digit light similar pixels, so they encode to similar hypervectors.

**3. Train.** A class prototype is the majority vote of the training images of that digit. One
pass over the data, and each class is one hypervector.

**4. Classify.** Encode the test image, then score it against all ten prototypes with
`torchhd.cosine_similarity` and pick the best.

## What runs on Sparsr, and what does not

| Step | Runs where | Why |
| --- | --- | --- |
| Item memory | Host | Random hypervectors, generated once. |
| Encode | Host | Needs a batch of hypervectors, which the device does not take. |
| Train | Host | The same vote, over thousands of members. |
| Classify | **Sparsr** | Each score is one wide AND instruction and its population count. |

Step 4 is 10 scores per test image, so a full run is 100,010 operations on the device. Each one
intersects the two hypervectors with one wide instruction, and the count of set bits comes
back from the same instruction: nothing reads a 4096-bit row back to the host to count it.

Steps 2 and 3 are on the host because of two limits this package documents in its own README,
not because a vote is unsuited to the device:

- **`torchhd.bundle()` refuses on the `"sparsr"` device.** Where two hypervectors disagree,
  Torchhd decides the position with a fair coin flip, and a fair coin flip is too dense to store
  in the processor's wide memory. The package raises an error rather than returning the
  all-zero hypervector that a sparse coin would produce.
- **`torchhd.multiset()` needs a batch**, and a `"sparsr"` tensor is one hypervector.

So the example votes on the host and keeps the comparison on the device. An uncompressed memory
path and real device capacity are both planned. Once both exist, steps 2 and 3 move to the
device without changing what the algorithm is.

## Before you write your own

### Dense codes, 48 lanes wide

The device receives a hypervector as a wide memory row, and **a row stores at most 48
non-zero four-byte lanes out of 128**. The limit counts *lanes*, never set bits, and it stores
each occupied lane whole, so bits inside an occupied lane are free.

That makes two very different codes storable, and they are not equally good:

| Code | Set bits | Fits a row? |
| --- | --- | --- |
| Sparse, spread over all 4096 positions | a few dozen | only while it stays sparse |
| Dense, confined to 48 lanes | about 768 | always |

This example uses the dense one, and it matters: it carries far more signal, and it classifies
much better. Every hypervector here, item memory, encoded image and prototype alike, stays
inside those 48 lanes by construction, because a majority vote does not set a bit unless one of
its members had it. That is why nothing here is refused for being too dense.

This is the constraint you will hit first when you write your own HDC code for Sparsr. Design
your encoding so that its results can be stored, rather than finding out afterwards that they
cannot.

### The vote, not the union

Superposing hypervectors has two forms. Picking the wrong one here produces a classifier no
better than chance.

- The **union** ORs every member together. It is exact and never loses a member, but it *grows*.
  OR a few hundred images of one digit together and every bit is set, so the prototype says
  nothing about that digit and every class looks the same.
- The **majority vote** stays informative however many members it has, which is what a prototype
  needs.

## Options

| Option | Meaning |
| --- | --- |
| `--data DIR` | Where MNIST is stored. Downloaded there on first run. Default: `./data`. |
| `--train N` | Training images to learn from. Default: all of them. |
| `--test N` | Test images to classify. Default: all of them. |
| `--lanes N` | Width of the code, 1 to 48 lanes. Default: 48. |
| `--seed N` | Seed for the item memory and the tie-breaking coin. Default: 1. |

`--lanes` is the interesting one. It is the density knob, and narrowing it costs accuracy
directly. Measured on 6,000 training and 1,000 test images:

| `--lanes` | Bits of the code | Accuracy |
| --- | --- | --- |
| 8 | 256 | 63.6% |
| 16 | 512 | 71.9% |
| 48 | 1536 | 73.5% |

48 is the widest a wide memory row can hold, so 1,536 bits is the ceiling this example
can reach today.

```sh
python example.py --train 6000 --test 1000 --lanes 8
```
