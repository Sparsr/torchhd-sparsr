# Contributing

Thank you for looking at this package. A few things about how it is built and checked.

## Building

The package builds against an unpacked Sparsr HDC tarball, the free download from the
[Sparsr Developer Zone](https://developers.sparsr.com/), and the PyTorch you intend to run
against:

```sh
SPARSR_HDC_ROOT=/path/to/sparsr-hdc pip install --no-build-isolation .[test]
SPARSR_HDC_ROOT=/path/to/sparsr-hdc pytest tests
```

`README.md` has the rest.

## Pull requests

Open one against `main` and fill in the template. Keep a change to one thing. Squash on
merge: the history here is meant to be read.

## Continuous integration

CI fetches the HDC tarball from a Sparsr release with a token this repository holds. A pull
request from a fork does not have that token, so its full check cannot run and reports why.
A maintainer runs it by pushing your branch to this repository. If you would rather run the
check yourself, register at the Developer Zone for the tarball and use the two commands
above.
