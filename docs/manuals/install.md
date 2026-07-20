# Install

## Installing dependencies with UV:

The easiest and recommended way to install packages is with `uv` (you can still use pip if you want, see below)

Follow the official installation instructions here: https://docs.astral.sh/uv/getting-started/installation/

```bash
# make sure you are in the ROOT DIRECTORY
uv sync
# and run this line each time you restart you command line
source .venv/bin/activate
```

This installs the default development dependency group, including Ruff, pre-commit, and Pyright.

This should be enough for most users.

There are additional dependencies you can install:

```bash
# install ray-tune to use with bayesian optimizer:
uv sync --extra tune

# install documentation-related dependencies:
uv sync --group doc
```


## Installing with pip

If you prefer `pip` over `uv`:

```bash
# Make sure you are in the root directory
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
```

With pip, install the base package and the tools needed for the workflow you use:

```bash
# Development tools:
pip install -e . build "pre-commit>=4.6.0" pyright "ruff>=0.15.22" twine

# Documentation related dependencies:
pip install -e . mkdocs pydoc-markdown pymdown-extensions python-markdown-math

# Legacy testing dependencies:
pip install -e . "gdown~=4.7.1" "requests~=2.32.0" "Augmentor~=0.2.12"

# install ray-tune (for bayesian optimization)
pip install -e ".[tune]"
```

# Other requirements

IMPORTANT NOTE: CaBRNet also requires the [GraphViz](https://graphviz.org/)
package to generate explanations.

# Contributing

If you want to contribute to CaBRNet, see (CONTRIBUTING.md)[../../CONTRIBUTING.md]
