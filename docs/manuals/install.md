# Install

## How to install all dependencies

```bash
cd ../../ # Go back to root directory
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -e .
```

- To install development related dependencies

```bash
pip install -e .[dev]
```

- To install documentation related dependencies

```bash
pip install -e .[doc]
```

- To install legacy testing related dependencies

```bash
pip install -e .[legacy]
```

IMPORTANT NOTE: CaBRNet also requires the [GraphViz](https://graphviz.org/) 
package to generate explanations.
