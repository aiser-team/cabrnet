# Build and install

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

## Building the package

Once the dependencies are downloaded, the CaBRNet package can be built from
`pyproject.toml` as follows:

```bash
cd ../../ # Go back to root directory
python3 -m build
```

NOTE: this operation requires the `build` python package.
