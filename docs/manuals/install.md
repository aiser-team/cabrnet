# Build and install

## How to install all dependencies

With `pip`:

```bash
cd ../../ # Go back to root directory
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt -e .
```

With `conda`:

```bash
cd ../../ # Go back to root directory
conda env create -f environment.yml
conda activate cabrnet
python3 -m pip install -e .
```

With `micromamba`:

```bash
cd ../../ # Go back to root directory
micromamba create -f environment.yml
micromamba activate cabrnet
python3 -m pip install -e .
```

NOTE: As maintaining different requirements files is not trivial, the current
process is to use `pip` even when creating a `micromamba/conda` environment.
This implies that you should NOT install anything with `micromamba/conda`
afterward, as this could break your environment.

## Building the package

Once the dependencies are downloaded, the CaBRNet package can be built from
`pyproject.toml` as follows:

```bash
cd ../../ # Go back to root directory
python3 -m build
```

NOTE: this operation requires the `build` python package.

