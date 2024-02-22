# Build and install
## How to install all dependencies
With `pip`:

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt -e .
```

With `conda`/`mamba`:

```bash
conda env create -f environment.yml
conda activate cabrnet
python3 -m pip install -e .
```

or

```bash
mamba env create -f environment.yml
mamba activate cabrnet
python3 -m pip install -e .
```

With `micromamba`:

```bash
micromamba create -f environment.yml
micromamba activate cabrnet
python3 -m pip install -e .
```

## Building the package
Once the dependencies are downloaded, the CaBRNet package can be built from `pyproject.toml` as follows: 
```bash
python3 -m build
```
NOTE: this operation requires the  `build` python package.