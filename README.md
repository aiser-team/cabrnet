<img src="./docs/banner.svg">

CaBRNet is an open source library aiming to offer an API to use state-of-the-art
prototype-based architectures, or easily add a new one.

## Build and install

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

Once the dependencies are downloaded, to build using `pyproject.toml` with package build installed, you
can use `python3 -m build`

## Testing a ProtoTree training on MNIST

```bash
cabrnet --device cpu --seed 42 --verbose --logger-level DEBUG train --model-config configs/prototree/mnist/model.yml --dataset configs/prototree/mnist/data.yml --training configs/prototree/mnist/training.yml --training-dir logs/
```

## Configuration files
CaBRNet uses YML files to specify:
- the model architecture (see TODO)
- which [datasets](src/cabrnet/utils/data.md) should be used during training.
- the [training](src/cabrnet/utils/optimizers.md) parameters.
- how to visualize (TODO) the prototypes and generate explanations. 

## Adding new applications

To add a new application to the CaBRNet main tool, simply add a new file
`<my_application_name.py>` into the directory `<src/apps>`. This file should
contain:

1. A string `description` containing the purpose of the application.
2. A method `create_parser` adding the application arguments to an existing
   parser (or creating one if necessary)
3. A method `execute` taking the parsed arguments and executing the application
   code.

```python
<src/apps/my_awesome_app.py>

description = "my new awesome CaBRNet application"


def create_parser(parser: ArgumentParser | None = None) -> ArgumentParser:
    if parser is None:
        parser = ArgumentParser(description)
    parser.add_argument(
        "--message",
        type=str,
        required=True,
        metavar="<message>",
        help="Message to be printed",
    )
    return parser


def execute(args: Namespace) -> None:
    print(args.message)
```

## Reproducibility
TODO