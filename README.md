# CaBRNet: Case-Based Reasoning Networks

CaBRNet is an open source library aiming to offer an API to use
state-of-the-art prototype-based architectures, or easily add a new one.


## Build and install

Create a conda/mamba environment using the provided environment file. With
micromamba:

```
micromamba create -f environment.yml
```

This may take a while. Then activate the environment.

```
micromamba activate cabrnet
```

Once the dependencies are downloaded, to build using pyproject.toml with package build installed, you can use `python3 -m build`

## Testing a ProtoTree training on MNIST

```bash
cabrnet --device cpu --seed 42 --verbose --logger-level DEBUG train --model-config configs/prototree/mnist/model.yml --dataset configs/prototree/mnist/data.yml --training configs/prototree/mnist/training.yml --training-dir logs/
```


## Adding new applications
To add a new application to the CaBRNet main tool, simply add
a new file <my_application_name.py> into the directory <src/apps>. 
This file should contain:
1. A string `description` containing the purpose of the application.
2. A method `create_parser` adding the application arguments to an existing parser (or creating one if necessary)
3. A method `execute` taking the parsed arguments and executing the application code. 
```
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
