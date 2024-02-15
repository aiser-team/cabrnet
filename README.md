<img src="./docs/banner.svg">

CaBRNet is an open source library that offers an API to use state-of-the-art
prototype-based architectures, or easily add a new one.

Currently, CaBRNet supports the following architectures:
- **ProtoPNet**, as described in *Chaofan Chen, Oscar Li, Chaofan Tao, Alina Jade Barnett,
Jonathan Su and Cynthia Rudin.* [This Looks like That: Deep Learning for Interpretable Image Recognition](https://proceedings.neurips.cc/paper_files/paper/2019/file/adf7ee2dcf142b0e11888e72b43fcb75-Paper.pdf). 
Proceedings of the 33rd International Conference on Neural Information Processing Systems, page 8930–8941, 2019.
- **ProtoTree**, as described in *Meike Nauta, Ron van Bree and Christin Seifert.* [Neural Prototype Trees for Interpretable Fine-grained Image
Recognition](https://openaccess.thecvf.com/content/CVPR2021/papers/Nauta_Neural_Prototype_Trees_for_Interpretable_Fine-Grained_Image_Recognition_CVPR_2021_paper.pdf). 
2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14928–14938, 2021.
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

# CaBRNet front-end applications
All CaBRNet applications are accessible through a single front-end script. To list all available applications, simply enter:
```bash
cabrnet --help
```
```
usage: cabrnet [-h] [--version] [--device DEVICE] [--seed SEED] [--logger-level LOGGER_LEVEL] [--verbose] {download_examples,evaluate,train,explain,explain_global} ...

CaBRNet front-end

positional arguments:
  {download_examples,evaluate,train,explain,explain_global}
                        sub-command help
    download_examples   download example models
    evaluate            evaluate a CaBRNet classifier
    train               train a CaBRNet classifier
    explain             explain the decision of a CaBRNet classifier
    explain_global      explain the global behaviour of a CaBRNet classifier

options:
  -h, --help            show this help message and exit
  --version, -V         show program's version number and exit
  --device DEVICE       Target hardware device
  --seed SEED, -s SEED  Seed for reproducible experiments
  --logger-level LOGGER_LEVEL
                        Logger level and verbosity
  --verbose             Verbose output
```
To obtain the documentation for a specific application, simple enter `cabrnet <app_name> --help`, *e.g.*:
```bash
cabrnet train --help
```
```
usage: cabrnet train [-h] [--model-config /path/to/file.yml] [--model-state-dict /path/to/model/state.pth] [--dataset /path/to/file.yml] (--training /path/to/file.yml | --resume-from /path/to/checkpoint/directory) --training-dir path/to/training/directory [--save-best metric]
                     [--checkpoint-frequency num_epochs] --visualization /path/to/file.yml [--sanity-check-only]

options:
  -h, --help            show this help message and exit
  --model-config /path/to/file.yml
                        Path to the model configuration file
  --model-state-dict /path/to/model/state.pth
                        Path to the model state dictionary
  --dataset /path/to/file.yml, -d /path/to/file.yml
                        path to the dataset config
  --training /path/to/file.yml, -t /path/to/file.yml
                        Path to the training configuration file
  --resume-from /path/to/checkpoint/directory
                        Path to existing checkpoint directory
  --training-dir path/to/training/directory
                        Path to output directory
  --save-best metric    Save best model based on accuracy or loss
  --checkpoint-frequency num_epochs
                        Checkpoint frequency (in epochs)
  --visualization /path/to/file.yml
                        Path to the visualization configuration file
  --sanity-check-only   Check the training pipeline without performing the entire process.
```

## Configuration files
As indicated in the example above, CaBRNet uses YML files to specify:
- the [model architecture](src/cabrnet/generic/model.md).
- which [datasets](src/cabrnet/utils/data.md) should be used during training.
- the [training](src/cabrnet/utils/optimizers.md) parameters.
- how to visualize (TODO) the prototypes and generate explanations.


## Example: ProtoTree / MNIST
### Training
```bash
cabrnet --device cpu --seed 42 --verbose --logger-level INFO train \
  --model-config configs/prototree/mnist/model.yml \
  --dataset configs/prototree/mnist/data.yml \
  --training configs/prototree/mnist/training.yml \
  --training-dir runs/mnist_prototree \
  --visualization configs/prototree/mnist/visualization.yml
```
This command trains a ProtoTree during one epoch, and stores the resulting checkpoint in 
`runs/mnist_prototree/final`.

### Global explanation
```bash
cabrnet --verbose explain_global \
  --model-config runs/mnist_prototree/final/model.yml \
  --model-state-dict runs/mnist_prototree/final/model_state.pth \
  --output-dir runs/mnist_prototree/global_explanation \ 
  --prototype-dir runs/mnist_prototree/prototypes/ 
```
This command generates a global explanation for the ProtoTree model and stores the result in 
`runs/mnist_prototree/global_explanation`.

<img src="./docs/website/docs/img/mnist_global_explanation.png">

### Local explanation
```bash
cabrnet --verbose explain \
  --model-config runs/mnist_prototree/final/model.yml  \
  --model-state-dict runs/mnist_prototree/final/model_state.pth \
  --dataset configs/prototree/mnist/data.yml \
  --visualization configs/prototree/mnist/visualization.yml \
  --prototype-dir runs/mnist_prototree/prototypes/ \
  --output-dir runs/mnist_prototree/local_explanations/  \
  --image examples/images/mnist_sample.png
```
This command generates a local explanation for the image stored in `examples/images/mnist_sample.png` and stores the result in 
`runs/mnist_prototree/local_explanation`.

<img src="./docs/website/docs/img/mnist_local_explanation.png">

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
### Random number initialization and deterministic operations
For reproducibility purposes, CaBRNet explicitly uses a random `seed` to initialize the various random number 
generators that may be used during the training process, as shown [here](src/main.py).
```python
import numpy as np
import torch
import random

torch.use_deterministic_algorithms(mode=True)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
```
Additionally, CaBRNet use a pytorch feature called [torch.use_deterministic_algorithm](https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html),
which ensures reproducible results for a given hardware/software configuration. IMPORTANT NOTE: for compatibility reasons,
it might be necessary to manually set the `CUBLAS_WORKSPACE_CONFIG` environment variable before launching the CaBRNet 
main tool.
```bash
export CUBLAS_WORKSPACE_CONFIG=:16:8
```
### Resuming a training process
CaBRNet allows the training process to generate *checkpoints* (using the `--checkpoint-frequency` option), so that 
the entire training process can be resumed from any epoch, under the same conditions and - therefore - with the same outcome.
More precisely, each training checkpoint contains:
- a copy of the YML file describing the model architecture, as specified [here](src/cabrnet/generic/model.md).
- a copy of the YML file describing the dataset, as specified [here](src/cabrnet/utils/data.md). 
- a copy of the YML file describing the training configuration, as specified [here](src/cabrnet/utils/optimizers.md). 
- the current model state dictionary.
- the current state of all optimizers and learning rate schedulers.
- a file `state.pickle` containing auxiliary information such as:
  - the index of the current epoch.
  - the hardware device used.
  - the current best metrics (*e.g.* accuracy or cross-entropy loss).
  - the random seed used originally.
  - the internal state of each random number generator (torch, numpy and python). 

In other words, from a given checkpoint, random number generators are not reset using the original random seed but 
rather restored to their appropriate state with respect to the current epoch.