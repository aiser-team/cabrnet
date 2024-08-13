# CaBRNet applications
All CaBRNet applications are accessible through a single front-end script. To list all available applications, simply enter:
```bash
cabrnet --help
```
```
usage: cabrnet [-h] [--version] {evaluate,train,import,explain_local,explain_global,benchmark} ...

CaBRNet front-end

positional arguments:
  {evaluate,train,import,explain,explain_global}
                        sub-command help
    evaluate            evaluates a CaBRNet classifier
    train               trains a CaBRNet classifier
    import              converts an existing legacy model into a CaBRNet version
    explain_local       explains the decision of a CaBRNet classifier
    explain_global      explains the global behaviour of a CaBRNet classifier
    benchmark           computes a set of evaluation metrics on a CaBRNet model

options:
  -h, --help            show this help message and exit
  -V, --version         show program's version number and exit
```
To obtain the documentation for a specific application, simple enter `cabrnet <app_name> --help`, *e.g.*:
```bash
cabrnet train --help
```
Although CaBRNet already supports most applications required to build, train, evaluate and explain prototype-based models. 
it is also easy to [add new applications](applications.md).

## Common options
Some options are present in all applications:

- `--device <device>` allows to specify a target hardware device (by default, it is set to `cuda:0`).
- `--seed <value>` allows to specify the random seed to improve the [reproducibility](#reproducibility) of all
experiments (by default, it is set to 42, as it should be ;)).
- `--logger-level <level>` indicates the level of debug messages that should be displayed. 
CaBRNet uses [loguru](https://loguru.readthedocs.io/en/stable/) for logging messages.
- `--logger-file <path/to/file>` indicates where the debug messages should be saved (in addition to the 
standard error output). 
- `-v, --verbose` enables [tqdm](https://tqdm.github.io/) progression bars during long operations.

## Training 
`cabrnet train` is used to train a prototype-based model.

- `--model-arch|-m </path/to/file.yml>` indicates how to [build and initialize the model](model.md).
- `--dataset|-d </path/to/file.yml>` indicates how to [load and prepare the data for training](data.md).
- `--training|-t </path/to/file.yml>` indicates the [training parameters of the model](training.md).
- `--save-best|-b <metric> <min/max>` indicates how to determine the "best" model.
- `--output-dir|-o <path/to/output/directory>` indicates where to store the model checkpoints during training.

Note: If all configuration files are located in the same directory, it is possible to start the training using the 
`--config-dir <dir>` option, that is effectively equivalent to:

- `--model-arch <dir>/model_arch.yml`
- `--dataset <dir>/dataset.yml`
- `--training <dir>/training.yml`

To avoid inadvertently erasing a previous training run, CaBRNet will abort the training process if the output 
directory already exists. To override this check, use the `--overwrite` option.

### Sanity check
For a quick sanity check of a particular architecture or overall training configuration, it is possible to use the 
`--sanity-check` option that only processes 5 batches per training epoch. 

### Resuming computations
CaBRNet provides options to save training checkpoints and resuming the training process from a given checkpoint.

- `--checkpoint-frequency|-f <num_epochs>` indicates the frequency of checkpoints (in number of epochs). 
If not provided, **only the best model is kept during training** (in the `best/` subdirectory)
- `--resume-from|-r </path/to/checkpoint/directory>` indicates the directory from which the training process should resume
(if not provided, the training process starts from the first epoch, see above). 
More precisely, each training checkpoint directory contains:
    - a copy of the YML file describing the model architecture, as specified [here](model.md).
    - a copy of the YML file describing the dataset, as specified [here](data.md). 
    - a copy of the YML file describing the training configuration, as specified [here](training.md). 
    - a copy of the YML file describing the visualization configuration, as specified [here](visualize.md). 
    - the current model state dictionary.
    - the current state of all optimizers and learning rate schedulers.
    - a file `state.pickle` containing auxiliary information such as:
        - the index of the current epoch.
        - the hardware device used.
        - the current best metrics (*e.g.* accuracy or cross-entropy loss).
        - the random seed used originally.
        - the internal state of each random number generator (torch, numpy and python).

In resume mode, specifying the output directory (option `--output-dir`) is optional. 
If not provided, the parent directory of the checkpoint is used.

### Training process

CaBRNet assumes that the high-level training process is common to all prototype-based architectures:

- The model is initialized, usually from a pre-trained convolutional neural network that is used as a feature extractor 
(backbone), and with random prototypes.
- The model is trained for several epochs, modifying the values of the prototypes and the weights of the backbone.
- An optional *epilogue* takes place, in which:
    - The prototypes are *projected* to their closest vectors from a projection dataset (usually the training set).
      A CSV file is generated, containing for each prototype:
      - the index of the corresponding image inside the projection dataset
      - the coordinates of the corresponding vector inside the convolutional feature 
    - Weak prototypes are pruned.

Note that the order of operations in the epilogue depends on the chosen architecture.
Finally, when resuming computations, it is possible to load a given checkpoint and perform only the epilogue using the 
`--epilogue` option. 

### Summary
`cabrnet train` provides multiples modes of operation, that can be summarized as follows (for readability, 
we assume that all configuration files are stored in a single directory):

- `cabrnet train --config-dir CONFIG_DIR --output-dir OUTPUT_DIR [--checkpoint-frequency NUM]`: the training starts from the first epoch, 
and the best model is stored and updated in the `OUTPUT_DIR/best` directory. 
If the `OUTPUT_DIR/best` directory already exists, option `--overwrite` must be used to authorize the application to overwrite the result of a previous training.
If specified, a checkpoint is created every `NUM` epoch in the `OUTPUT_DIR/epoch_XXX` directory. 
At the end of the training, the epilogue is applied on the model stored in `OUTPUT_DIR/best`, and the resulting model is
saved in the `OUTPUT_DIR/final` directory.
- `cabrnet train --resume-from OUTPUT_DIR/CHECKPOINT [--checkpoint-frequency NUM]`: the training resumes from the model
saved in `OUTPUT_DIR/CHECKPOINT`, and the best model is stored and updated in the `OUTPUT_DIR/best` directory.
If specified, a checkpoint is created every `NUM` epoch in the `OUTPUT_DIR/epoch_XXX` directory.
At the end of the training, the epilogue is applied on the model stored in `OUTPUT_DIR/best`, and the resulting model is
saved in the `OUTPUT_DIR/final` directory. Note that in this case, option `--overwrite` is not necessary since the 
training process resumes from the same original directory `OUTPUT_DIR`.
- `cabrnet train --resume-from OUTPUT_DIR/CHECKPOINT --output-dir ALT_OUTPUT_DIR [--checkpoint-frequency NUM]`: 
the training resumes from the model saved in `OUTPUT_DIR/CHECKPOINT`, and the best model (**if any**) is stored and updated in the `ALT_OUTPUT_DIR/best` directory. 
If the `ALT_OUTPUT_DIR/best` directory already exists, option `--overwrite` must be used to authorize the application to overwrite the result of a previous training.
If specified, a checkpoint is created every `NUM` epoch in the `OUTPUT_DIR/epoch_XXX` directory.
At the end of the training, the epilogue is applied on the model stored in `ALT_OUTPUT_DIR/best` (if present), or on the model
stored in `OUTPUT_DIR/best` if no better model was found since resuming the training process, and the resulting model is
saved in the `ALT_OUTPUT_DIR/final` directory.
- `cabrnet train --resume-from OUTPUT_DIR/CHECKPOINT --epilogue`: the application loads the model saved in `OUTPUT_DIR/CHECKPOINT`,
and goes straight to the epilogue phase, applying the operation on that model (not necessarily the best model). The resulting model is
saved in the `OUTPUT_DIR/final` directory. 
If the `OUTPUT_DIR/final` directory already exists, option `--overwrite` must be used to authorize the application to 
overwrite the result of a previous epilogue.

## Importing a legacy model
To avoid restarting previous computations performed using the codes provided by the original authors,
CaBRNet offers a tool to import an existing dictionary into the new format, using the `cabrnet import` command.
Currently, this tool only supports ProtoPNet and ProtoTree.

Here is a short description of the options. As in `cabrnet train`:

- `--model-arch </path/to/file.yml>` indicates how to [build the model](model.md). 
- `--model-state-dict </path/to/model/state.pth>` indicates the location of the legacy state dictionary that should be used 
to initialize the model.
- `--output-dir <path/to/output/directory>` indicates where to store the imported model.

Note that after the loading the CaBRNet model with the parameters of the legacy model (feature extractor, prototypes, etc.),
CaBRNet **finalizes the import process by performing the optional model [epilogue](#training-process)**.
Therefore, the `cabrnet import` tool also requires the following information:

- `--dataset|-d </path/to/file.yml>` indicates how to [load and prepare the data for prototype projection](data.md).
- `--training|-t </path/to/file.yml>` indicates the [parameters of the epilogue](training.md) (if any).

Similar to the option in `cabrnet train`, the `--config-dir <dir>` option is equivalent to:

- `--model-arch <dir>/model_arch.yml`
- `--dataset <dir>/dataset.yml`
- `--training <dir>/training.yml`

## Evaluating a CaBRNet model
After training, it is possible to evaluate the loss and accuracy of a model using the `cabrnet evaluate` tool. 
To evaluate a model, the tool uses the following options:

- `--model-arch </path/to/file.yml>` indicates how to [build the model](model.md).
- `--model-state-dict </path/to/model/state.pth>` indicates
the location of a CaBRNet or legacy state dictionary that should be used to initialize the model.
- `--dataset|-d </path/to/file.yml>` indicates how to [load and prepare the test data for the evaluation](data.md).

To compute dedicated metrics for case-based reasoning models, the `cabrnet benchmark` tool uses the following options:

- `--model-arch </path/to/file.yml>` indicates how to [build the model](model.md).
- `--model-state-dict </path/to/model/state.pth>` indicates
the location of a CaBRNet or legacy state dictionary that should be used to initialize the model.
- `--dataset|-d </path/to/file.yml>` indicates how to [load and prepare the test data for the evaluation](data.md).
- `--visualization </path/to/file.yml>` indicates how to [visualize the prototypes](visualize.md).
- `--benchmark-configuration <path_to_file>` indicates how to [configure the metrics](evaluation.md).
- `--output-dir <path/to/output/directory>` indicates where to store the evaluation reports.

Similar to the `--config-dir` option in `cabrnet train`, the `--checkpoint-dir <dir>` option is equivalent to:

- `--model-arch <dir>/model_arch.yml`
- `--model-state-dict <dir>/model_state.pth`
- `--dataset <dir>/dataset.yml`

## Generating explanations
Prototype-based architectures provide both global and local explanations:

- global explanations provide an overview of the decision-making process of the entire model.
- local explanations provide information regarding a particular decision (for a particular image). 

### Global explanations
A global explanation is generated using the `explain_global` application (see the 
[MNIST example](mnist.md)). To generate such an explanation, 
the tool uses the following options:

- `--model-arch </path/to/file.yml>` and `--model-state-dict </path/to/model/state.pth>` indicate how to 
[build and initialize the model](model.md).
- `--dataset|-d </path/to/file.yml>` indicates how to [load and prepare the projection data](data.md) before extracting 
the prototype visualizations.
- `--projection-info <path/to/file.csv>` contains the [projection information](#training-process) of each prototype.
- `--visualization </path/to/file.yml>` indicates how to [visualize the prototypes](visualize.md).
- `--output-dir <path/to/output/directory>` indicates where to store the global explanation.

Similar to `cabrnet evaluate`, the `--checkpoint-dir <dir>` option is equivalent to:

- `--model-arch <dir>/model_arch.yml`
- `--model-state-dict <dir>/model_state.pth`
- `--dataset <dir>/dataset.yml`
- `--projection-info <dir>/projection_info.csv`

### Local explanations
A local explanation is generated using the `explain_local` application (see the 
[MNIST example](mnist.md)). Importantly, **local explanations require first to generate prototype 
visualizations** using the `cabrnet explain_global` [application](#global-explanations).
To generate such an explanation, the tool uses the following options:

- `--model-arch </path/to/file.yml>` and `--model-state-dict </path/to/model/state.pth>` indicate how to 
[build and initialize the model](model.md).
- `--image <path/to/image>` indicates which image should be classified by the model. 
- `--dataset|-d </path/to/file.yml>` indicates how to [prepare the image](data.md) before it is processed by the model,
based on the transformations applied to the *test* dataset.
- `--visualization </path/to/file.yml>` indicates how to [visualize the patches of the test image](visualize.md).
- `--prototype-dir <path/to/prototype/directory>` indicates where the prototype visualizations extracted during 
the [global explanation](#global-explanations) are stored (usually in `<explanation_directory>/prototypes/`).
- `--output-dir <path/to/output/directory>` indicates where to store the local explanation.
- `--overwrite` indicates that any existing explanation in the output directory can be overwritten.

As in `cabrnet evaluate`, the `--checkpoint-dir <dir>` option is equivalent to:

- `--model-arch <dir>/model_arch.yml`
- `--model-state-dict <dir>/model_state.pth`
- `--dataset <dir>/dataset.yml`

## Reproducibility
### Random number initialization and deterministic operations
For reproducibility purposes, CaBRNet explicitly uses a random `seed` to initialize the various random number 
generators that may be used during the training process, as shown [here](https://github.com/aiser-team/cabrnet/tree/master/src/main.py).
```python
import numpy as np
import torch
import random

seed = 42
torch.use_deterministic_algorithms(mode=True)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
```
Additionally, CaBRNet use a pytorch feature called [torch.use_deterministic_algorithm](https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html),
which ensures reproducible results for a given hardware/software configuration. 

**IMPORTANT NOTE**: for compatibility reasons,
it might be necessary to manually set the `CUBLAS_WORKSPACE_CONFIG` environment variable before launching the CaBRNet 
main tool.
```bash
export CUBLAS_WORKSPACE_CONFIG=:16:8
```
### Resuming a training process
As indicated [here](#resuming-computations), CaBRNet allows the training process to generate *checkpoints* (using the `--checkpoint-frequency` option), so that 
the entire training process can be resumed from any epoch, under the same conditions and - therefore - with the same outcome.
In other words, from a given checkpoint, random number generators are not reset using the original random seed but 
rather restored to their appropriate state with respect to the current epoch.

### Caveats on reproducibility
The reproducibility of experiments depends on various factors (software version, hardware), some of which
may not be obvious. For instance, although the batch size has a direct effect on batch normalization during training, it is
**also true during model evaluation**, as discussed [here](https://discuss.pytorch.org/t/cudnn-causing-inconsistent-test-results-depending-on-batch-size/189277).
In other words, a model in `eval()` mode does not return the same outputs depending on the size of the batch of data 
(and the target hardware). While this effect is limited when performing traditional linear operations (the tensors are usually `allclose` from pytorch point of view), 
the use of the **L2 distance between vectors tends to amplify the phenomenon**. In particular, this may have an effect during prototype projection, where an image patch may
be considered closer or farther than another patch depending on the batch size.

