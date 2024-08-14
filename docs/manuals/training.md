# Training configuration
The training configuration associated with a model is stored in a YML file, according to the following specification.
For more examples, see the [ProtoPNet](https://github.com/aiser-team/cabrnet/tree/main/configs/protopnet/cub200/training.yml) and 
[ProtoTree](https://github.com/aiser-team/cabrnet/tree/main/configs/prototree/cub200/training.yml) configuration files.

## Defining parameter groups
Parameters of the model can be sorted into different groups, using the following formats:
```yaml
param_groups:
  <GROUP_NAME_1>: [<SUBMODULE_OR_PARAM_NAME_1>, <SUBMODULE_OR_PARAM_NAME_2>]
  <GROUP_NAME_2>:
    start: <SUBMODULE_OR_PARAM_NAME_1> 
    stop: <SUBMODULE_OR_PARAM_NAME_2>
```
In other words, parameter groups can be either defined as an explicit list of submodule/parameters names, 
or as a range of submodules/parameters, where the names of the parameters can be found as follows:
```python
for name, _ in model.named_parameters():
    print(name)
```
Note that for ranges of parameters, either `start` or `stop` keywords can be omitted (not both):

- If the `start` keyword is omitted, the range starts from the first parameter of the model.
- If the `stop` keyword is omitted, the range ends at the last parameter of the model.



Finally, when no parameter group is specified, all model parameters are automatically regrouped into a `main` group. 

## Configuring optimizers
Optimizers are configured under the `optimizers` keyword, according to the following format:
```yaml
optimizers:
  <OPTIMIZER_NAME_1>:
    type: <optimization_function> # As given in torch.optim (e.g. Adam, SGD)
    groups:
      <GROUP_NAME_1>:
        <OPTIMIZER_PARAM_1>: <value>
        <OPTIMIZER_PARAM_2>: <value>
        ...
      <GROUP_NAME_2>:
        <OPTIMIZER_PARAM_1>: <value>
        <OPTIMIZER_PARAM_2>: <value>
        ...
    params: # Applied globally
      <OPTIMIZER_PARAM_1>: <value>
      <OPTIMIZER_PARAM_2>: <value>
    scheduler:
      type: <lr_scheduler_function> # As given in torch.optim.lr_scheduler (e.g. StepLR)
      params:
        <SCHEDULER_PARAM_1>: <value>
        <SCHEDULER_PARAM_2>: <value>
        ...
  <OPTIMIZER_NAME_2>:
    ...
```
Hence, a single optimizer can be used for different parameter groups, 
with different configurations (*e.g.* learning rate) for each parameter group.

Additionally, each optimizer can be associated with an **optional** learning rate scheduler.

## Specifying the number of training epochs
The total number of training epochs is given using the `num_epochs` keyword, as follows:
```yaml
num_epochs: <VALUE>
```
Note that, as an option of the [train](cabrnet.md#training-) application, it is possible to 
define an early stop condition based on the number of epochs spent without finding a better model
(`--patience` option).

## Training periods
Using the keyword `periods`, it is possible to define training periods, i.e. ranges of epochs where specific optimizers are used
and specific parameter groups can be frozen (not updated during the back-propagation phase).

```yaml
periods:
  <PERIOD_NAME_1>:
    epoch_range: [<FIRST_EPOCH_INDEX>, <LAST_EPOCH_INDEX] # LAST_EPOCH_INDEX is included
    optimizers: <OPTIMIZER_NAME_1> or [<OPTIMIZER_NAME_1>, <OPTIMIZER_NAME_2>, ...]
    freeze: <GROUP_NAME_1> or [<GROUP_NAME_1>, <GROUP_NAME_2>, ...]
  <PERIOD_NAME_2>:
    ...
```
When no period matches the current epoch index, all optimizers are used for each training epoch and 
no parameter groups are frozen. It is also true in the particular case when no period is specified.

As an alternative to `epoch_range`, it is possible to define the number of epochs *per period* using the
`num_epochs` keyword inside the period definition. In this case, periods are treated one after another (no overlap is possible), *e.g.*

```yaml
num_epochs: 10
periods:
  warmup:
    epoch_range: [0, 3]
    optimizers: warmup_optimizer
  main_training:
    epoch_range: [4, 9]
    optimizers: main_optimizer
```
can be written as:
```yaml
num_epochs: 10
periods:
  warmup:
    num_epochs: 4
    optimizers: warmup_optimizer
  main_training:
    num_epochs: 6 # Optional for the last period
    optimizers: main_optimizer
```
Note that the `num_epochs` field is optional for the last period (in this case, it is replaced by the remaining number
of epochs, as given in the global `num_epochs` field).

## Configuring the epilogue
Specific architectures such as ProtoPNet or ProtoTree can end the training process with an additional step, 
called the epilogue, in which operations such as prototype pruning are performed. The configuration of this step
is specific to each architecture but follows the same format:
```yaml
epilogue:
  <EPILOGUE_PARAM_1>: <VALUE>
  <EPILOGUE_PARAM_2>: <VALUE>
...
```

## Creating an Optimizer Manager
CaBRNet provides a class [OptimizerManager](https://github.com/aiser-team/cabrnet/tree/main/src/cabrnet/utils/optimizers.py) in charge of parsing the configuration file 
and handling optimizers/schedulers during the training process.
```python
from cabrnet.utils.optimizers import OptimizerManager
from cabrnet.utils.parser import load_config
from cabrnet.generic.model import CaBRNet

model = CaBRNet.build_from_config(config="<path/to/model/configuration/file.yml") 
training_config = load_config(config_file="<path/to/training/configuration/file.yml>")
optimizer_mngr=OptimizerManager(config_dict=training_config,module=model)
```
For more information on how to build a CaBRNet model from a configuration file, see [here](model.md).
An Optimizer Manager provides four main functions that are used during training:

- `zero_grad()`: Reset all optimizer(s) gradients (before each batch of data)
- `freeze(epoch: int)`: Freeze all relevant model parameters, according to the current `epoch`.
- `optimizer_step(epoch: int)`: Update all relevant optimizers, according to the current `epoch`. 
- `scheduler_step(epoch: int)`: Update all relevant schedulers, according to the current `epoch`.

# Hyperparameter tuning using Bayesian optimization
Case-based reasoning models often require to balance multiple hyperparameters corresponding to 
different learning objectives. To explore the search space of possible parameter values in an
efficient manner, CaBRNet supports hyperparameter tuning using the Bayesian optimization engine
of [Ray-Tune](https://docs.ray.io/en/latest/tune/index.html).

## Defining the search space
In practice, the search space of any **discrete** parameter that is used to configure the model (`model_arch.yml`), 
the dataset (`dataset.yml`), or the training process (`training.yml`) can be defined using a *search space
configuration file* (usually `search_space.yml`), according to the following format:

```yaml
model:
  <KEY_INSIDE_MODEL_CONFIG>: [VALUE1, VALUE2, ...]

training:
  <KEY_INSIDE_TRAINING_CONFIG>: [VALUE1, VALUE2, ...]
  
dataset:
  <KEY_INSIDE_DATASET_CONFIG>: [VALUE1, VALUE2, ...]
```

For example, given the following configuration files:
```yaml
<model_arch.yml>
top_arch:
  module: cabrnet.protopnet.model
  name: ProtoPNet
extractor: ...
similarity: ...
classifier:
  module: cabrnet.protopnet.decision
  name: ProtoPNetClassifier
  params:
    num_classes: 10
    num_proto_per_class: 10
    ...

<training.yml>
param_groups:
  backbone: extractor.convnet
  main_layers: [extractor.add_on, classifier.prototypes]
  last_layer: classifier.last_layer

optimizers:
  warmup_optimizer:
    type: Adam
    groups:
      main_layers:
        lr: 0.001
        momentum: 0.9
  joint_optimizer:
    type: Adam
    groups:
      backbone:
        lr: 0.001
      main_layers:
        lr: 0.001
        momentum: 0.9
  last_layer_optimizer:
    type: Adam
    groups:
      last_layer:
        lr: 0.0001

num_epochs: 100

periods:
  warmup:
    epoch_num : 10
    optimizers: warmup_optimizer
  main_training:
    optimizers: joint_optimizer
```
it possible to define the search space of various parameters as follows:
```yaml
model:
    classifier:
      params:
        num_proto_per_class: [5, 10, 15] # List of possible values to explore
training:
    optimizers:
      warmup_optimizer:
        groups:
          main_layers:
            lr: [0.001, 0.0001, 0.0005]
            momentum: [0.9, 0.5]
      joint_optimizer:
        type: Adam
        groups:
          backbone:
            lr: [0.001, 0.01]
    
    num_epochs: [100, 150]
    
    periods:
      warmup:
        epoch_num : [10, 20, 50]
```
Currently, CaBRNet only supports lists of explicit values that can be tested for each parameter
(using [`tune.choice`](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.choice.html)).

## Defining the training / optimization objectives
For each trial (corresponding to one particular point inside the search space), CaBRNet models are optimized
*w.r.t.* a given metric (*e.g.* average loss) that is usually computed on the training set and specified by the `--save-best` option 
(see [here](cabrnet.md#defining-the-objective-function)).

However, the general optimization objective for hyperparameter tuning - specified with the [`--search-space` option](cabrnet.md#hyperparameter-tuning-using-bayesian-optimization) - 
can be different from the training objective
(*e.g.* accuracy on a validation set after the epilogue takes place). In practice, for each trial, after the epilogue is 
applied on the "best" trained model, the model is evaluated on the `test_set` (as defined [here](data.md)). Similar
to the training objective, any metric returned by the [`evaluate`](model.md#defining-a-new-top-module) method can be
used as a global optimization objective for the hyperparameter tuning.
