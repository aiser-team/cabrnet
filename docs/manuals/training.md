# Training configuration
The training configuration associated with a model is stored in a YML file, according to the following specification.
For more examples, see the [ProtoPNet](https://github.com/aiser-team/cabrnet/tree/master/configs/protopnet/cub200/training.yml) and 
[ProtoTree](https://github.com/aiser-team/cabrnet/tree/master/configs/prototree/cub200/training.yml) configuration files.

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
CaBRNet provides a class [OptimizerManager](https://github.com/aiser-team/cabrnet/tree/master/src/cabrnet/utils/optimizers.py) in charge of parsing the configuration file 
and handling optimizers/schedulers during the training process.
```python
from cabrnet.utils.optimizers import OptimizerManager
from cabrnet.utils.parser import load_config
from cabrnet.generic.model import CaBRNet

model = CaBRNet.build_from_config(config_file="<path/to/model/configuration/file.yml") 
training_config = load_config(config_file="<path/to/training/configuration/file.yml>")
optimizer_mngr=OptimizerManager(config_dict=training_config,module=model)
```
For more information on how to build a CaBRNet model from a configuration file, see [here](model.md).
An Optimizer Manager provides four main functions that are used during training:

- `zero_grad()`: Reset all optimizer(s) gradients (before each batch of data)
- `freeze(epoch: int)`: Freeze all relevant model parameters, according to the current `epoch`.
- `optimizer_step(epoch: int)`: Update all relevant optimizers, according to the current `epoch`. 
- `scheduler_step(epoch: int)`: Update all relevant schedulers, according to the current `epoch`.

