# Data configuration
The data configuration associated with a particular experiment is stored in a YML file, according to the following specification.
For more examples, see the [CUB200](https://github.com/aiser-team/cabrnet/blob/main/configs/prototree/cub200/dataset.yml) and 
[Stanford Cars](https://github.com/aiser-team/cabrnet/blob/main/configs/prototree/stanford_cars/dataset.yml) configuration files.

## Configuring datasets 
Each configuration file contains the location of one or several datasets, along with the list of 
preprocessing operations that should be applied to these datasets.

```yaml
<DATASET_NAME>:
  module: <MODULE_NAME> # Name of the module containing the dataset class (e.g. torchvision.datasets) 
  name: <CLASS_NAME> # Class name of the dataset (e.g. ImageFolder)
  num_workers: <NUM> # Optional: number of worker processes for data preprocessing 
  drop_last: <BOOL> # Optional: drop last incomplete batch
  batch_size: <BATCH_SIZE> # Size of each batch
  shuffle: <True | False> # Should data be shuffled (should be True for train_set)
  pin_memory: <BOOL> # Optional: pin dataset to memory
  params: # How to initialize the dataset
    <CLASS_PARAM_1>: <VALUE>
    <CLASS_PARAM_2>: <VALUE>
    transform: <LIST_OF_OPERATIONS> # Optional
    target_transform: <LIST_OF_OPERATIONS> # Optional
...
```
The `params` keyword contains all relevant parameters for building the object of the indicated class name.
For example, parameters for the `StanfordCars` class in the `torchvision.datasets.stanford_cars` module include:

- `root`: Path to the root directory containing the dataset
- `split`: Either `train` or `test`
- `transform`: Image preprocessing function
- `target_transform`: Label preprocessing function

Note that, to be used seamlessly by the main CaBRNet tool (`cabrnet`), 
each file **must** contain the description of a `train_set`, `test_set` and `projection_set` datasets.

## Configuring data preprocessing
The use of `transform` and `target_transform` being consistent over most dataset classes, CaBRNet provides a method for
easily describing the list of operations that should be carried out in both cases.

```yaml
transform | target_transform:
  <OPERATION_NAME_1>:
    module: <OPERATION_MODULE_1> # Optional. If not present, set to torchvision.transforms
    type: <OPERATION_TYPE> # Name of the function inside the module 
    params: # Optional
      <OP_PARAM_1>: <VALUE>
      <OP_PARAM_2>: <VALUE>
  <OPERATION_NAME_2>:
    type : [Compose, RandomOrder, RandomChoice]
    transforms: # Sublist of operation
      <OPERATION_NAME_3>:
        module: <OPERATION_MODULE_3>
        type: <OPERATION_TYPE> 
        params: # Optional
          <OP_PARAM_1>: <VALUE>
          <OP_PARAM_2>: <VALUE>
      <OPERATION_NAME_4>:
        module: <OPERATION_MODULE_4>
        type: <OPERATION_TYPE> 
        params: # Optional
          <OP_PARAM_1>: <VALUE>
          <OP_PARAM_2>: <VALUE>
...
```
CaBRNet supports all functions provided by [torchvision.transforms](https://pytorch.org/vision/stable/transforms.html),
and also authorizes the use of custom modules through the `module` keyword. 
Blocks of operations can be nested through the `Compose`, `RandomOrder` or `RandomChoice` keywords. 

Note that, by default, transform operations are applied in the order given in the configuration file, 
unless they are regrouped into a `RandomOrder` or `RandomChoice` block.

## Creating datasets and dataloaders
CaBRNet provides two main functions for creating datasets and dataloaders from a configuration file:

- `DatasetManager.get_datasets` parses the configuration file and returns a dictionary of entries, indexed by the name of the 
dataset. Each entry is a dictionary containing the following information:
    - `dataset`: A dataset object of the class given in the configuration file, with data preprocessing as specified in 
the `transform` and `target_transform` keywords.
    - `raw_dataset`: A dataset object of the class given in the configuration file, **without** data preprocessing. 
This allows the user to access raw images rather than their preprocessed counterparts.
    - `batch_size`: Size of each batch. Used when building a dataloader from the dataset.
    - `shuffle` (True or False): Whether data should be shuffled. Used when building a dataloader from the dataset.
- `DatasetManager.get_dataloaders` parses the configuration file and returns a dictionary of dataloaders. 
More precisely, each dataset specified in the configuration file produces two dataloaders:
    - `<DATASET_NAME>`: Dataloader returning preprocessed data
    - `<DATASET_NAME>_raw`: Dataloader returning raw data (see above)

## Downloading datasets to reproduce experiments from the state of the art
CaBRNet provides a tool to download and pre-process datasets as described in 
ProtoPNet and ProtoTree.

```bash
python ./tools/download_datasets.py -h
```

```
usage: download_datasets.py [-h] --target name [name ...] [--output-dir path/to/root/output/directory] [--use-segmentation]

Download datasets and perform preprocessing for ProtoTree and ProtoPNet

options:
  -h, --help            show this help message and exit
  --target name [name ...], -t name [name ...]
                        Select target(s) to download
                                CUB_200_2011 --> Caltech-UCSD Birds-200-2011 dataset, downloaded in <output_dir>/CUB_200_2011
                                all --> everything above
  --output-dir path/to/root/output/directory, -o path/to/root/output/directory
                        path to root output directory (default: ./examples)
  --use-segmentation, -s
                        Download segmentation dataset alongside regular dataset
```
