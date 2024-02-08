---
sidebar_label: data
title: protolib.utils.data
---

This file holds all the necessary functions to create datasets and dataloaders from configuration files.

#### create\_dataset\_parser

```python
def create_dataset_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser
```

Create the argument parser for ProtoLib datasets.

**Arguments**:

- `parser` - Existing parser (if any)

**Returns**:

  The parser itself.

#### get\_transform

```python
def get_transform(trans_config: dict[str, Any]) -> Callable
```

Build a data transformation from a dictionary.

**Arguments**:

- `trans_config` - Transformation configuration

**Returns**:

  transformation

**Raises**:

  ValueError whenever the configuration is incorrect.

#### get\_datasets

```python
def get_datasets(config_file: str) -> dict[str, dict[str, Dataset | int | bool]]
```

Load datasets from yaml configuration file.

**Arguments**:

- `config_file` - path to configuration file
  

**Returns**:

  dictionary of datasets with their respective batch size and shuffle property
  

**Raises**:

  ValueError whenever a dataset could not be loaded

#### get\_dataloaders

```python
def get_dataloaders(config_file: str) -> dict[str, DataLoader]
```

Create dataloaders from yaml configuration file.

**Arguments**:

- `config_file` - path to configuration file
  

**Returns**:

  dictionary of dataloaders
  

**Raises**:

  ValueError whenever a dataset could not be loaded or a parameter is invalid

#### get\_dataset\_transform

```python
def get_dataset_transform(config_file: str, dataset: str = "test_set") -> Callable | None
```

Return transform function associated with a given dataset

**Arguments**:

- `config_file` - path to configuration file
- `dataset` - name of target dataset
  

**Returns**:

  transform function if any
  

**Raises**:

  ValueError whenever the configuration is incorrect.

