---
sidebar_label: parser
title: cabrnet.utils.parser
---

This file contains all the necessary tools to parse the various config files.

#### load\_config

```python
def load_config(config_file: str) -> dict
```

Load a configuration file for CaBRNet.

**Arguments**:

- `config_file` - Path to the configuration file.
  

**Returns**:

  The properly loaded config file.
  

**Raises**:

- `ValueError` - The config file is in an unsupported file format.

#### create\_training\_parser

```python
def create_training_parser(
        parser: argparse.ArgumentParser | None = None
) -> argparse.ArgumentParser
```

Create the argument parser for CaBRNet training configuration.

**Arguments**:

- `parser` - Existing parser (if any)

**Returns**:

  The parser itself.

