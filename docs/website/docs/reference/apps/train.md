---
sidebar_label: train
title: apps.train
---

Declare the necessary functions to create a ProtoLib app to train a ProtoClassifier.

#### create\_parser

```python
def create_parser(parser: ArgumentParser | None = None) -> ArgumentParser
```

Create the argument parser for training a ProtoClassifier.

**Returns**:

  The parser itself.

#### execute

```python
def execute(args: Namespace) -> None
```

Create Protolib model, then train it.

**Arguments**:

- `args` - Parsed arguments.

