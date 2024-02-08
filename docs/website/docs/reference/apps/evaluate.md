---
sidebar_label: evaluate
title: apps.evaluate
---

Declare the necessary functions to create a ProtoLib app to evaluate a ProtoClassifier.

#### create\_parser

```python
def create_parser(parser: ArgumentParser | None = None) -> ArgumentParser
```

Create the argument parser for evaluating a ProtoClassifier.

**Returns**:

  The parser itself.

#### execute

```python
def execute(args: Namespace) -> None
```

Evaluate a ProtoLib model.

**Arguments**:

- `args` - Parsed arguments.

