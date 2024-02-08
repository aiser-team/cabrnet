---
sidebar_label: explain
title: apps.explain
---

Declare the necessary functions to create a ProtoLib app to explain a classification result.

#### create\_parser

```python
def create_parser(parser: ArgumentParser | None = None) -> ArgumentParser
```

Create the argument parser for explaining the decision of a ProtoClassifier.

**Returns**:

  The parser itself.

#### execute

```python
def execute(args: Namespace) -> None
```

Explain the decision of a ProtoLib model.

**Arguments**:

- `args` - Parsed arguments.

