---
sidebar_label: explain_global
title: apps.explain_global
---

#### create\_parser

```python
def create_parser(parser: ArgumentParser | None = None) -> ArgumentParser
```

Create the argument parser for explaining the global behaviour of a ProtoClassifier.

**Returns**:

  The parser itself.

#### execute

```python
def execute(args: Namespace) -> None
```

Explain the global behaviour of a ProtoClassifier

**Arguments**:

- `args` - Parsed arguments.

