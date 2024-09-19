## Adding new applications

To add a new application to the CaBRNet main tool, simply add a new file
`<my_application_name.py>` into the directory `<src/apps>`. This file should
contain:

1. A string `description` containing the purpose of the application.
2. A method `create_parser` adding the application arguments to an existing
   parser (or creating one if necessary)
3. A method `execute` taking the parsed arguments and executing the application
   code.

Here is an example of a new application:
```python
from argparse import ArgumentParser, Namespace

description = "my new awesome CaBRNet application"


def create_parser(parser: ArgumentParser | None = None) -> ArgumentParser:
    if parser is None:
        parser = ArgumentParser(description)
    parser.add_argument(
        "--message",
        type=str,
        required=True,
        metavar="<message>",
        help="Message to be printed",
    )
    return parser


def execute(args: Namespace) -> None:
    print(args.message)

```

