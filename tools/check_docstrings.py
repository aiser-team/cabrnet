import ast
import os
import re
from argparse import ArgumentParser
from typing import Any

from loguru import logger


ATTRIBUTES_HEADER = re.compile(r"\n\s*Attributes:\s*(?:\n|$)")
RETURNS_HEADER = re.compile(r"\n\s*Returns:\s*(?:\n|$)")
TYPE_PATTERN = r"[\w., \[\]|]+"


def has_class_attributes(class_node: ast.ClassDef) -> bool:
    r"""Checks whether a class declares attributes in its body.

    Args:
        class_node (ast.ClassDef): Class definition to inspect.

    Returns:
        True if the class declares an attribute.
    """
    return any(isinstance(statement, ast.AnnAssign) for statement in class_node.body)


def is_property(function_node: ast.FunctionDef) -> bool:
    r"""Checks whether a function is decorated as a property.

    Args:
        function_node (FunctionDef): Function definition to inspect.

    Returns:
        True if the function uses the built-in property decorator.
    """
    return any(
        isinstance(decorator, ast.Name) and decorator.id == "property" for decorator in function_node.decorator_list
    )


def create_parser() -> ArgumentParser:
    r"""Creates the argument parser for checking docstrings.

    Returns:
        The parser itself.
    """
    parser = ArgumentParser("Check docstrings")
    parser.add_argument("-d", "--dir", type=str, metavar="/path/to/dir", help="path to a source directory")
    parser.add_argument("files", nargs="*", metavar="FILE", help="Python files to check")
    parser.add_argument(
        "--ignore-imperative-warnings",
        action="store_true",
        help="ignore warnings related to usage of imperative (improves readability)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="do not show success messages (improves readability)",
    )
    return parser


def parse_ast(ast_module: Any, filename: str, ignore_imperative_warnings: bool) -> bool:
    r"""Parses an AST module, looking for docstrings.

    Args:
        ast_module (AST object): Parsed module.
        filename (str): Source filename.
        ignore_imperative_warnings (bool): If True, does not display warnings related to usage of imperative.

    Returns:
        True if and only if entire file complies with docstring policy.
    """
    complies = True
    for body_content in ast_module.body:
        if isinstance(body_content, ast.ClassDef):
            module_docstring = ast.get_docstring(body_content)
            class_name = body_content.name
            if not module_docstring:
                logger.error(f"Missing docstring for class '{class_name}' ({filename}:{body_content.lineno})")
                complies = False
            else:
                attr_location = ATTRIBUTES_HEADER.search(module_docstring)
                if attr_location is None and has_class_attributes(body_content):
                    logger.warning(
                        f"Missing 'Attributes' field in docstring for class '{class_name}' "
                        f"({filename}:{body_content.lineno})"
                    )
                elif attr_location is not None:
                    module_docstring = module_docstring[: attr_location.start()]
                if not module_docstring[0].isupper():
                    logger.error(
                        f"Missing uppercase in class '{class_name}' description: {module_docstring} "
                        f"({filename}:{body_content.lineno})"
                    )
                    complies = False
                if not module_docstring.rstrip().endswith((".", "?", "!")):
                    logger.error(
                        f"Missing dot in function '{class_name}' description: {module_docstring} "
                        f"({filename}:{body_content.lineno})"
                    )
                    complies = False
            complies = complies and parse_ast(body_content, filename, ignore_imperative_warnings)
        elif isinstance(body_content, ast.FunctionDef):
            # Get function docstring
            function_docstring = ast.get_docstring(body_content)
            function_name = body_content.name
            if not function_docstring:
                logger.error(f"Missing docstring for function '{function_name}' ({filename}:{body_content.lineno})")
                complies = False
                continue

            # Get number of true arguments
            num_args = len(body_content.args.args)
            if num_args > 0 and body_content.args.args[0].arg == "self":
                num_args -= 1

            args_location = re.search(r"\n[\s]*Args:", function_docstring)
            if num_args > 0:
                if args_location is None:
                    logger.error(
                        f"Missing 'Args' keyword in function '{function_name}' description "
                        f"({filename}:{body_content.lineno})"
                    )
                    complies = False
                    continue
                function_desc = function_docstring[: args_location.start()]

            return_location = RETURNS_HEADER.search(function_docstring)

            if (
                return_location is None
                and body_content.returns is not None
                and not (isinstance(body_content.returns, ast.Constant) and body_content.returns.value is None)
                and not is_property(body_content)
            ):
                logger.error(
                    f"Missing docstring for return value in function '{function_name}' "
                    f"({filename}:{body_content.lineno})"
                )
                complies = False

            if num_args == 0:
                if return_location is None:
                    function_desc = function_docstring
                elif return_location.start() == 0:
                    # Special case of a function with no arguments, where the function description also describes the
                    # returned value
                    function_desc = function_docstring
                else:
                    function_desc = function_docstring[: return_location.start()]
            else:
                function_desc = function_docstring[: args_location.start()]
            function_desc = function_desc.rstrip().lstrip()

            # Check usage of 3rd person and uppercase
            first_word = function_desc.split(" ")[0]
            if not first_word.endswith("s") and not ignore_imperative_warnings:
                logger.warning(
                    f"Possible incorrect spelling (use 3rd person) in function '{function_name}': {function_desc} "
                    f"({filename}:{body_content.lineno})"
                )
            if not first_word[0].isupper():
                logger.error(
                    f"Missing uppercase in function '{function_name}' description: {function_desc} "
                    f"({filename}:{body_content.lineno})"
                )
                complies = False
            if not function_desc.rstrip().endswith((".", "?", "!")):
                logger.error(
                    f"Missing dot in function '{function_name}' description: {function_desc} "
                    f"({filename}:{body_content.lineno})"
                )
                complies = False

            if return_location:
                return_docstring = function_docstring[return_location.end() :].lstrip().rstrip()
                if return_docstring == "":
                    logger.error(
                        f"Empty docstring for return value "
                        f"in function '{function_name}' ({filename}:{body_content.lineno})"
                    )
                    complies = False
                else:
                    if not return_docstring.endswith((".", "?", "!")):
                        logger.error(
                            f"Missing dot in return value description "
                            f"for function '{function_name}': {return_docstring} ({filename}:{body_content.lineno})"
                        )
                        complies = False
            if num_args == 0:
                continue
            args_docstring = function_docstring[args_location.end() :]
            function_args = body_content.args
            num_default = len(function_args.defaults)
            for arg_index, arg in enumerate(function_args.args):
                arg_name = arg.arg
                next_arg_name = (
                    function_args.args[arg_index + 1].arg if arg_index < len(function_args.args) - 1 else "Returns"
                )
                if arg_name == "self":
                    continue

                # Seek argument in docstring
                arg_docstring_location = re.search(r"\n[\s]*" + re.escape(arg_name) + " ", args_docstring)
                if arg_docstring_location is None:
                    logger.error(
                        f"Missing docstring for argument '{arg_name}' "
                        f"in function '{function_name}' ({filename}:{body_content.lineno})"
                    )
                    complies = False
                    continue
                arg_docstring_location = arg_docstring_location.start()
                if next_arg_name != "Returns":
                    next_arg_docstring_location = re.search(r"\n[\s]*" + re.escape(next_arg_name) + " ", args_docstring)
                    if next_arg_docstring_location is None:
                        logger.error(
                            f"Missing docstring for argument '{next_arg_name}' "
                            f"in function '{function_name}' ({filename}:{body_content.lineno})"
                        )
                        complies = False
                        continue
                    next_arg_docstring_location = next_arg_docstring_location.start()
                    arg_docstring = args_docstring[arg_docstring_location:next_arg_docstring_location]
                else:
                    next_arg_docstring_location = RETURNS_HEADER.search(args_docstring)
                    if next_arg_docstring_location is None:
                        arg_docstring = args_docstring[arg_docstring_location:]
                    else:
                        next_arg_docstring_location = next_arg_docstring_location.start()
                        arg_docstring = args_docstring[arg_docstring_location:next_arg_docstring_location]
                arg_docstring = arg_docstring.lstrip().rstrip()
                optional_arg = arg_index >= len(function_args.args) - num_default

                arg_desc = re.search(r"^" + re.escape(arg_name) + r" \(" + TYPE_PATTERN + r"\):", arg_docstring)
                if arg_desc is None:
                    logger.error(
                        f"Missing type for argument '{arg_name}' "
                        f"in function '{function_name}': {arg_docstring} ({filename}:{body_content.lineno})"
                    )
                    complies = False
                    continue
                arg_desc = arg_docstring[arg_desc.end() :].lstrip().rstrip()
                if not arg_desc[0].isupper():
                    logger.error(
                        f"Missing uppercase in first word of argument '{arg_name}' description "
                        f"in function '{function_name}': {arg_desc} ({filename}:{body_content.lineno})"
                    )
                    complies = False
                if not arg_desc.endswith((".", "?", "!")):
                    logger.error(
                        f"Missing dot in argument '{arg_name}' description "
                        f"in function '{function_name}': {arg_desc} ({filename}:{body_content.lineno})"
                    )
                    complies = False
                if optional_arg:
                    if (
                        re.search(
                            r"^" + re.escape(arg_name) + r" \(" + TYPE_PATTERN + r", optional\):",
                            arg_docstring,
                        )
                        is None
                    ):
                        logger.error(
                            f"Missing 'optional' keyword in docstring for argument '{arg_name}' "
                            f"in function '{function_name}': {arg_docstring} ({filename}:{body_content.lineno})"
                        )
                        complies = False
                        continue
                    if re.search(r"Default: [^.]+\.", arg_desc) is None:
                        logger.error(
                            f"Missing default value in argument '{arg_name}' description "
                            f"in function '{function_name}': {arg_desc} ({filename}:{body_content.lineno})"
                        )
                        complies = False
    return complies


def check_docstring_file(filepath: str, ignore_imperative_warnings: bool, quiet: bool) -> bool:
    r"""Checks the docstring format of a Python file.

    Args:
        filepath (str): Path to the Python file.
        ignore_imperative_warnings (bool): If True, does not display warnings related to usage of imperative.
        quiet (bool): If True, does not display success messages.

    Returns:
        True if and only if the checked file complies with the docstring policy.
    """
    with open(filepath, "r") as fin:
        if not parse_ast(
            ast_module=ast.parse(fin.read()),
            filename=filepath,
            ignore_imperative_warnings=ignore_imperative_warnings,
        ):
            logger.error(f"Errors found in {filepath}")
            return False
        if not quiet:
            logger.success(f"File {filepath} complies with docstring policy")
    return True


def check_docstrings(dir_path: str, ignore_imperative_warnings: bool, quiet: bool) -> bool:
    r"""Checks the docstring format of all Python files inside a directory.

    Args:
        dir_path (str): Path to a source directory.
        ignore_imperative_warnings (bool): If True, does not display warnings related to usage of imperative.
        quiet (bool): If True, does not display success messages.

    Returns:
        True if and only if all checked files comply with the docstring policy.
    """

    complies = True

    for filename in os.listdir(dir_path):
        path = os.path.join(dir_path, filename)
        if os.path.isfile(path) and path.endswith(".py"):
            complies = (
                check_docstring_file(filepath=path, ignore_imperative_warnings=ignore_imperative_warnings, quiet=quiet)
                and complies
            )
        elif os.path.isdir(path):
            # Recursive call
            complies = (
                check_docstrings(
                    dir_path=path,
                    ignore_imperative_warnings=ignore_imperative_warnings,
                    quiet=quiet,
                )
                and complies
            )
    return complies


def main() -> None:
    r"""Checks the docstring format of all files inside a given directory."""
    parser = create_parser()
    args = parser.parse_args()
    if args.dir is None and not args.files:
        parser.error("one of --dir or FILE is required")
    if args.dir is not None and args.files:
        parser.error("--dir cannot be combined with FILE")

    complies = (
        check_docstrings(
            dir_path=args.dir, ignore_imperative_warnings=args.ignore_imperative_warnings, quiet=args.quiet
        )
        if args.dir is not None
        else all(
            check_docstring_file(
                filepath=filepath, ignore_imperative_warnings=args.ignore_imperative_warnings, quiet=args.quiet
            )
            for filepath in args.files
        )
    )
    if not complies:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
