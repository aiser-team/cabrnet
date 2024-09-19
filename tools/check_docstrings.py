import ast
import os
import re
from argparse import ArgumentParser
from typing import Any

from loguru import logger


def create_parser() -> ArgumentParser:
    r"""Creates the argument parser for checking docstrings.

    Returns:
        The parser itself.
    """
    parser = ArgumentParser("Check docstrings")
    parser.add_argument(
        "-d", "--dir", type=str, required=True, metavar="/path/to/dir", help="path to a source directory"
    )
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
                logger.error(f"Missing docstring for class '{class_name}' " f"({filename}:{body_content.lineno})")
                complies = False
            else:
                attr_location = re.search("\n[\s]*Attributes:", module_docstring)  # type: ignore
                if attr_location is None:
                    logger.warning(
                        f"Missing 'Attributes' field in docstring for class '{class_name}' "
                        f"({filename}:{body_content.lineno})"
                    )
                else:
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
                logger.error(f"Missing docstring for function '{function_name}' " f"({filename}:{body_content.lineno})")
                complies = False
                continue

            # Get number of true arguments
            num_args = len(body_content.args.args)
            if num_args > 0 and body_content.args.args[0].arg == "self":
                num_args -= 1

            args_location = re.search("\n[\s]*Args:", function_docstring)  # type: ignore
            if args_location is None and num_args > 0:
                logger.error(
                    f"Missing 'Args' keyword in function '{function_name}' description "
                    f"({filename}:{body_content.lineno})"
                )
                complies = False
                continue

            return_location = re.search("[\s]*Returns", function_docstring)  # type: ignore

            if (
                return_location is None
                and body_content.returns is not None
                and hasattr(body_content.returns, "value")
                and body_content.returns.value is not None  # type: ignore
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
                function_desc = function_docstring[: args_location.start()]  # type: ignore
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
                return_docstring = (
                    function_docstring[return_location.end() :].lstrip(":").lstrip().rstrip()
                    if return_location.start() > 0
                    else function_docstring.rstrip()
                )
                if return_docstring == "":
                    logger.error(
                        f"Empty docstring for return value "
                        f"in function '{function_name}' ({filename}:{body_content.lineno})"
                    )
                    complies = False
                else:
                    if not return_docstring[0].isupper():
                        logger.error(
                            f"Missing uppercase in return value description "
                            f"for function '{function_name}': {return_docstring} ({filename}:{body_content.lineno})"
                        )
                        complies = False
                    if not return_docstring.endswith((".", "?", "!")):
                        logger.error(
                            f"Missing dot in return value description "
                            f"for function '{function_name}': {return_docstring} ({filename}:{body_content.lineno})"
                        )
                        complies = False
            if num_args == 0:
                continue
            args_docstring = function_docstring[args_location.end() :]  # type: ignore
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
                    next_arg_docstring_location = re.search(r"\n[\s]*" + re.escape(next_arg_name) + " ", args_docstring)
                    if next_arg_docstring_location is None:
                        arg_docstring = args_docstring[arg_docstring_location:]
                    else:
                        next_arg_docstring_location = next_arg_docstring_location.start()
                        arg_docstring = args_docstring[arg_docstring_location:next_arg_docstring_location]
                arg_docstring = arg_docstring.lstrip().rstrip()
                optional_arg = arg_index >= len(function_args.args) - num_default

                arg_desc = re.search(r"^" + re.escape(arg_name) + r" \([\w|\s|,|\[|\]]+\):", arg_docstring)
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
                        re.search(r"^" + re.escape(arg_name) + r" \([\w|\s|,|\[|\]]+, optional\):", arg_docstring)
                        is None
                    ):
                        logger.error(
                            f"Missing 'optional' keyword in docstring for argument '{arg_name}' "
                            f"in function '{function_name}': {arg_docstring} ({filename}:{body_content.lineno})"
                        )
                        complies = False
                        continue
                    if re.search("Default: [^\.]+\.", arg_desc) is None:  # type: ignore
                        logger.error(
                            f"Missing default value in argument '{arg_name}' description "
                            f"in function '{function_name}': {arg_desc} ({filename}:{body_content.lineno})"
                        )
                        complies = False
    return complies


def check_docstrings(dir_path: str, ignore_imperative_warnings: bool, quiet: bool) -> None:
    r"""Checks the docstring format of all python files inside the directory *dir_path*.

    Args:
        dir_path (str): Path to a source directory.
        ignore_imperative_warnings (bool): If True, does not display warnings related to usage of imperative.
        quiet (bool): If True, does not display success messages.

    Raises:
        DocStringFormatError when DocString format does not comply with policy.
    """

    def check_file(filepath: str):
        with open(filepath, "r") as fin:
            if not parse_ast(
                ast_module=ast.parse(fin.read()),
                filename=filepath,
                ignore_imperative_warnings=ignore_imperative_warnings,
            ):
                logger.error(f"Errors found in {filepath}")
            elif not quiet:
                logger.success(f"File {filepath} complies with docstring policy")

    for filename in os.listdir(dir_path):
        path = os.path.join(dir_path, filename)
        if os.path.isfile(path) and path.endswith(".py"):
            check_file(filepath=path)
        elif os.path.isdir(path):
            # Recursive call
            check_docstrings(
                dir_path=path,
                ignore_imperative_warnings=ignore_imperative_warnings,
                quiet=quiet,
            )


def main():
    r"""Checks the docstring format of all files inside a given directory."""
    parser = create_parser()
    args = parser.parse_args()
    check_docstrings(dir_path=args.dir, ignore_imperative_warnings=args.ignore_imperative_warnings, quiet=args.quiet)


if __name__ == "__main__":
    main()
