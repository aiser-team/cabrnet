from argparse import ArgumentParser, RawTextHelpFormatter
from zenodo_get import zenodo_get
import os

file_list = [
    {
        "identifier": "resnet50_inat",
        "description": "ResNet50 pretrained on INaturalist dataset",
        "type": "zenodo",
        "record": "10066893",
        "dir": "pretrained_conv_extractors",
    }
]


def show_file_list() -> str:
    res = ""
    for entry in file_list:
        res += f"\t{entry['identifier']} --> {entry['description']}, downloaded in <output_dir>/{entry['dir']}\n"
    res += "\tall --> everything above"
    return res


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Download datasets and pretrained models", formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "--target",
        "-t",
        type=str,
        required=True,
        metavar="name",
        nargs="+",
        choices=["all"] + [entry["identifier"] for entry in file_list],
        help=f"Select target(s) to download\n{show_file_list()}",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="examples",
        required=False,
        metavar="path/to/root/output/directory",
        help="path to root output directory (default: ./examples)",
    )
    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    # Files to download
    files_to_download = (
        file_list
        if "all" in args.target
        else [file_entry for file_entry in file_list if file_entry["identifier"] in args.target]
    )

    for entry in files_to_download:
        target_path = os.path.join(args.output_dir, entry["dir"])
        if entry["type"] == "zenodo":
            zenodo_get(["-o", target_path, "-r", entry["record"]])


if __name__ == "__main__":
    main()
