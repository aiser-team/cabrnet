from argparse import ArgumentParser, RawTextHelpFormatter
from zenodo_get import zenodo_get
import zipfile
import os

file_list = [
    {
        "identifier": "resnet50_inat",
        "description": "ResNet50 pretrained on INaturalist dataset",
        "type": "zenodo",
        "record": "10066893",
        "dir": "examples/pretrained_conv_extractors",
        "file": "resnet50_inat.pth",
    },
    {
        "identifier": "legacy_models",
        "description": "Legacy models trained using ProtoPNet and ProtoTree on CUB200",
        "type": "zenodo",
        "record": "11284813",
        "dir": "src/legacy/compatibility_tests",
        "file": "cabrnet_legacy_models.zip",
    },
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
        target_path = entry["dir"]
        if entry["type"] == "zenodo":
            zenodo_get(["-o", target_path, "-r", entry["record"]])
            if entry["file"].endswith(".zip"):
                filepath = os.path.join(target_path, entry["file"])
                with zipfile.ZipFile(filepath, "r") as zip_ref:
                    zip_ref.extractall(target_path)


if __name__ == "__main__":
    main()
