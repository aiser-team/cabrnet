import os
import zipfile
from argparse import ArgumentParser, RawTextHelpFormatter

from zenodo_get import zenodo_get

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
    {
        "identifier": "cabrnet_protopnet_cub200_resnet50",
        "description": "ProtoPNet models trained using CaBRNet on CUB200 "
        "(with ResNet50 backbone pretrained on INaturalist)",
        "type": "zenodo",
        "record": "12610876",
        "dir": "trained_models",
        "file": "cabrnet_protopnet_cub200_resnet50.zip",
    },
    {
        "identifier": "cabrnet_protopnet_stanfordcars_resnet50",
        "description": "ProtoPNet models trained using CaBRNet on Stanford Cars "
        "(with ResNet50 backbone pretrained on ImageNet)",
        "type": "zenodo",
        "record": "12610808",
        "dir": "trained_models",
        "file": "cabrnet_protopnet_stanfordcars_resnet50.zip",
    },
    {
        "identifier": "cabrnet_prototree_stanfordcars_resnet50",
        "description": "ProtoTree models trained using CaBRNet on Stanford Cars "
        "(with ResNet50 backbone pretrained on ImageNet)",
        "type": "zenodo",
        "record": "12610556",
        "dir": "trained_models",
        "file": "cabrnet_prototree_stanfordcars_resnet50_depth9.zip",
    },
    {
        "identifier": "cabrnet_prototree_cub200_resnet50_depth9",
        "description": "ProtoTree models trained using CaBRNet on CUB200 "
                       "(with ResNet50 backbone pretrained on INaturalist and depth 9)",
        "type": "zenodo",
        "record": "13305449",
        "dir": "trained_models",
        "file": "prototree_cub200_resnet50_depth9.zip",
    },
    {
        "identifier": "cabrnet_prototree_cub200_resnet50_depth10",
        "description": "ProtoTree models trained using CaBRNet on CUB200 "
                       "(with ResNet50 backbone pretrained on INaturalist and depth 10)",
        "type": "zenodo",
        "record": "13305449",
        "dir": "trained_models",
        "file": "prototree_cub200_resnet50_depth10.zip",
    },
]


def show_file_list() -> str:
    """Shows list of files to download.

    Returns:
        List of files to download.
    """
    res = ""
    for entry in file_list:
        res += f"\t{entry['identifier']} --> {entry['description']}, downloaded in <output_dir>/{entry['dir']}\n"
    res += "\tall --> everything above"
    return res


def create_parser() -> ArgumentParser:
    """Creates parser.

    Returns:
        Parser containing all the arguments.
    """
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
    """Main entry point of the tool."""
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
