import os
import shutil
import tarfile
from argparse import ArgumentParser, RawTextHelpFormatter

import Augmentor
import requests
from loguru import logger
from PIL import Image


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
    parser = ArgumentParser(
        description="Download datasets and perform preprocessing for ProtoTree and ProtoPNet",
        formatter_class=RawTextHelpFormatter,
    )
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
    parser.add_argument(
        "--use-segmentation",
        "-s",
        action="store_true",
        help="Download segmentation dataset alongsite regular dataset",
    )
    return parser


def download_cub(path: str, use_segmentation: bool) -> None:
    """Downloads the CUB200 dataset.

    Args:
        path (str): Path where to download the dataset to.
        use_segmentation (bool): Whether to download the segmentation dataset too.
    """
    ds_url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
    ds_path = os.path.join(path, "CUB-200-2011.tgz")

    if not os.path.exists(ds_path):
        logger.info("Downloading CUB dataset")
        ds_response = requests.get(ds_url, stream=True, allow_redirects=True)
        if ds_response.status_code == 200:
            with open(ds_path, "wb") as f:
                f.write(ds_response.raw.read())
        logger.info("CUB dataset downloaded")
    else:
        logger.info("CUB dataset archive already exists, skipping download")

    if not os.path.exists(os.path.join(path, "CUB_200_2011")):
        logger.info("Extracting dataset archive")
        tar = tarfile.open(ds_path, "r:gz")
        tar.extractall(path=path)
        tar.close()
        if os.path.exists(os.path.join(path, "attributes.txt")):
            os.remove(os.path.join(path, "attributes.txt"))
        logger.info("Dataset archive extracted")
    else:
        logger.info("CUB dataset archive already extracted, skipping extraction")

    if use_segmentation:
        seg_url = "https://data.caltech.edu/records/w9d68-gec53/files/segmentations.tgz"
        seg_path = os.path.join(path, "segmentations.tgz")

        if not os.path.exists(seg_path):
            logger.info("Downloading CUB segmentations")
            seg_response = requests.get(seg_url, stream=True, allow_redirects=True)
            if seg_response.status_code == 200:
                with open(seg_path, "wb") as f:
                    f.write(seg_response.raw.read())
            logger.info("CUB segmentations downloaded")
        else:
            logger.info("CUB segmentations archive already exists, skipping download")

        if not os.path.exists(os.path.join(path, "segmentations")):
            logger.info("Extracting segmentations archive")
            tar = tarfile.open(seg_path, "r:gz")
            tar.extractall(path=os.path.join(path, "CUB_200_2011"))
            tar.close()
            logger.info("Segmentations archive extracted")
        else:
            logger.info("CUB segmentations archive already extracted, skipping extraction")


def preprocess_cub(path: str) -> None:
    """Preprocessing function to create proper datasets used by ProtoTree and ProtoPNet.

    Args:
        path (str): Path where the dataset is located.
    """
    # ProtoTree
    path_images = os.path.join(path, "images.txt")
    path_split = os.path.join(path, "train_test_split.txt")
    train_crop_path = os.path.join(path, "dataset/train_crop/")
    test_crop_path = os.path.join(path, "dataset/test_crop/")
    bbox_path = os.path.join(path, "bounding_boxes.txt")

    use_segmentation = os.path.isdir(os.path.join(path, "segmentations"))
    logger.info("Using segmentation:", use_segmentation)
    train_seg_path = os.path.join(path, "dataset/train_crop_seg/") if use_segmentation else ""
    test_seg_path = os.path.join(path, "dataset/test_crop_seg/") if use_segmentation else ""

    images = []
    with open(path_images, "r") as f:
        for line in f:
            images.append(list(line.strip("\n").split(",")))
    split = []
    with open(path_split, "r") as f_:
        for line in f_:
            split.append(list(line.strip("\n").split(",")))

    bboxes = dict()
    with open(bbox_path, "r") as bf:
        for line in bf:
            id, x, y, w, h = tuple(map(float, line.split(" ")))
            bboxes[int(id)] = (x, y, w, h)

    num = len(images)
    for k in range(num):
        id, fn = images[k][0].split(" ")
        id = int(id)
        file_name = fn.split("/")[0]
        if int(split[k][0][-1]) == 1:
            dst_dir = train_crop_path
            dst_seg_dir = train_seg_path
        else:
            dst_dir = test_crop_path
            dst_seg_dir = test_seg_path

        if not os.path.isdir(os.path.join(dst_dir, file_name)):
            os.makedirs(os.path.join(dst_dir, file_name))
        if use_segmentation and not os.path.isdir(os.path.join(dst_seg_dir, file_name)):
            os.makedirs(os.path.join(dst_seg_dir, file_name))
        img = Image.open(os.path.join(os.path.join(path, "images"), images[k][0].split(" ")[1])).convert("RGB")
        x, y, w, h = bboxes[id]
        cropped_img = img.crop((x, y, x + w, y + h))
        cropped_img.save(os.path.join(os.path.join(dst_dir, file_name), images[k][0].split(" ")[1].split("/")[1]))
        if use_segmentation:
            seg_path = os.path.splitext(images[k][0].split(" ")[1])[0]
            seg_path = os.path.join(path, "segmentations", seg_path + ".png")
            seg_img = Image.open(seg_path).convert("RGB")
            cropped_img = seg_img.crop((x, y, x + w, y + h))
            cropped_img.save(
                os.path.join(os.path.join(dst_seg_dir, file_name), images[k][0].split(" ")[1].split("/")[1])
            )
        logger.info("%s" % images[k][0].split(" ")[1].split("/")[1])

    train_full_path = os.path.join(path, "dataset/train_full/")
    train_seg_path = os.path.join(path, "dataset/train_full_seg/") if use_segmentation else ""
    train_corners_path = os.path.join(path, "dataset/train_corners/")
    train_seg_corners_path = os.path.join(path, "dataset/train_corners_seg/")
    test_full_path = os.path.join(path, "dataset/test_full/")
    test_seg_path = os.path.join(path, "dataset/test_full_seg/") if use_segmentation else ""

    num = len(images)
    for k in range(num):
        id, fn = images[k][0].split(" ")
        id = int(id)
        file_name = fn.split("/")[0]
        if int(split[k][0][-1]) == 1:
            if not os.path.isdir(train_full_path + file_name):
                os.makedirs(os.path.join(train_full_path, file_name))
            if use_segmentation and not os.path.isdir(os.path.join(train_seg_path, file_name)):
                os.makedirs(os.path.join(train_seg_path, file_name))
            shutil.copy(
                path + "images/" + images[k][0].split(" ")[1],
                os.path.join(os.path.join(train_full_path, file_name), images[k][0].split(" ")[1].split("/")[1]),
            )
            if use_segmentation:
                seg_fname = os.path.splitext(images[k][0].split(" ")[1])[0] + ".png"
                shutil.copy(
                    os.path.join(path, "segmentations", seg_fname),
                    os.path.join(train_seg_path, file_name, seg_fname.split("/")[1]),
                )
            if not os.path.isdir(train_corners_path + file_name):
                os.makedirs(os.path.join(train_corners_path, file_name))
            if use_segmentation and not os.path.isdir(train_seg_corners_path + file_name):
                os.makedirs(os.path.join(train_seg_corners_path, file_name))

            if use_segmentation:
                seg_fname = os.path.splitext(images[k][0].split(" ")[1])[0] + ".png"
                os.path.join(path, "segmentations", seg_fname)

            def corners_img(img_path, dir_path, suffix):
                img = Image.open(img_path).convert("RGB")
                x, y, w, h = bboxes[id]
                width, height = img.size
                hmargin = int(0.1 * h)
                wmargin = int(0.1 * w)

                cropped_img = img.crop((0, 0, min(x + w + wmargin, width), min(y + h + hmargin, height)))
                cropped_img.save(os.path.join(dir_path, file_name, "upperleft_" + suffix))
                cropped_img = img.crop((0, max(y - hmargin, 0), min(x + w + wmargin, width), height))
                cropped_img.save(os.path.join(dir_path, file_name, "lowerleft_" + suffix))
                cropped_img = img.crop((max(x - wmargin, 0), 0, width, min(y + h + hmargin, height)))
                cropped_img.save(os.path.join(dir_path, file_name, "upperright_" + suffix))
                cropped_img = img.crop(((max(x - wmargin, 0), max(y - hmargin, 0), width, height)))
                cropped_img.save(os.path.join(dir_path, file_name, "lowerright_" + suffix))
                img.save(os.path.join(dir_path, file_name, "normal_" + suffix))

            img_path = os.path.join(os.path.join(path, "images"), images[k][0].split(" ")[1])
            suffix = images[k][0].split(" ")[1].split("/")[1]
            corners_img(img_path, train_corners_path, suffix)
            if use_segmentation:
                seg_fname = os.path.splitext(images[k][0].split(" ")[1])[0] + ".png"
                img_path = os.path.join(path, "segmentations", seg_fname)
                corners_img(img_path, train_seg_corners_path, suffix)

            logger.info("%s" % images[k][0].split(" ")[1].split("/")[1])
        else:
            if not os.path.isdir(os.path.join(test_full_path, file_name)):
                os.makedirs(os.path.join(test_full_path, file_name))
            if use_segmentation and not os.path.isdir(os.path.join(test_seg_path, file_name)):
                os.makedirs(os.path.join(test_seg_path, file_name))
            shutil.copy(
                path + "images/" + images[k][0].split(" ")[1],
                os.path.join(test_full_path, file_name, images[k][0].split(" ")[1].split("/")[1]),
            )
            if use_segmentation:
                seg_fname = os.path.splitext(images[k][0].split(" ")[1])[0] + ".png"
                shutil.copy(
                    os.path.join(path, "segmentations", seg_fname),
                    os.path.join(test_seg_path, file_name, seg_fname.split("/")[1]),
                )
            logger.info("%s" % images[k][0].split(" ")[1].split("/")[1])

    # ProtoPNet
    train_aug_path = "../../train_crop_augmented/"
    os.makedirs("data/CUB_200_2011/dataset/train_crop_augmented", exist_ok=True)
    class_dirs = [os.path.join(train_crop_path, dir) for dir in next(os.walk(train_crop_path))[1]]
    class_aug_dirs = [os.path.join(train_aug_path, dir) for dir in next(os.walk(train_crop_path))[1]]

    for class_dir, class_aug_dir in zip(class_dirs, class_aug_dirs):
        # rotation
        p = Augmentor.Pipeline(source_directory=class_dir, output_directory=class_aug_dir)
        p.rotate(
            probability=1, max_left_rotation=10, max_right_rotation=10  # Reduce angle to avoid errors in Augmentor
        )
        p.flip_left_right(probability=0.5)
        for _ in range(10):
            p.sample(0, multi_threaded=False)  # Use single thread for reproducibility
        del p
        # skew
        p = Augmentor.Pipeline(source_directory=class_dir, output_directory=class_aug_dir)
        p.skew(probability=1, magnitude=1)  # max 45 degrees
        p.flip_left_right(probability=0.5)
        for _ in range(10):
            p.sample(0, multi_threaded=False)  # Use single thread for reproducibility
        del p
        # shear
        p = Augmentor.Pipeline(source_directory=class_dir, output_directory=class_aug_dir)
        p.shear(probability=1, max_shear_left=10, max_shear_right=10)
        p.flip_left_right(probability=0.5)
        for _ in range(10):
            p.sample(0, multi_threaded=False)  # Use single thread for reproducibility
        del p


file_list = [
    {
        "identifier": "CUB_200_2011",
        "description": "Caltech-UCSD Birds-200-2011 dataset",
        "dir": "CUB_200_2011",
        "download_fn": download_cub,
        "preprocess_fn": preprocess_cub,
    }
]


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

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    for entry in files_to_download:
        entry["download_fn"](output_dir, args.use_segmentation)
        if entry.get("preprocess_fn") is not None:
            entry["preprocess_fn"](os.path.join(output_dir, entry["dir"], ""))


if __name__ == "__main__":
    main()
