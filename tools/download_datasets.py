import os
import shutil
import tarfile
import xml.etree.ElementTree as ET
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from urllib.parse import urljoin

import Augmentor
import requests
import scipy.io
from loguru import logger
from PIL import Image


def show_file_list() -> str:
    """Shows list of files to download.

    Returns:
        List of files to download.
    """
    res = ""
    for entry in FILE_LIST:
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
        choices=["all"] + [entry["identifier"] for entry in FILE_LIST],
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
        help="Download segmentation dataset alongside regular dataset",
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
    """Preprocesses data to create proper datasets used by ProtoTree and ProtoPNet.

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
            probability=1,
            max_left_rotation=10,
            max_right_rotation=10,  # Reduce angle to avoid errors in Augmentor
        )
        p.flip_left_right(probability=0.5)
        for _ in range(10):
            p.sample(0, multi_threaded=False)  # Use single thread for reproducibility
        del p
        # skew (max 45 degrees)
        p = Augmentor.Pipeline(source_directory=class_dir, output_directory=class_aug_dir)
        p.skew(probability=1, magnitude=0.2)  # type: ignore
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


def download_dogs(path: str, use_segmentation: bool) -> None:
    """Downloads the Stanford Dogs dataset.

    Args:
        path (str): Path where to download the dataset to.
        use_segmentation (bool): Ignored, as segmentation is not available for this dataset.
    """
    dataset_dir = Path(path) / "stanford_dogs"
    os.makedirs(dataset_dir, exist_ok=True)

    # URL for the Stanford Dogs dataset
    url_dir = "http://vision.stanford.edu/aditya86/ImageNetDogs/"
    filenames = ["images.tar", "annotation.tar", "lists.tar"]

    # Download helper
    def download_file(url, dest):
        if not dest.exists():
            logger.info(f"Downloading {dest}...")
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(dest, "wb") as f:
                    f.write(response.raw.read())
            logger.info(f"Downloaded {dest}")
        else:
            logger.info(f"{os.path.basename(dest)} already exists.")

    for filename in filenames:
        download_file(urljoin(url_dir, filename), Path(dataset_dir) / filename)

    # Extract files
    if not os.path.exists(os.path.join(dataset_dir, "train_list.mat")):
        for filename in filenames:
            tar_path = Path(dataset_dir) / filename
            logger.info(f"Extracting {tar_path}...")
            with tarfile.open(tar_path) as tar:
                # Extract specifically into the dataset_dir
                tar.extractall(path=dataset_dir)
        logger.info("Stanford Dogs dataset extracted.")
    else:
        logger.info("Stanford Dogs dataset already extracted.")

    if use_segmentation:
        logger.warning("Segmentation dataset is not available for Stanford Dogs.")


def preprocess_dogs(path: str) -> None:
    """Preprocesses Stanford Dogs dataset for ProtoTree/ProtoPNet (Cropping).

    Args:
        path (str): Path where the dataset is located.
    """
    # Define paths
    images_root = os.path.join(path, "Images")
    annotations_root = os.path.join(path, "Annotation")

    # Stanford dogs uses .mat files for splits
    train_list_path = os.path.join(path, "train_list.mat")
    test_list_path = os.path.join(path, "test_list.mat")

    logger.info(f"Train list path: {train_list_path}")
    logger.info(f"Test list path: {test_list_path}")

    if not os.path.exists(train_list_path) or not os.path.exists(test_list_path):
        logger.error("Train/Test lists not found. content of lists.tar might be missing.")
        return

    # Output directories
    train_full_path = os.path.join(path, "dataset/train_full/")
    test_full_path = os.path.join(path, "dataset/test_full/")
    train_crop_path = os.path.join(path, "dataset/train_crop/")
    test_crop_path = os.path.join(path, "dataset/test_crop/")

    # Helper to process splits
    def process_split(mat_path, dst_dir, crop_bbox: bool = True):
        mat = scipy.io.loadmat(mat_path)
        # file_list is an array of arrays of strings. shape (N, 1)
        file_list = [f[0][0] for f in mat["file_list"]]

        for file_name in file_list:
            # file_name example: 'n02085620-Chihuahua/n02085620_10976.jpg'
            class_dir_name = os.path.dirname(file_name)
            base_name = os.path.splitext(os.path.basename(file_name))[0]

            # Paths
            src_img_path = os.path.join(images_root, file_name)
            xml_path = os.path.join(annotations_root, class_dir_name, base_name)  # Annotation has no extension

            if not os.path.exists(src_img_path):
                continue

            # Create destination folder
            save_dir = os.path.join(dst_dir, class_dir_name)
            os.makedirs(save_dir, exist_ok=True)

            # Load and Crop
            img = Image.open(src_img_path).convert("RGB")

            # Parse Bounding Box from XML
            # LOGIC CHANGE HERE
            bbox = None
            if crop_bbox and os.path.exists(xml_path):
                tree = ET.parse(xml_path)
                root = tree.getroot()
                bndbox = root.find("object/bndbox")
                if bndbox is not None:
                    xmin = int(bndbox.find("xmin").text)
                    ymin = int(bndbox.find("ymin").text)
                    xmax = int(bndbox.find("xmax").text)
                    ymax = int(bndbox.find("ymax").text)
                    bbox = (xmin, ymin, xmax, ymax)

            if bbox:
                cropped_img = img.crop(bbox)
            else:
                # If crop_bbox is False OR no bbox found, use full image
                cropped_img = img

            cropped_img.save(os.path.join(save_dir, base_name + ".jpg"))

        logger.info(f"Processed {len(file_list)} images for {dst_dir}")

    logger.info("Preprocessing Stanford Dogs (Cropping to BBox)...")
    process_split(train_list_path, train_crop_path)
    process_split(train_list_path, train_full_path, crop_bbox=False)
    process_split(test_list_path, test_crop_path)
    process_split(test_list_path, test_full_path, crop_bbox=False)
    logger.info("Stanford Dogs preprocessing completed.")


def download_flowers(path: str, use_segmentation: bool) -> None:
    """Downloads the Oxford Flowers 102 dataset.

    Args:
        path (str): Path where to download the dataset to.
        use_segmentation (bool): Whether to download the segmentation dataset too. Deprecated, as no segmentation dataset is available.
    """
    from torchvision.datasets import Flowers102

    logger.info("Downloading Oxford Flowers 102 dataset")
    flowers_dataset = Flowers102(root=path, download=True)
    logger.info("Oxford Flowers 102 dataset downloaded")

    if use_segmentation:
        logger.warning("Segmentation dataset is not available for Oxford Flowers 102. Skipping segmentation download.")


def download_cifar10(path: str, use_segmentation: bool) -> None:
    """Downloads the CIFAR-10 dataset.

    Args:
        path (str): Path where to download the dataset to.
        use_segmentation (bool): Whether to download the segmentation dataset too. Deprecated, as no segmentation dataset is available.
    """
    from torchvision.datasets import CIFAR10

    logger.info("Downloading CIFAR-10 dataset")
    train_dataset = CIFAR10(root=path, train=True, download=True)
    test_dataset = CIFAR10(root=path, train=False, download=True)
    logger.info("CIFAR-10 dataset downloaded")

    if use_segmentation:
        logger.warning("Segmentation dataset is not available for CIFAR-10. Skipping segmentation download.")


def download_cifar100(path: str, use_segmentation: bool) -> None:
    """Downloads the CIFAR-100 dataset.

    Args:
        path (str): Path where to download the dataset to.
        use_segmentation (bool): Whether to download the segmentation dataset too. Deprecated, as no segmentation dataset is available.
    """
    from torchvision.datasets import CIFAR100

    logger.info("Downloading CIFAR-100 dataset")
    train_dataset = CIFAR100(root=path, train=True, download=True)
    test_dataset = CIFAR100(root=path, train=False, download=True)
    logger.info("CIFAR-100 dataset downloaded")

    if use_segmentation:
        logger.warning("Segmentation dataset is not available for CIFAR-100. Skipping segmentation download.")


def download_pets(path: str, use_segmentation: bool) -> None:
    """Downloads the Oxford Pets dataset.

    Args:
        path (str): Path where to download the dataset to.
        use_segmentation (bool): Whether to download the segmentation dataset too. Deprecated, as no segmentation dataset is available.
    """
    from torchvision.datasets import OxfordIIITPet

    _ = OxfordIIITPet(
        root=path,
        download=True,
        transform=None,  # No transformation needed for downloading
        split="trainval",  # Download the training split
    )


def download_tiny_imagenet(path: str, use_segmentation: bool) -> None:
    """Downloads the Tiny ImageNet dataset.

    Args:
        path (str): Path where to download the dataset to.
        use_segmentation (bool): Whether to download the segmentation dataset too. Deprecated, as no segmentation dataset is available.
    """
    from tiny_imagenet_torch import TinyImageNet

    _ = TinyImageNet(
        root=path,
        download=True,
        transform=None,  # No transformation needed for downloading
        train=True,  # Download the training split
    )


FILE_LIST = [
    {
        "identifier": "CUB_200_2011",
        "description": "Caltech-UCSD Birds-200-2011 dataset",
        "dir": "CUB_200_2011",
        "download_fn": download_cub,
        "preprocess_fn": preprocess_cub,
    },
    {
        "identifier": "flowers102",
        "description": "Oxford Flowers 102 dataset",
        "dir": "flowers-102",  # Standard directory from 'torchvision'
        "download_fn": download_flowers,
        "preprocess_fn": None,  # No preprocessing needed for this dataset
    },
    {
        "identifier": "oxford_iiit_pet",
        "description": "Oxford IIIT Pets dataset",
        "dir": "oxford-iiit-pet",  # Standard directory from 'torchvision'
        "download_fn": download_pets,
        "preprocess_fn": None,  # No preprocessing needed for this dataset
    },
    {
        "identifier": "cifar10",
        "description": "CIFAR-10 dataset",
        "dir": "cifar-10-batches-py",  # Standard directory from 'torchvision'
        "download_fn": download_cifar10,
        "preprocess_fn": None,  # No preprocessing needed for this dataset
    },
    {
        "identifier": "cifar100",
        "description": "CIFAR-100 dataset",
        "dir": "cifar-100-python",  # Standard directory from 'torchvision'
        "download_fn": download_cifar100,
        "preprocess_fn": None,  # No preprocessing needed for this dataset
    },
    {
        "identifier": "tiny_imagenet",
        "description": "Tiny ImageNet dataset",
        "dir": "tiny-imagenet-200",  # Standard directory from 'tiny_imagenet_torch'
        "download_fn": download_tiny_imagenet,
        "preprocess_fn": None,  # No preprocessing needed for this dataset
    },
    {
        "identifier": "stanford_dogs",
        "description": "Stanford Dogs dataset",
        "dir": "stanford_dogs",
        "download_fn": download_dogs,
        "preprocess_fn": preprocess_dogs,
    },
]


def main() -> None:
    """Runs the tool."""
    parser = create_parser()
    args = parser.parse_args()

    # Files to download
    files_to_download = (
        FILE_LIST
        if "all" in args.target
        else [file_entry for file_entry in FILE_LIST if file_entry["identifier"] in args.target]
    )

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    for entry in files_to_download:
        entry["download_fn"](output_dir, args.use_segmentation)
        if entry.get("preprocess_fn") is not None:
            entry["preprocess_fn"](os.path.join(output_dir, entry["dir"], ""))


if __name__ == "__main__":
    main()
