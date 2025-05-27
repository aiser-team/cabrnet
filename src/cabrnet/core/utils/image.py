from PIL import Image
import torch
from typing import Callable


def square_resize(img: Image.Image) -> Image.Image:
    r"""Resizes image to a square aspect ratio.

    Args:
        img (image): Source image.

    Returns:
        Squared image.
    """
    final_size = min(img.width, img.height)
    return img.resize((final_size, final_size))


def open_image(
    img: str | Image.Image, preprocess: Callable[[Image.Image], Image.Image]
) -> tuple[Image.Image, torch.Tensor]:
    r"""Cleanly turns the specified image into a tensor.  This method ensures that the output tensor has shape
    `(1,3,H,W)` where `H` and `W` are the size of the image or the size specified by the preprocess.

    Args:
      img (str|Image.Image): Image as a PIL.Image or a filename.
      preprocess (Callable[[Image],Image]): Preprocess to apply to the image.

    Returns:
      The pair `(Image.Image,Tensor)`.
    """
    if isinstance(img, str):
        img = Image.open(img).convert("RGB")

    if preprocess is None:
        preprocess = torch.ToTensor()

    img_tensor = preprocess(img)
    if img_tensor.dim() != 4:
        # Fix number of dimensions if necessary
        img_tensor = torch.unsqueeze(img_tensor, dim=0)

    return img, img_tensor
