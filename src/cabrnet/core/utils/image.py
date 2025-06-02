from contextlib import contextmanager
from PIL import Image
import torch
from torchvision.transforms import ToTensor
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


@contextmanager
class contextual_open_image:
    r"""Cleanly turns the specified image into a tensor within a `with` statement.
    This method ensures that the output tensor has shape `(1,C,H,W)`
    where `H` and `W` are the size of the image or the size specified by the preprocess.
    `C`, the number of channels, is set to `3` if `rgb` is True;
    otherwise, `C` is the number of channels of the original image.
    This is the preferred way to open images if the image is opened temporarily
    (if the input image was a file name, the image is destroyed when the context is left).

    The best way to use the context manager is as follows::

      with safe_open_image(input=filename_or_image) as (img,img_tensor):
          # ...
      # img and img_tensor should no longer be accessed here.


    Args:
      input (str|Image.Image): Image as a PIL.Image or a filename.
      preprocess (Callable[[Image],Image]): Preprocess to apply to the image.
      rgb (bool, Optional): If True, makes sure that the image is in RGB mode (ignores the issue otherwise).

    Returns:
      The pair `(Image.Image,Tensor)`.
    """

    def __init__(
        self, input: str | Image.Image, preprocess: Callable[[Image.Image], Image.Image] | None, rgb: bool = True
    ):
        self._input = input
        self._preprocess = preprocess
        self._rgb = rgb
        self._image_to_delete = None  # If __enter__ opened a file, will be deleted by __exit__
        self._tensor = None

    def __enter__(self):
        img, img_tensor, new_image = _open_image(self._input, self._preprocess, self._rgb)
        self._image_to_delete = new_image
        self._tensor = img_tensor
        return img, img_tensor

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._image_to_delete is not None:
            self._image_to_delete.close()
        del self._tensor


def _open_image(
    img: str | Image.Image, preprocess: Callable[[Image.Image], Image.Image], rgb: bool
) -> tuple[Image.Image, torch.Tensor, Image.Image | None]:
    # Does the opening (cf. open_image) and returns the image and the tensor.
    # Additionally, returns the "new image" if the process created an image.
    # This tells the context manager that it should eventually destroy it.
    new_image = None
    if isinstance(img, str):
        img = Image.open(img)
        new_image = img

    if rgb:
        img = img.convert("RGB")

    if preprocess is None:
        preprocess = ToTensor()

    img_tensor = preprocess(img)
    if img_tensor.dim() != 4:
        # Fix number of dimensions if necessary
        img_tensor = torch.unsqueeze(img_tensor, dim=0)

    return img, img_tensor, new_image


def open_image(
    img: str | Image.Image, preprocess: Callable[[Image.Image], Image.Image] | None, rgb: bool = True
) -> tuple[Image.Image, torch.Tensor]:
    r"""Cleanly turns the specified image into a tensor.  This method ensures that the output tensor has shape
    `(1,3,H,W)` where `H` and `W` are the size of the image or the size specified by the preprocess
    (if `rgb` is True, otherwise it does not modify the number of channels).

    Args:
      img (str|Image.Image): Image as a PIL.Image or a filename.
      preprocess (Callable[[Image],Image], Optional): Preprocess to apply to the image.  Default: ToTensor().
      rgb (bool, Optional): If True, makes sure that the image is in RGB mode (ignores the issue otherwise).

    Returns:
      The pair `(Image.Image,Tensor)`.
    """
    img, img_tensor, _ = _open_image(img, preprocess, rgb)
    return img, img_tensor
