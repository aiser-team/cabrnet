from PIL import Image


def square_resize(img: Image.Image) -> Image.Image:
    r"""Resizes image to a square aspect ratio.

    Args:
        img (image): Source image.

    Returns:
        Squared image.
    """
    final_size = min(img.width, img.height)
    return img.resize((final_size, final_size))
