import numpy as np
import cv2
from typing import Union, List, Tuple


def letterbox(
    im: np.ndarray,
    new_shape: Union[int, Tuple[int, int]] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = True,
    scaleup: bool = True,
    stride: int = 32,
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    """Apply letterbox augmentation to an image.
    New image will have the relation of
    New_w = round(Old_w * scale_ratio) + 2 * pad_width
    New_h = round(Old_h * scale_ratio) + 2 * pad_height
    The scale_ratio is the same across width and height, so the image is not distorted.

    Args:
        im (np.ndarray): Original numpy image.
        new_shape (Union[int, Tuple[int, int]], optional): Target new shape in hw format,
            but it will not be smaller (in multiples of stride) if auto=True, as to create a minimum
            sized rectangle image that contains the original image. Defaults to (640, 640).
        color (Tuple[int, int, int], optional): The color to fill the padding with. Defaults to (114, 114, 114).
        auto (bool, optional): If True, minimally pad images so height and width are multiples of stride. Defaults to True.
        scaleup (bool, optional): Whether this augmentation should scale up the image. Defaults to True.
        stride (int, optional): Stride. Defaults to 32.

    Returns:
        Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
            - Resized image.
            - Scale ratios (width, height).
            - Half paddings (width, height).
    """
    height, width = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    scale_ratio = min(new_shape[0] / height, new_shape[1] / width)

    if not scaleup:
        # Not scaling up, lead to better val mAP.
        scale_ratio = min(scale_ratio, 1.0)

    scale_ratios = scale_ratio, scale_ratio
    new_width, new_height = int(round(width * scale_ratio)), int(
        round(height * scale_ratio)
    )
    pad_width, pad_height = (
        new_shape[1] - new_width,
        new_shape[0] - new_height,
    )
    if auto:  # minimum rectangle
        pad_width, pad_height = np.mod(pad_width, stride), np.mod(pad_height, stride)

    pad_width /= 2
    pad_height /= 2

    if (width, height) != (new_width, new_height):
        im = cv2.resize(im, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(pad_height - 0.1)), int(round(pad_height + 0.1))
    left, right = int(round(pad_width - 0.1)), int(round(pad_width + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )

    return im, scale_ratios, (pad_width, pad_height)
