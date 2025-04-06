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
):
    # new-shape is hw

    # Resize and pad image while meeting stride-multiple constraints
    height, width = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale scale_ratios (new / old)
    scale_ratio = min(new_shape[0] / height, new_shape[1] / width)
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        scale_ratio = min(scale_ratio, 1.0)

    # Compute padding
    scale_ratios = scale_ratio, scale_ratio  # width, height ratios
    new_width, new_height = int(round(width * scale_ratio)), int(
        round(height * scale_ratio)
    )
    pad_width, pad_height = (
        new_shape[1] - new_width,
        new_shape[0] - new_height,
    )  # wh padding
    if auto:  # minimum rectangle
        pad_width, pad_height = np.mod(pad_width, stride), np.mod(
            pad_height, stride
        )  # wh padding

    pad_width /= 2  # divide padding into 2 sides
    pad_height /= 2

    if (width, height) != (new_width, new_height):  # resize
        im = cv2.resize(im, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(pad_height - 0.1)), int(round(pad_height + 0.1))
    left, right = int(round(pad_width - 0.1)), int(round(pad_width + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return im, scale_ratios, (pad_width, pad_height)
