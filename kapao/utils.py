import torch
import numpy as np
from typing import Tuple
import math

def xywhn2xyxy(
    x: torch.Tensor | np.ndarray,
    w: int = 640,
    h: int = 640,
    padw: float = 0,
    padh: float = 0,
) -> torch.Tensor | np.ndarray:
    """Convert a keypoint label from [cx, cy, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right.
    If there are more than 4 columns, the rest are assumed to be keypoints in [x, y, v] format.

    Args:
        x (torch.Tensor | np.ndarray): Tensor or ndarray of shape (n, 4) or (n, 4 + k * 3) where n is the number of boxes and k is the number of potential keypoints.
        w (int, optional): Width of image. Defaults to 640.
        h (int, optional): Height of image. Defaults to 640.
        padw (float, optional): Padding for width. Defaults to 0.
        padh (float, optional): Padding for height. Defaults to 0.

    Returns:
        torch.Tensor | np.ndarray: Resulting tensor or ndarray in the same shape.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh

    if y.shape[-1] > 4:
        nl = y.shape[0]
        kp = y[:, 4:].reshape(nl, -1, 3)
        kp[..., 0] *= w
        kp[..., 0] += padw
        kp[..., 1] *= h
        kp[..., 1] += padh
        y[:, 4:] = kp.reshape(nl, -1)

    return y


def clip_coords(boxes: torch.Tensor | np.ndarray, width: int, height: int):
    """Clip bboxes (x1, y1, x2, y2) to image boundaries (width, height) in place.
    Perform different operations depending on the type of input (torch.Tensor or np.ndarray) for optimization.

    Args:
        boxes (torch.Tensor | np.ndarray): A 2D tensor or ndarray of shape (n, 4) where n is the number of boxes.
        width (int): Image width.
        height (int): Image height.
    """
    if isinstance(boxes, torch.Tensor):
        boxes[:, 0].clamp_(0, width)
        boxes[:, 1].clamp_(0, height)
        boxes[:, 2].clamp_(0, width)
        boxes[:, 3].clamp_(0, height)
    else:
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)


def xyxy2xywhn(
    x: torch.Tensor | np.ndarray,
    w: int = 640,
    h: int = 640,
    clip: bool = False,
    eps: float = 0.0,
) -> torch.Tensor | np.ndarray:
    """Convert bbox xyxy to normalized cxcywh format.
    If keypoints are present, they are also converted to normalized scale.

    Args:
        x (torch.Tensor | np.ndarray): Input tensor or ndarray of shape (n, 4) or (n, 4 + k * 3) where
            n is the number of boxes and k is the number of potential keypoints.
        w (int, optional): Image width. Defaults to 640.
        h (int, optional): Image height. Defaults to 640.
        clip (bool, optional): Clip before normalize. Defaults to False.
        eps (float, optional): IDK why we need this (numerical stability?). Anyway defaults to 0.0.

    Returns:
        torch.Tensor | np.ndarray: Return a copy of the original tensor or ndarray.
    """
    if clip:
        clip_coords(x, width=w - eps, height=h - eps)
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h
    y[:, 2] = (x[:, 2] - x[:, 0]) / w
    y[:, 3] = (x[:, 3] - x[:, 1]) / h

    if y.shape[-1] > 4:
        nl = y.shape[0]
        kp = y[:, 4:].reshape(nl, -1, 3)
        kp[..., 0] /= w
        kp[..., 1] /= h
        y[:, 4:] = kp.reshape(nl, -1)

    return y

def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1
