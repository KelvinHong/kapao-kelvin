import torch
import numpy as np


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
