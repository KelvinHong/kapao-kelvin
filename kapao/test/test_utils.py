from kapao.utils import xywhn2xyxy, xyxy2xywhn
import torch


def test_xywhn2xyxy():
    original_tensor = torch.Tensor(
        [[0.5, 0.5, 0.2, 0.2, 0.4, 0.5, 1.0], [0.3, 0.3, 0.1, 0.1, 0.3, 0.9, 2.0]]
    )
    w, h = 1000, 1000
    padw, padh = 50, 50
    expected_tensor = torch.Tensor(
        [
            [450.0, 450.0, 650.0, 650.0, 450.0, 550.0, 1.0],
            [300.0, 300.0, 400.0, 400.0, 350.0, 950.0, 2.0],
        ]
    )

    result_tensor = xywhn2xyxy(original_tensor, w, h, padw, padh)
    assert torch.allclose(
        result_tensor, expected_tensor
    ), f"Expected {expected_tensor}, but got {result_tensor}"


def test_xyxy2xywhn():
    original_tensor = torch.Tensor(
        [
            [450.0, 450.0, 650.0, 650.0, 450.0, 550.0, 1.0],
            [300.0, 300.0, 400.0, 400.0, 350.0, 950.0, 2.0],
        ]
    )
    w, h = 1000, 2000
    expected_tensor = torch.Tensor(
        [
            [0.55, 0.275, 0.2, 0.1, 0.45, 0.275, 1.0],
            [0.35, 0.175, 0.1, 0.05, 0.35, 0.475, 2.0],
        ]
    )

    result_tensor = xyxy2xywhn(original_tensor, w=w, h=h)
    assert torch.allclose(
        result_tensor,
        expected_tensor,
        atol=1e-6,
        rtol=1e-5,
    ), f"Expected {expected_tensor}, but got {result_tensor}"
