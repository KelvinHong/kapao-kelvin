from kapao.augmentations import letterbox
from PIL import Image
import numpy as np
import pytest


@pytest.mark.parametrize(
    "original_shape,new_shape,auto,expected_shape",
    [
        ((640, 640), (640, 640), True, (640, 640)),
        ((640, 640), (1280, 1280), True, (1280, 1280)),
        ((640, 640), (1280, 1280), False, (1280, 1280)),
        ((800, 600), (800, 800), True, (800, 608)),
        ((800, 600), (800, 800), False, (800, 800)),
        ((640, 401), (1280, 1280), True, (1280, 832)),
    ],
)
def test_letterbox(original_shape, new_shape, auto, expected_shape, tmp_path):
    path = tmp_path / "test.jpg"
    img = Image.new(
        "RGB", original_shape[::-1]
    )  # Image.new expects (width, height) but our original_shape is (height, width)
    img.save(path)
    numpy_image = np.array(img)

    result_image = letterbox(
        numpy_image,
        new_shape=new_shape,
        auto=auto,
    )[0]

    assert (
        result_image.shape[:2] == expected_shape
    ), f"Expected {expected_shape}, but got {result_image.shape[:2]}"
