import pytest
import numpy as np
from typing import Tuple
from PIL import Image
from kapao.dataset import (
    exif_size,
    exif_transpose,
    convert_flip,
    img2label_paths,
    extract_images_from_txtfile,
)


@pytest.fixture
def image_shape() -> Tuple[int, int]:
    return (100, 200)


@pytest.fixture
def pil_image(image_shape) -> Image.Image:
    return Image.new("RGB", image_shape)


def test_exif_size(image_shape, pil_image):
    assert exif_size(pil_image) == image_shape


def test_exif_transpose(pil_image):
    assert exif_transpose(pil_image) == pil_image


@pytest.mark.parametrize(
    "kp_flip,expected_flip",
    [
        (
            [0, 1, 2],
            {0: 0, 1: 1, 2: 2, 3: 3},
        ),
        (
            [1, 0],
            {0: 0, 1: 2, 2: 1},
        ),
    ],
)
def test_convert_flip(kp_flip, expected_flip):
    assert convert_flip(kp_flip) == expected_flip


@pytest.mark.parametrize(
    "img_paths,label_paths",
    [
        (
            ["a.jpg", "b.jpg"],
            ["a.txt", "b.txt"],
        ),
        (
            ["c.png"],
            ["c.txt"],
        ),
        (
            ["folder/d.png"],
            ["folder/d.txt"],
        ),
    ],
)
def test_img2label_paths(img_paths, label_paths):
    assert img2label_paths(img_paths) == label_paths


def test_extract_images_from_txtfile(tmp_path):
    txtfile = tmp_path / "file.txt"
    txtfile.write_text("b.png\na.jpg\nc.txt")
    assert extract_images_from_txtfile(txtfile) == ["a.jpg", "b.png"]
