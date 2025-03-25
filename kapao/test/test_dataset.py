import pytest
import numpy as np
from typing import Tuple
from PIL import Image
import random
from kapao.dataset import (
    exif_size,
    exif_transpose,
    convert_flip,
    img2label_paths,
    extract_images_from_txtfile,
    read_sample,
)


@pytest.fixture
def image_shape() -> Tuple[int, int]:
    return (random.randint(200, 300), random.randint(100, 250))


@pytest.fixture
def pil_image(image_shape) -> Image.Image:
    return Image.new("RGB", image_shape)


@pytest.fixture(params=[4, 10, 34])
def image_with_label(tmp_path, pil_image: Image.Image, request):
    num_coords = request.param
    tmp_img_path = tmp_path / "img.jpg"
    pil_image.save(tmp_img_path)

    tmp_label_path = tmp_path / "label.txt"
    # Insert a random superobject label and a random keypoint label.
    random_superobject = [0] + [random.random() for _ in range(4)]
    for _ in range(num_coords // 2):
        random_keypoint = [random.random(), random.random(), random.randint(0, 2)]
        random_superobject += random_keypoint
    random_superobject_str = " ".join(map(str, random_superobject))
    random_keypoint = (
        [1] + [random.random() for _ in range(4)] + [0] * (3 * num_coords // 2)
    )
    random_keypoint_str = " ".join(map(str, random_keypoint))
    tmp_label_path.write_text(f"{random_superobject_str}\n{random_keypoint_str}\n")

    return tmp_img_path, tmp_label_path, num_coords


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


def test_read_sample(image_with_label):
    tmp_img_path, tmp_label_path, num_coords = image_with_label

    img_file, labels, shape, segments, nm, nf, ne, nc = read_sample(
        tmp_img_path, tmp_label_path, num_coords
    )
    assert labels.shape == (2, 3 * num_coords // 2 + 5)
    assert nf == 1
    assert nm == 0
    assert ne == 0
    assert nc == 0
