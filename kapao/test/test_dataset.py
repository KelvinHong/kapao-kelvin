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
    read_samples,
    reorder_rectangle_shapes,
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
    tmp_img_path = tmp_path / "001.jpg"
    pil_image.save(tmp_img_path)

    tmp_label_path = tmp_path / "001.txt"
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


def test_read_samples(tmp_path, image_with_label):
    tmp_img_path, tmp_label_path, num_coords = image_with_label
    txtfile = tmp_path / "file.txt"
    txtfile.write_text(f"{tmp_img_path}\n")
    cache_path = txtfile.with_suffix(".cache")

    assert not cache_path.is_file()

    samples = read_samples(txtfile, num_coords, labels_dir="")
    assert len(samples) == 3  # results, version, one sample.
    assert cache_path.is_file()

    samples = read_samples(txtfile, num_coords, labels_dir="")
    assert len(samples) == 3  # results, version, one sample.


@pytest.mark.parametrize(
    "original_shapes,batch_size,expected_shapes,expected_reordering",
    [
        (
            np.array([[400, 640], [640, 640]]),
            1,
            np.array([[1280, 1280], [1280, 832]]),
            np.array([1, 0]),
        ),
        (
            np.array([[400, 640], [640, 640]]),
            2,
            np.array([[1280, 1280]]),
            np.array([1, 0]),
        ),
        (
            np.array([[400, 640], [640, 640], [800, 640], [450, 640]]),
            2,
            np.array([[1280, 1280], [1280, 960]]),
            np.array([2, 1, 3, 0]),
        ),
    ],
)
def test_reorder_rectangle_shapes(
    original_shapes, batch_size, expected_shapes, expected_reordering
):
    img_size = 1280
    stride = 64
    padding = 0

    result_1, result_2 = reorder_rectangle_shapes(
        original_shapes, batch_size, img_size, stride, padding
    )
    assert np.array_equal(result_1, expected_shapes)
    assert np.array_equal(result_2, expected_reordering)
