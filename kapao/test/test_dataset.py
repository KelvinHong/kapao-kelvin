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
    load_and_reshape_image,
    COCOImage,
    COCOKeypointDataset,
)
import torch


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
        (
            np.array([[640, 400], [640, 600]]),
            2,
            np.array([[1216, 1280]]),
            np.array([0, 1]),
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


@pytest.mark.parametrize(
    "augment",
    (True, False),
)
@pytest.mark.parametrize(
    "original_shape,input_size,expected_final_shape",
    [
        ((640, 640), 1280, (1280, 1280)),
        ((400, 640), 1280, (800, 1280)),
        ((800, 640), 1280, (1280, 1024)),
        ((640, 400), 1000, (1000, 625)),
    ],
)
def test_load_and_reshape_image(
    original_shape, input_size, expected_final_shape, tmp_path, augment
):
    path = tmp_path / "test.jpg"
    img = Image.new(
        "RGB", original_shape[::-1]
    )  # Image.new expects (width, height) but our original_shape is (height, width)
    img.save(path)

    img, result_og_shape = load_and_reshape_image(path, input_size, augment)

    assert result_og_shape == original_shape
    assert img.shape[:2] == expected_final_shape


class TestDataset:
    NUM_SAMPLES = 16
    KP_BBOX = 0.05

    @pytest.fixture
    def dataset_class(self):
        return COCOKeypointDataset

    @pytest.fixture
    def batch_size(self):
        return 2

    @pytest.fixture
    def dataset_kwargs(self, batch_size):
        return {
            "json_path": "data/datasets/coco/annotations/person_keypoints_val2017_mini.json",
            "image_dir": "data/datasets/coco/images/val2017",
            "num_keypoints": 17,
            "rect": True,
            "batch_size": batch_size,
        }

    @pytest.fixture
    def dataset(self, dataset_class, dataset_kwargs):
        return dataset_class(**dataset_kwargs)

    def test_init(self, dataset: COCOKeypointDataset):
        for image in dataset.images.values():
            assert isinstance(image, COCOImage)

        assert dataset.num_keypoints == 17
        assert dataset.kp_bbox == self.KP_BBOX

    def test_len(self, dataset: COCOKeypointDataset):
        assert len(dataset) == self.NUM_SAMPLES

    def test_batch_shapes(self, dataset: COCOKeypointDataset, batch_size):
        num_shapes = self.NUM_SAMPLES // batch_size

        assert len(dataset.batch_shapes) == num_shapes
        hw_ratios = [shape[0] / shape[1] for shape in dataset.batch_shapes]
        assert all(x<=y for x, y in zip(hw_ratios, hw_ratios[1:]))

    @pytest.mark.parametrize(
        "labels,expected_labels",
        [
            (
                np.zeros((0, 3 * 17 + 5), dtype=np.float32),
                np.zeros((0, 3 * 17 + 5), dtype=np.float32),
            ),
            (   # Pose without keypoints
                np.array([
                    [0, 0.3, 0.4, 0.2, 0.1,
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0.,],
                ]),
                np.array([
                    [0, 0.3, 0.4, 0.2, 0.1,
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0.,],
                ]),
            ),
            (   # Pose with 2 keypoints at keypoint id 2 & 13 in range 1-17.
                np.array([
                    [0, 0.3, 0.4, 0.2, 0.1,
                     0., 0., 0., .1, .2, 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0., .1, .5, 2., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0.,],
                ]),
                np.array([
                    [0, 0.3, 0.4, 0.2, 0.1,
                     0., 0., 0., .1, .2, 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0., .1, .5, 2., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0.,],
                    [2, 0.1, 0.2, 0.05, 0.075,
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0.,],
                    [13, 0.1, 0.5, 0.05, 0.075,
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0.,],
                ]),
            ),
            (
                # 2 poses with keypoints
                np.array([
                    [0, 0.3, 0.4, 0.2, 0.1,
                     0., 0., 0., .1, .2, 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0., .1, .5, 2., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0.,],
                    [0, 0.7, 0.8, 0.1, 0.15,
                     .4, .3, 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     .5, .5, .5, .1, .1, .3,],
                ]),
                np.array([
                    [0, 0.3, 0.4, 0.2, 0.1,
                     0., 0., 0., .1, .2, 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0., .1, .5, 2., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0.,],
                    [2, 0.1, 0.2, 0.05, 0.075,
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0.,],
                    [13, 0.1, 0.5, 0.05, 0.075,
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0.,],
                     [0, 0.7, 0.8, 0.1, 0.15,
                     .4, .3, 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     .5, .5, .5, .1, .1, .3,],
                     [1, 0.4, 0.3, 0.05, 0.075,
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0.,],
                     [16, 0.5, 0.5, 0.05, 0.075,
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0.,],
                     [17, 0.1, 0.1, 0.05, 0.075,
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0.,],
                ]),
            )
        ]
    )
    def test_expand_to_kp_objects(self, dataset: COCOKeypointDataset, labels, expected_labels):
        h, w = 200, 300
        assert np.allclose(dataset._expand_to_kp_objects(labels, h, w), expected_labels)

    @pytest.mark.parametrize(
        "index", list(range(16)),
    )
    def test_getitem(self, dataset: COCOKeypointDataset, index):
        sample = dataset[index]
        image: torch.Tensor = sample["image"]
        bboxes: torch.Tensor = sample["bboxes"]
        keypoints: torch.Tensor = sample["keypoints"]
        class_ids: torch.Tensor = sample["class_ids"]

        assert image.ndim == 3
        assert image.shape[0] == 3
        assert image.dtype == torch.uint8

        assert bboxes.ndim == 2
        assert bboxes.shape[1] == 4

        assert keypoints.ndim == 3
        assert keypoints.shape[1] == dataset.num_keypoints
        assert keypoints.shape[2] == 3

        assert bboxes.shape[0] == keypoints.shape[0]
        assert bboxes.shape[0] == class_ids.shape[0]
