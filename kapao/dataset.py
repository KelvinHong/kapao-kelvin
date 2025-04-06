from PIL import Image
from typing import Tuple, List, Dict, Any
from torch.utils.data.dataloader import DataLoader
import os
import numpy as np
import cv2
from pathlib import Path

# The key for ExifTags Orientation:
# https://github.com/python-pillow/Pillow/blob/bca693bd82ce1dab40a375d101c5292e3a275143/src/PIL/ExifTags.py#L40
ORIENTATION_KEY = 0x0112
IMG_FORMATS = [
    "bmp",
    "jpg",
    "jpeg",
    "png",
    "tif",
    "tiff",
    "dng",
    "webp",
    "mpo",
]


def exif_size(img: Image.Image) -> Tuple[int, int]:
    """Return Exif corrected image size.

    Args:
        img (Image.Image): Pillow image.

    Returns:
        Tuple[int, int]: Correct image size in (width, height).

    References:
        https://sirv.com/help/articles/rotate-photos-to-be-upright
    """
    shape = img.size
    exif = img.getexif()
    if exif is not None and isinstance(exif, dict):
        rotation = dict(exif.items())[ORIENTATION_KEY]
        if rotation == 6:  # rotation 270
            shape = (shape[1], shape[0])
        elif rotation == 8:  # rotation 90
            shape = (shape[1], shape[0])

    return shape


# TODO: Why not just use this directly from Pillow?
def exif_transpose(image: Image.Image) -> Image.Image:
    """Transpose a PIL image accordingly if it has an EXIF Orientation tag.
    From https://github.com/python-pillow/Pillow/blob/bca693bd82ce1dab40a375d101c5292e3a275143/src/PIL/ImageOps.py#L686

    Args:
        image (Image.Image): Input image.

    Returns:
        Image.Image: Corrected image.
    """
    exif = image.getexif()
    default_orientation_key = 1
    orientation = exif.get(ORIENTATION_KEY, default_orientation_key)
    if orientation != default_orientation_key:
        method = {
            2: Image.FLIP_LEFT_RIGHT,
            3: Image.ROTATE_180,
            4: Image.FLIP_TOP_BOTTOM,
            5: Image.TRANSPOSE,
            6: Image.ROTATE_270,
            7: Image.TRANSVERSE,
            8: Image.ROTATE_90,
        }.get(orientation)
        if method is not None:
            image = image.transpose(method)
            del exif[ORIENTATION_KEY]
            image.info["exif"] = exif.tobytes()
    return image


def convert_flip(kp_flip: List[int]) -> Dict[int, int]:
    """Convert keypoint flip to also include superobject information.
    Consider id 0 as superobject.
    Example: [1, 0, 3, 2] -> {0:0, 1:2, 2:1, 3:4, 4:3}

    Args:
        kp_flip (List[int]): Keypoint flip info.

    Returns:
        Dict[int, int]: Keypoint flip with superobject info.
    """
    object_flip = {0: 0}  # Means superobject doesn't flip.
    for i in range(len(kp_flip)):
        object_flip[i + 1] = kp_flip[i] + 1

    return object_flip


class InfiniteDataLoader(DataLoader):
    """Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def img2label_paths(img_paths, image_dir="images", labels_dir="labels"):
    return [
        os.path.splitext(s.replace(image_dir, labels_dir))[0] + ".txt"
        for s in img_paths
    ]


def extract_images_from_txtfile(path: str | Path) -> List[str]:
    """Extract image paths from a txt file.

    Args:
        path (str | Path): Path to the txt file.

    Returns:
        List[str]: List of image paths.
    """
    img_files = []
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"ERROR: {path} is not a file.")

    with open(path, "r") as img_txts:
        img_files += img_txts.read().strip().splitlines()
    img_files = sorted(
        [str(Path(x)) for x in img_files if x.lower().endswith(tuple(IMG_FORMATS))]
    )
    return img_files


def read_sample(
    img_file: str, label_file: str, num_coords: int
) -> Tuple[str, np.ndarray, Tuple[int, int], List, int, int, int, int]:
    # Verify one image-label pair
    # Returning 2nd (np.ndarray) is of shape [N, 3K+5].
    nm, nf, ne, nc, segments = (
        0,
        0,
        0,
        0,
        [],
    )  # number (missing, found, empty, corrupt), message, segments
    # verify images
    im = Image.open(img_file)
    im.verify()  # PIL verify
    shape = exif_size(im)  # image size
    assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}"
    if im.format.lower() in ("jpg", "jpeg"):
        with open(img_file, "rb") as f:
            f.seek(-2, 2)
            if f.read() != b"\xff\xd9":  # corrupt JPEG
                Image.open(img_file).save(
                    img_file, format="JPEG", subsampling=0, quality=100
                )  # re-save image

    # verify labels
    if os.path.isfile(label_file):
        nf = 1  # label found
        with open(label_file, "r") as f:
            l = [x.split() for x in f.read().strip().splitlines() if len(x)]
            l = np.array(l, dtype=np.float32)
        if len(l):
            assert (l >= 0).all(), "negative labels"
        else:
            ne = 1  # label empty
            l = np.zeros((0, 5 + num_coords * 3 // 2), dtype=np.float32)
    else:
        nm = 1  # label missing
        l = np.zeros((0, 5 + num_coords * 3 // 2), dtype=np.float32)
    return img_file, l, shape, segments, nm, nf, ne, nc


def read_samples(
    path: str | Path, num_coords: int, labels_dir: str = "labels"
) -> Dict[str, Any]:
    """Read all samples from a txt file with caching.

    Args:
        path (str | Path): A text file containing all image paths to be used.
        num_coords (int): Number of coordinates which is 2 * number of keypoints.
        labels_dir (str, optional): Name of the labels directory, to be used for swapping. Defaults to "labels".

    Returns:
        Dict[str, Any]: A dictionary, containing all the sample data and metadata.
        Example:
        {
            "results": Tuple[int * 5] = (found, missing, empty label, corrupted, len(img_files)),
            "version": float = 0.4,  # Might be useless, TODO: remove this if okay.
            "data/datasets/coco/images/val2017/000000006012.jpg": Tuple[np.ndarray (N, 3K+5), (w, h), segment (not important)],
            "data/datasets/coco/images/val2017/000000010977.jpg": {same},
            ...
        }
        One (3K+5) tensor seems to be
        [class_id, cx, cy, w, h, x1, y1, v1, ..., xK, yK, vK]
        class_id = 0 means superobject, 1->K means keypoint.
        All coordinate values are normalized within 0-1,
        visibility flags can be 0, 1, 2 only.
    """
    path = Path(path)
    img_files = extract_images_from_txtfile(path)
    label_files = img2label_paths(img_files, labels_dir=labels_dir)
    cache_path: Path = path.with_suffix(".cache")
    if cache_path.is_file():
        sample_data = np.load(cache_path, allow_pickle=True).item()
    else:
        sample_data = {}
        nm, nf, ne, nc = 0, 0, 0, 0  # number missing, found, empty, corrupt
        for img_file, label_file in zip(img_files, label_files):
            (
                img_file,
                label_array,
                shape,
                segments,
                nm_f,
                nf_f,
                ne_f,
                nc_f,
            ) = read_sample(img_file, label_file, num_coords)
            nm += nm_f
            nf += nf_f
            ne += ne_f
            nc += nc_f
            if img_file:
                sample_data[img_file] = [label_array, shape, segments]
        if nf == 0:
            raise ValueError(f"No labels found in {cache_path}.")
        sample_data["results"] = nf, nm, ne, nc, len(img_files)
        sample_data["version"] = 0.4  # cache version

        np.save(cache_path, sample_data)  # save cache for next time
        cache_path.with_suffix(".cache.npy").rename(cache_path)  # remove .npy suffix

    return sample_data


def reorder_rectangle_shapes(
    original_shapes: np.ndarray,
    batch_size: int,
    img_size: int,
    stride: int,
    padding: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reorder original shapes by height/width increasing ratio, outputs a number of shapes to be used in letterbox augmentation.
    In the same batch, all images are resized to the same size, they will be as square as possible.

    Args:
        original_shapes (np.ndarray): The original shapes of the images, do not need to be sorted. Shape (N, 2) in wh format.
        batch_size (int): Batch size of dataset.
        img_size (int): This is model input shape, usually it is square, so we only require int.
        stride (int): stride of the model.
        padding (int, optional): Padding to be added. Defaults to 0.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - 1st np.ndarray: The reordered shapes of the images to be used in letterbox augmentation, Shape (N, 2) in hw format.
            - 2nd np.ndarray: The index obtained from np.argsort.
    """
    num_samples = len(original_shapes)
    batch_indices = np.floor(np.arange(num_samples) / batch_size).astype(int)
    num_batches = batch_indices[-1] + 1  # number of batches

    aspect_ratios = original_shapes[:, 1] / original_shapes[:, 0]  # aspect ratio
    increasing_indices = aspect_ratios.argsort()
    aspect_ratios = aspect_ratios[increasing_indices]

    shapes = [[1, 1]] * num_batches
    for i in range(num_batches):
        batch_aspect_ratios = aspect_ratios[batch_indices == i]
        min_ratio, max_ratio = batch_aspect_ratios.min(), batch_aspect_ratios.max()
        if max_ratio < 1:
            shapes[i] = [max_ratio, 1]
        elif min_ratio > 1:
            shapes[i] = [1, 1 / min_ratio]

    reordered_shapes = (
        np.ceil(np.array(shapes) * img_size / stride + padding).astype(int) * stride
    )  # hw

    return reordered_shapes, increasing_indices


def load_and_reshape_image(
    image_path: str, input_size: int, augment: bool = False
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Load an image and reshape it to the desired input size for model forward.
    Assuming we perform augmentation only when training, then
    we only use INTER_AREA interpolation when shrinking & validating.
    Everything else uses INTER_LINEAR interpolation.

    Args:
        image_path (str): Path to the image file.
        input_size (int): Desired input size for the model.
        augment (bool, optional): Augmentation is enabled or not. Defaults to False.

    Returns:
        Tuple[np.ndarray, Tuple[int, int]]: Reshaped image and its original size.
    """
    img = cv2.imread(image_path)  # BGR
    original_h, original_w = img.shape[:2]

    scale = input_size / max(original_h, original_w)
    if scale != 1:
        interpolation_method = (
            cv2.INTER_AREA if scale < 1 and not augment else cv2.INTER_LINEAR
        )
        img = cv2.resize(
            img,
            (int(original_w * scale), int(original_h * scale)),
            interpolation_method,
        )
    return img, (original_h, original_w)
