from PIL import Image
from typing import Tuple, List, Dict
from torch.utils.data.dataloader import DataLoader
import os
import numpy as np
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


def verify_image_label(args):
    # Verify one image-label pair
    im_file, lb_file, prefix, num_coords = args
    nm, nf, ne, nc, msg, segments = (
        0,
        0,
        0,
        0,
        "",
        [],
    )  # number (missing, found, empty, corrupt), message, segments
    # verify images
    im = Image.open(im_file)
    im.verify()  # PIL verify
    shape = exif_size(im)  # image size
    assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}"
    if im.format.lower() in ("jpg", "jpeg"):
        with open(im_file, "rb") as f:
            f.seek(-2, 2)
            if f.read() != b"\xff\xd9":  # corrupt JPEG
                Image.open(im_file).save(
                    im_file, format="JPEG", subsampling=0, quality=100
                )  # re-save image
                msg = f"{prefix}WARNING: corrupt JPEG restored and saved {im_file}"

    # verify labels
    if os.path.isfile(lb_file):
        nf = 1  # label found
        with open(lb_file, "r") as f:
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
    return im_file, l, shape, segments, nm, nf, ne, nc, msg
