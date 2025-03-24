from PIL import Image
from typing import Tuple, List, Dict
from torch.utils.data.dataloader import DataLoader
import os

# The key for ExifTags Orientation:
# https://github.com/python-pillow/Pillow/blob/bca693bd82ce1dab40a375d101c5292e3a275143/src/PIL/ExifTags.py#L40
ORIENTATION_KEY = 0x0112


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
    exif = img._getexif()
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
