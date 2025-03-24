
from PIL import Image
from typing import Tuple


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
