
from PIL import Image, ExifTags

# The key for ExifTags Orientation: 
# https://github.com/python-pillow/Pillow/blob/bca693bd82ce1dab40a375d101c5292e3a275143/src/PIL/ExifTags.py#L40
orientation = 0x0112

def exif_size(img: Image.Image):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    exif = img._getexif()
    if exif is not None and isinstance(exif, dict):
        rotation = dict(exif.items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])

    return s
