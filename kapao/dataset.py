from PIL import Image
from typing import Tuple, List, Dict, Any, Annotated
import torch
from torch.utils.data.dataloader import DataLoader
import os
from torch.utils.data import Dataset
import numpy as np
import cv2
from pathlib import Path
from pydantic import BaseModel, AfterValidator, Field
import json

from .augmentations import letterbox


import albumentations as alb
from albumentations import LongestMaxSize, ToTensorV2

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

    Augment flag only affects the interpolation method, no special extra augmentations are applied.

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


class COCOPose(BaseModel):
    category_id: int = 0
    bbox: List[float] = Field(min_length=4, max_length=4)  # xywh
    keypoints: List[float]  # x, y, visibility


def validate_poses(poses: List[COCOPose]) -> List[COCOPose]:
    length_set = set()
    for pose in poses:
        if len(pose.keypoints) % 3 != 0:
            raise ValueError("Keypoints must be a multiple of 3 (x, y, visibility)")
        length_set.add(len(pose.keypoints))

        if len(length_set) > 1:
            raise ValueError("All poses must have the same number of keypoints")
    return poses


class COCOImage(BaseModel):
    id: int
    file_name: str
    width: int
    height: int
    poses: Annotated[List[COCOPose], AfterValidator(validate_poses)] = Field(
        default_factory=list
    )


class COCOKeypointDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        image_dir: str,
        num_keypoints: int,
        image_size: int = 640,
        batch_size: int = 16,
        training: bool = False,
        rect: bool = False,
        stride: int = 32,
        pad: int = 0.0,
    ):
        if image_size % stride != 0:
            raise ValueError(
                f"image_size {image_size} must be multiple of stride {stride}"
            )
        self.image_size = image_size
        self.stride = stride
        self.kp_bbox = 0.05

        # self.images will have bbox and poses of normalized values.
        # Still adhere to COCO format (XYWH), but we purely normalize the values.
        with open(json_path, "r") as f:
            data = json.load(f)

        self.training = training
        self.image_order: List[int] = [image["id"] for image in data["images"]]
        self.images: Dict[int, COCOImage] = {
            image["id"]: COCOImage(**image) for image in data["images"]
        }
        for image in self.images.values():
            image.file_name = os.path.abspath(os.path.join(image_dir, image.file_name))
            if not os.path.exists(image.file_name):
                raise FileNotFoundError(
                    f"Image file {image.file_name} does not exist, please check image directory '{image_dir}' is correct."
                )

        self.num_keypoints = num_keypoints
        for annotation in data["annotations"]:
            image_id = annotation["image_id"]
            if image_id not in self.images:
                raise ValueError(f"Image ID {image_id} not found in images.")
            width, height = self.images[image_id].width, self.images[image_id].height

            # Following pose_bbox will be in YOLO format, which means cx, cy, w, h.
            pose_bbox = annotation["bbox"]
            pose_bbox[0] /= width
            pose_bbox[1] /= height
            pose_bbox[2] /= width
            pose_bbox[3] /= height
            pose_bbox[0] += pose_bbox[2] / 2
            pose_bbox[1] += pose_bbox[3] / 2

            pose_kpts = torch.tensor(
                annotation["keypoints"], dtype=torch.float32
            ).reshape(-1, 3)
            pose_kpts[:, 0] /= width
            pose_kpts[:, 1] /= height
            if len(pose_kpts) != self.num_keypoints:
                raise ValueError(
                    f"Keypoints length {len(pose_kpts)} does not match expected {self.num_keypoints}."
                )

            pose = COCOPose(
                bbox=pose_bbox,
                keypoints=pose_kpts.flatten().tolist(),
            )

            self.images[image_id].poses.append(pose)

        self.shapes: np.ndarray = np.array(
            [
                [
                    self.images[self.image_order[i]].width,
                    self.images[self.image_order[i]].height,
                ]
                for i in range(len(self))
            ]
        ).astype(np.int64)

        num_samples = len(self)
        self.batch_indices = np.floor(np.arange(num_samples) / batch_size).astype(int)

        # Rectangular Training
        # Reorder the images so that letterbox will apply the least padding possible in the same batch.
        self.rect = rect
        if self.rect:
            self.batch_shapes, reorder_indices = reorder_rectangle_shapes(
                self.shapes, batch_size, image_size, stride, padding=pad
            )
            self.shapes = self.shapes[reorder_indices]
            self.image_order = [self.image_order[i] for i in reorder_indices]

        bbox_params = alb.BboxParams(format="yolo", clip=True)
        keypoint_params = alb.KeypointParams(format="xy")
        self.pre_transform = alb.Compose(
            transforms=[
                LongestMaxSize(
                    max_size=self.image_size, interpolation=cv2.INTER_LINEAR
                ),
            ],
            bbox_params=bbox_params,
            keypoint_params=keypoint_params,
        )
        self.end_transform = alb.Compose(
            [
                ToTensorV2(),
            ],
            bbox_params=bbox_params,
            keypoint_params=keypoint_params,
        )

    def __len__(self):
        return len(self.image_order)

    def _read_image(self, index):
        image_id = self.image_order[index]
        image = self.images[image_id]
        image_array = cv2.imread(image.file_name)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        return image_array

    def _read_label(self, index):
        # Return [N, 3*K+5] with YOLO format of bounding box (cxcywh).
        # (class_id, cx, cy, w, h, x1, y1, v1, ..., xK, yK, vK)
        image_id = self.image_order[index]
        image = self.images[image_id]
        labels = []
        for pose in image.poses:
            # single superobject bro!
            labels.append([0.0] + pose.bbox + pose.keypoints)
        if labels == []:
            return np.zeros((0, 3 * self.num_keypoints + 5), dtype=np.float32)
        labels = np.array(labels)
        labels = self._expand_to_kp_objects(labels, image.height, image.width)
        return labels

    def _expand_to_kp_objects(
        self, labels: np.ndarray, image_h: int, image_w: int
    ) -> np.ndarray:
        kp_w = self.kp_bbox * max(image_h, image_w) / image_w
        kp_h = self.kp_bbox * max(image_h, image_w) / image_h
        # TODO: we keep the order of pose instances here but it is actually unimportant
        expanded_labels = []
        for superobject_id in range(labels.shape[0]):
            superobject_instance = labels[superobject_id].copy()
            expanded_labels.append(superobject_instance.tolist())
            keypoints = superobject_instance[5:].reshape(-1, 3)
            for kp_ind, (kp_x, kp_y, visibility) in enumerate(keypoints):
                if visibility > 0:
                    expanded_labels.append(
                        [kp_ind + 1, kp_x, kp_y, kp_w, kp_h]
                        + [0.0] * (3 * self.num_keypoints)
                    )

        if expanded_labels == []:
            return np.zeros((0, 3 * self.num_keypoints + 5), dtype=np.float32)

        return np.array(expanded_labels, dtype=np.float32)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get item.

        Parameters
        ----------
        index : int
            Index

        Returns
        -------
        Dict[str, Any]
            "image": of shape [3, H, W]
            "bboxes": of shape [N, 4]
            "keypoints": of shape [N, K, 3]
            "class_ids": of shape [N], where 0 is superobject, 1..K are keypoints.
            "shapes": ((h0, w0), ((scale_h, scale_w), (pad_h, pad_w)))
            "filename: str  # like '000000040083.jpg'
        """
        original_image = self._read_image(index)
        h0, w0 = original_image.shape[:2]
        original_label = self._read_label(index)
        class_ids = original_label[:, 0]
        num_objects = original_label.shape[0]
        kpts_and_vis = original_label[:, 5:].reshape(-1, self.num_keypoints, 3)
        kpts_flatten = kpts_and_vis[:, :, :2].reshape(-1, 2)
        kpts_flatten_unnormalized = kpts_flatten.copy()
        kpts_flatten_unnormalized[:, 0] *= w0
        kpts_flatten_unnormalized[:, 1] *= h0
        vis = kpts_and_vis[:, :, 2]

        pre_transformed = self.pre_transform(
            image=original_image,
            bboxes=original_label[:, 1:5],
            keypoints=kpts_flatten_unnormalized,
        )
        loaded_image = pre_transformed["image"]
        pre_pad_image_w, pre_pad_image_h = loaded_image.shape[1], loaded_image.shape[0]
        loaded_bbox = pre_transformed["bboxes"]
        loaded_kpts_flatten_unnormalized = pre_transformed["keypoints"]
        loaded_kpts_flatten = loaded_kpts_flatten_unnormalized.copy()
        loaded_kpts_flatten[:, 0] /= pre_pad_image_w
        loaded_kpts_flatten[:, 1] /= pre_pad_image_h
        batch_shape = (
            self.batch_shapes[self.batch_indices[index]]
            if self.rect
            else self.image_size
        )
        loaded_image, (scale_w, scale_h), (pad_w, pad_h) = letterbox(
            loaded_image, batch_shape, auto=False, scaleup=self.training
        )
        post_pad_image_w, post_pad_image_h = (
            loaded_image.shape[1],
            loaded_image.shape[0],
        )
        loaded_bbox[:, ::2] *= pre_pad_image_w / post_pad_image_w
        loaded_bbox[:, 1::2] *= pre_pad_image_h / post_pad_image_h
        loaded_bbox[:, 0] += pad_w / post_pad_image_w
        loaded_bbox[:, 1] += pad_h / post_pad_image_h
        loaded_kpts_flatten[:, 0] *= pre_pad_image_w / post_pad_image_w
        loaded_kpts_flatten[:, 1] *= pre_pad_image_h / post_pad_image_h
        loaded_kpts_flatten[:, 0] += pad_w / post_pad_image_w
        loaded_kpts_flatten[:, 1] += pad_h / post_pad_image_h

        final_transformed = self.end_transform(
            image=loaded_image, bboxes=loaded_bbox, keypoints=loaded_kpts_flatten
        )

        final_kpts = final_transformed["keypoints"].reshape(-1, self.num_keypoints, 2)
        final_kpts = np.concatenate(
            [
                final_kpts,
                vis[:, :, np.newaxis],
            ],
            axis=2,
        )

        return {
            "image": final_transformed["image"],
            "bboxes": torch.from_numpy(final_transformed["bboxes"]).to(torch.float32),
            "keypoints": torch.from_numpy(final_kpts).to(torch.float32),
            "class_ids": torch.from_numpy(class_ids).to(
                torch.float32
            ),  # Include keypoints as objects
            "shapes": (
                (h0, w0),
                ((pre_pad_image_h / h0, pre_pad_image_w / w0), (pad_w, pad_h)),
            ),
            "filename": os.path.basename(
                self.images[self.image_order[index]].file_name
            ),
        }

    # Keep original collate format first. Once we done experiment ensure it is correct, we can refactor to dictionary output.
    @staticmethod
    def collate_fn(batch):
        images = torch.stack([x["image"] for x in batch])
        labels = []
        for batch_index in range(len(batch)):
            sample = batch[batch_index]
            label_wo_image_id = torch.cat(
                [
                    sample["class_ids"].unsqueeze(1),
                    sample["bboxes"],
                    sample["keypoints"].flatten(1),
                ],
                dim=1,
            )  # [N, 3*K+5]
            num_objects = label_wo_image_id.shape[0]
            label_image_id = torch.full(
                (num_objects, 1), batch_index, dtype=torch.float32
            )
            label = torch.cat([label_image_id, label_wo_image_id], dim=1)
            labels.append(label)
        labels = torch.cat(labels, dim=0)
        paths = [x["filename"] for x in batch]
        shapes = [x["shapes"] for x in batch]
        return images, labels, paths, shapes

    @property
    def labels(self):
        return [self._read_label(index) for index in range(len(self))]
