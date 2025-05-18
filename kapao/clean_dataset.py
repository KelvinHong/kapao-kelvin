from torch.utils.data import Dataset
import json
import os
from pydantic import BaseModel, AfterValidator, Field
from typing import List, Dict, Annotated, Any
import torch
import numpy as np
from kapao.dataset import reorder_rectangle_shapes
from kapao.augmentations import letterbox
from kapao.utils import xywhn2xyxy
import cv2
import albumentations as alb
from albumentations import LongestMaxSize, Normalize, ToTensorV2


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
