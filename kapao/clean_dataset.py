from torch.utils.data import Dataset
import json
import os
from pydantic import BaseModel, AfterValidator, Field
from typing import List, Dict, Annotated
import torch
import numpy as np
from kapao.dataset import reorder_rectangle_shapes
import cv2
import albumentations as alb
from albumentations import LongestMaxSize
import cv2


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

            pose_bbox = annotation["bbox"]
            pose_bbox[0] /= width
            pose_bbox[1] /= height
            pose_bbox[2] /= width
            pose_bbox[3] /= height

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

        self.transform = alb.Compose(
            transforms = [
                LongestMaxSize(max_size=self.image_size, interpolation=cv2.INTER_LINEAR),
            ]
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
        # Return [N, 3*K+5]
        # (class_id, cx, cy, w, h, x1, y1, v1, ..., xK, yK, vK)
        image_id = self.image_order[index]
        image = self.images[image_id]
        labels = []
        for pose in image.poses:
            # single superobject bro!
            labels.append(
                [0.0] + pose.bbox + pose.keypoints
            )
        return np.array(labels)
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Get item.

        Parameters
        ----------
        index : int
            Index

        Returns
        -------
        Dict[str, torch.Tensor]
            "image": of shape [C, H, W]
            "bboxes": of shape [N, 4]
            "keypoints": of shape [N, K, 3]
        """
        

        original_image = self._read_image(index)
        loaded_image = self.transform(image=original_image)["image"]
        original_label = self._read_label(index)

