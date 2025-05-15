# This tests whether refactor is faithful to original implementation.

from kapao.clean_dataset import COCOKeypointDataset
from kapao.dataset import LoadImagesAndLabels
import torch
import os
from collections import defaultdict

if __name__ == "__main__":
    training = True
    rect = True
    stride = 64
    batch_size = 1

    new_dataset = COCOKeypointDataset(
        json_path="data/datasets/coco/annotations/person_keypoints_val2017_mini.json",
        image_dir="data/datasets/coco/images/val2017",
        num_keypoints=17,
        image_size=1280,
        batch_size=batch_size,
        rect=rect,
        stride=stride,
        training=training,
    )
    old_dataset = LoadImagesAndLabels(
        path="data/datasets/coco/kp_labels/img_txt/val2017_mini.txt",
        labels_dir="kp_labels",
        num_keypoints=17,
        img_size=1280,
        batch_size=batch_size,
        augment=training,
        rect=rect,
        stride=stride,
    )
    assert len(new_dataset) == len(
        old_dataset
    ), f"Length mismatch: {len(new_dataset)} vs {len(old_dataset)}"

    pairs = defaultdict(dict)

    for ind in range(len(new_dataset))[:]:
        new_data = new_dataset[ind]
        new_image = new_data["image"]
        new_bboxes = new_data["bboxes"]
        new_path = new_data["filename"]
        pairs[new_path]["new"] = new_data
        old_data = old_dataset[ind]
        old_image = old_data[0]
        old_label = old_data[1]
        old_path = os.path.basename(old_data[2])
        pairs[old_path]["old"] = old_data

    for ind, (path, new_old_data) in enumerate(pairs.items()):
        new_data = new_old_data["new"]
        new_image = new_data["image"]
        new_bboxes = new_data["bboxes"]
        old_data = new_old_data["old"]
        old_image = old_data[0]
        old_label = old_data[1]

        assert (
            new_image.shape == old_image.shape
        ), f"Shape mismatch: {new_image.shape} vs {old_image.shape}"
        # Check superobjects bbox
        old_superobjects = old_label[:, 2:6][old_label[:, 1] == 0]
        new_superobjects = new_bboxes[:, :4][old_label[:, 1] == 0]
        assert torch.allclose(
            new_superobjects, old_superobjects, atol=1e-4
        ), f"{ind}, {new_superobjects}, {old_superobjects}"

