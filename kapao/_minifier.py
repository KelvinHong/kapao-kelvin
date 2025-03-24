# Create a minimal training set annotations/person_keypoints_train2017_mini.json
# To speed up refactor and testing.

import os
import json
import random
import logging
from typing import Literal


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_DIR = os.path.join("data", "datasets", "coco")
ANNOTATION_DIR = os.path.join(DATASET_DIR, "annotations")
KP_IMG_DIR = os.path.join(DATASET_DIR, "kp_labels", "img_txt")


def create_mini_annotations(
    split: Literal["train", "val"], sample_size: int = 16, seed: int = 0
):
    if split not in ["train", "val"]:
        raise ValueError("split should be either 'train' or 'val'")

    annotation_path = os.path.join(ANNOTATION_DIR, f"person_keypoints_{split}2017.json")
    new_annotation_path = os.path.join(
        ANNOTATION_DIR, f"person_keypoints_{split}2017_mini.json"
    )
    new_img_txt_path = os.path.join(KP_IMG_DIR, f"{split}2017_mini.txt")
    with open(annotation_path, "r") as f:
        original_annotations = json.load(f)

    images = original_annotations["images"]
    annotations = original_annotations["annotations"]

    random.seed(seed)
    selected_images = random.sample(images, sample_size)
    selected_ids = [img["id"] for img in selected_images]
    selected_image_paths = [
        os.path.join(DATASET_DIR, "images", f"{split}2017", img["file_name"])
        for img in selected_images
    ]

    selected_annotations = [
        ann for ann in annotations if ann["image_id"] in selected_ids
    ]
    logger.info(
        f"Selected {len(selected_images)} images and {len(selected_annotations)} annotations."
    )

    # Replace original_annotations with minified annotations.
    original_annotations["images"] = selected_images
    original_annotations["annotations"] = selected_annotations

    with open(new_annotation_path, "w") as f:
        json.dump(original_annotations, f)

    with open(new_img_txt_path, "w") as f:
        f.write("\n".join(selected_image_paths))


if __name__ == "__main__":
    create_mini_annotations("train", seed=0)
    create_mini_annotations("val", seed=2)
