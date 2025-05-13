# This tests whether refactor is faithful to original implementation.

from kapao.clean_dataset import COCOKeypointDataset
from kapao.dataset import LoadImagesAndLabels


if __name__ == "__main__":
    training = False
    rect = True
    stride = 64

    new_dataset = COCOKeypointDataset(
        json_path="data/datasets/coco/annotations/person_keypoints_val2017_mini.json",
        image_dir="data/datasets/coco/images/val2017",
        num_keypoints=17,
        image_size=1280,
        batch_size=4,
        rect=rect,
        stride=stride,
        training=training,
    )
    old_dataset = LoadImagesAndLabels(
        path="data/datasets/coco/kp_labels/img_txt/val2017_mini.txt",
        labels_dir="kp_labels",
        num_keypoints=17,
        img_size=1280,
        batch_size=4,
        augment=training,
        rect=rect,
        stride=stride,
    )
    label1 = new_dataset._read_label(1)
    breakpoint()
