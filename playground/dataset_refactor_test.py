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
    assert len(new_dataset) == len(
        old_dataset
    ), f"Length mismatch: {len(new_dataset)} vs {len(old_dataset)}"
    for ind in range(len(new_dataset))[1:2]:
        new_image = new_dataset[ind]["image"]
        old_image = old_dataset[ind][0]
        assert (
            new_image.shape == old_image.shape
        ), f"Shape mismatch: {new_image.shape} vs {old_image.shape}"
        print("new_image.shape", new_image.shape)
        print("old_image.shape", old_image.shape)
