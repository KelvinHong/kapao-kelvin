#!/bin/bash
# Example usage: bash data/scripts/get_coco_kp_val.sh

# Make dataset directories
mkdir -p data/datasets/coco/images

# Download/unzip annotations
d='data/datasets/coco' # unzip directory
f1='annotations_trainval2017.zip'
url=http://images.cocodataset.org/annotations/
echo 'Downloading' $url$f1 '...'
curl -L $url$f1 -o $f1 && unzip -q $f1 -d $d && rm $f1

# Download/unzip images
d='data/datasets/coco/images' # unzip directory
url=http://images.cocodataset.org/zips/
f2='val2017.zip'   # 1G, 5k images
echo 'Downloading' $url$f2 '...'
curl -L $url$f2 -o $f2 && unzip -q $f2 -d $d && rm $f2
