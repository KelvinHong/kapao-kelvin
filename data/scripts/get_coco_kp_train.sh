d='data/datasets/coco/images' # unzip directory
url=http://images.cocodataset.org/zips/
f1='train2017.zip' # 19G, 118k images
echo 'Downloading' $url$f1 '...'
curl -L $url$f1 -o $f1 && unzip -q $f1 -d $d && rm $f1
