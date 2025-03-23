# Refactor

This is a KAPAO refactor that works on `Python 3.12.8`

## Setup

Using `uv`, 

```console
uv venv --python 3.12.8
source .venv/bin/activate
uv pip install -r requirements.txt
```

Download all datasets following the original README.

Run
```console
python kapao/_minifier.py
```

## Try training

```console
python train.py \
--img 1280 \
--batch 8 \
--epochs 1 \
--data data/coco-kp-light.yaml \
--hyp data/hyps/hyp.kp-p6.yaml \
--val-scales 1 \
--val-flips -1 \
--weights yolov5s6.pt \
--project runs/s_e500 \
--name train \
--workers 1
```
