# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Dataloaders and dataset utils
"""

import glob
import hashlib
import json
import logging
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import ThreadPool, Pool
from pathlib import Path
from threading import Thread
import pdb

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.augmentations import (
    Albumentations,
    augment_hsv,
    copy_paste,
    letterbox,
    mixup,
    random_perspective,
)
from utils.general import (
    check_requirements,
    check_file,
    check_dataset,
    xywh2xyxy,
    xywhn2xyxy,
    xyxy2xywhn,
    xyn2xy,
    segments2boxes,
    clean_str,
)
from utils.torch_utils import torch_distributed_zero_first
from kapao.dataset import (
    exif_size,
    InfiniteDataLoader,
    img2label_paths,
    convert_flip,
    extract_images_from_txtfile,
    IMG_FORMATS,
    read_samples,
    reorder_rectangle_shapes,
)

# Parameters
HELP_URL = "https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data"

VID_FORMATS = [
    "mov",
    "avi",
    "mp4",
    "mpg",
    "mpeg",
    "m4v",
    "wmv",
    "mkv",
]  # acceptable video suffixes
NUM_THREADS = os.cpu_count()  # number of multiprocessing threads


def create_dataloader(
    path,
    labels_dir,
    imgsz,
    batch_size,
    stride,
    hyp=None,
    augment=False,
    cache=False,
    pad=0.0,
    rect=False,
    rank=-1,
    workers=8,
    image_weights=False,
    quad=False,
    prefix="",
    kp_flip=None,
    kp_bbox=None,
):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabels(
            path,
            labels_dir,
            imgsz,
            batch_size,
            augment=augment,  # augment images
            hyp=hyp,  # augmentation hyperparameters
            rect=rect,  # rectangular training
            cache_images=cache,
            stride=int(stride),
            pad=pad,
            image_weights=image_weights,
            prefix=prefix,
            kp_flip=kp_flip,
            kp_bbox=kp_bbox,
        )

        # for i in range(10):
        #     dataset.__getitem__(i)
        # import sys
        # sys.exit()

    batch_size = min(batch_size, len(dataset))
    nw = min(
        [os.cpu_count(), batch_size if batch_size > 1 else 0, workers]
    )  # number of workers
    sampler = (
        torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    )
    loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader
    # Use torch.utils.data.DataLoader() if dataset.properties will update during training else InfiniteDataLoader()
    dataloader = loader(
        dataset,
        batch_size=batch_size,
        num_workers=nw,
        sampler=sampler,
        pin_memory=True,
        collate_fn=LoadImagesAndLabels.collate_fn4
        if quad
        else LoadImagesAndLabels.collate_fn,
    )
    return dataloader, dataset


class LoadImages:  # for inference
    def __init__(self, path, img_size=640, stride=32, auto=True):
        p = str(Path(path).absolute())  # os-agnostic absolute path
        if p.endswith(".txt"):
            with open(p, "r") as f:
                files = f.readlines()
            files = [l.strip() for l in files]
        elif "*" in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, "*.*")))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f"ERROR: {p} does not exist")

        images = [x for x in files if x.split(".")[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split(".")[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = "image"
        self.auto = auto
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, (
            f"No images or videos found in {p}. "
            f"Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}"
        )

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = "video"
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            # print(f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: ', end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, "Image Not Found " + path
            print(f"image {self.count}/{self.nf} {path}: ", end="")

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files


class LoadWebcam:  # for inference
    def __init__(self, pipe="0", img_size=640, stride=32):
        self.img_size = img_size
        self.stride = stride
        self.pipe = eval(pipe) if pipe.isnumeric() else pipe
        self.cap = cv2.VideoCapture(self.pipe)  # video capture object
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord("q"):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Read frame
        ret_val, img0 = self.cap.read()
        img0 = cv2.flip(img0, 1)  # flip left-right

        # Print
        assert ret_val, f"Camera Error {self.pipe}"
        img_path = "webcam.jpg"
        print(f"webcam {self.count}: ", end="")

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return img_path, img, img0, None

    def __len__(self):
        return 0


class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, sources="streams.txt", img_size=640, stride=32, auto=True):
        self.mode = "stream"
        self.img_size = img_size
        self.stride = stride

        if os.path.isfile(sources):
            with open(sources, "r") as f:
                sources = [
                    x.strip() for x in f.read().strip().splitlines() if len(x.strip())
                ]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs, self.fps, self.frames, self.threads = (
            [None] * n,
            [0] * n,
            [0] * n,
            [None] * n,
        )
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        self.auto = auto
        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            print(f"{i + 1}/{n}: {s}... ", end="")
            if "youtube.com/" in s or "youtu.be/" in s:  # if source is YouTube video
                check_requirements(("pafy", "youtube_dl"))
                import pafy

                s = pafy.new(s).getbest(preftype="mp4").url  # YouTube URL
            s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
            cap = cv2.VideoCapture(s)
            assert cap.isOpened(), f"Failed to open {s}"
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps[i] = (
                max(cap.get(cv2.CAP_PROP_FPS) % 100, 0) or 30.0
            )  # 30 FPS fallback
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float(
                "inf"
            )  # infinite stream fallback

            _, self.imgs[i] = cap.read()  # guarantee first frame
            self.threads[i] = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(
                f" success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)"
            )
            self.threads[i].start()
        print("")  # newline

        # check for common shapes
        s = np.stack(
            [
                letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[0].shape
                for x in self.imgs
            ]
        )
        self.rect = (
            np.unique(s, axis=0).shape[0] == 1
        )  # rect inference if all shapes equal
        if not self.rect:
            print(
                "WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams."
            )

    def update(self, i, cap):
        # Read stream `i` frames in daemon thread
        n, f, read = (
            0,
            self.frames[i],
            1,
        )  # frame number, frame array, inference every 'read' frame
        while cap.isOpened() and n < f:
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n % read == 0:
                success, im = cap.retrieve()
                self.imgs[i] = im if success else self.imgs[i] * 0
            time.sleep(1 / self.fps[i])  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord(
            "q"
        ):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img0 = self.imgs.copy()
        img = [
            letterbox(
                x, self.img_size, stride=self.stride, auto=self.rect and self.auto
            )[0]
            for x in img0
        ]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None

    def __len__(self):
        return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years


class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(
        self,
        path,
        labels_dir="labels",
        img_size=640,
        batch_size=16,
        augment=False,
        hyp=None,
        rect=False,
        image_weights=False,
        cache_images=False,
        stride=32,
        pad=0.0,
        prefix="",
        kp_flip=None,
        kp_bbox=None,
    ):
        self.labels_dir = labels_dir
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = (
            self.augment and not self.rect
        )  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = Path(path)
        self.albumentations = Albumentations() if augment else None
        self.kp_flip = kp_flip
        self.kp_bbox = kp_bbox
        self.num_coords = len(kp_flip) * 2

        self.obj_flip = None if self.kp_flip is None else convert_flip(self.kp_flip)

        image_and_labels = read_samples(
            self.path, self.num_coords, labels_dir=self.labels_dir
        )

        # Read cache
        [image_and_labels.pop(k) for k in ("version", "results")]  # remove items
        labels, shapes, self.segments = zip(*image_and_labels.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)  # wh
        self.img_files = list(image_and_labels.keys())
        self.label_files = img2label_paths(self.img_files, labels_dir=self.labels_dir)

        n = len(shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(int)  # batch index
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

        # Rectangular Training
        # Reorder the images so that letterbox will apply the least padding possible in the same batch.
        if self.rect:
            self.batch_shapes, reorder_indices = reorder_rectangle_shapes(
                self.shapes, batch_size, img_size, stride, padding=pad
            )
            self.img_files = [self.img_files[i] for i in reorder_indices]
            self.label_files = [self.label_files[i] for i in reorder_indices]
            self.labels = [self.labels[i] for i in reorder_indices]
            self.shapes = self.shapes[reorder_indices]

    def __len__(self):
        return len(self.img_files)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp["mosaic"]
        if mosaic:
            # Load mosaic
            img, labels = load_mosaic(self, index, kp_bbox=self.kp_bbox)
            shapes = None

            # MixUp augmentation
            if random.random() < hyp["mixup"]:
                img, labels = mixup(
                    img, labels, *load_mosaic(self, random.randint(0, self.n - 1))
                )

        else:
            # Load image
            img, (h0, w0), (h, w) = load_image(self, index)

            # Letterbox
            shape = (
                self.batch_shapes[self.batch[index]] if self.rect else self.img_size
            )  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(
                    labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1]
                )

            if self.augment:
                img, labels = random_perspective(
                    img,
                    labels,
                    degrees=hyp["degrees"],
                    translate=hyp["translate"],
                    scale=hyp["scale"],
                    shear=hyp["shear"],
                    perspective=hyp["perspective"],
                    kp_bbox=self.kp_bbox,
                )

        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:] = xyxy2xywhn(
                labels[:, 1:], w=img.shape[1], h=img.shape[0], clip=True, eps=1e-3
            )

        if self.augment:
            # Albumentations
            img, labels = self.albumentations(img, labels)
            nl = len(labels)  # update after albumentations

            # HSV color-space
            augment_hsv(img, hgain=hyp["hsv_h"], sgain=hyp["hsv_s"], vgain=hyp["hsv_v"])

            # Flip up-down
            if random.random() < hyp["flipud"]:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            # Flip left-right
            if random.random() < hyp["fliplr"]:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

                    if self.kp_flip and labels.shape[1] > 5:
                        labels[:, 5::3] = (
                            1 - labels[:, 5::3]
                        )  # flip keypoints in person object
                        keypoints = labels[:, 5:].reshape(nl, -1, 3)
                        keypoints = keypoints[
                            :, self.kp_flip
                        ]  # reorder left / right keypoints
                        labels[:, 5:] = keypoints.reshape(nl, -1)

                    if self.obj_flip:
                        for i, cls in enumerate(labels[:, 0]):
                            labels[i, 0] = self.obj_flip[labels[i, 0]]

        # img_h, img_w = img.shape[:2]
        # img = img.copy()
        # person_obj = labels[labels[:, 0] == 0]
        # for lbl in person_obj:
        #     xc, yc, w, h = lbl[1:5].copy()
        #     pt1 = (int((xc - w / 2) * img_w), int((yc - h / 2) * img_h))
        #     pt2 = (int((xc + w / 2) * img_w), int((yc + h / 2) * img_h))
        #     cv2.rectangle(img, pt1, pt2, (255, 0, 255), thickness=2)
        #
        #     kp = lbl[5:]
        #     kp = np.array(kp).reshape(-1, 3)
        #     kp[:, 0] = kp[:, 0] * img_w
        #     kp[:, 1] = kp[:, 1] * img_h
        #     for i, (x, y, v) in enumerate(kp):
        #         if v:
        #             if i in COCO_KP_LEFT:
        #                 color = (0, 255, 255)
        #             else:
        #                 color = (255, 255, 0)
        #             cv2.circle(img, (int(round(x)), int(round(y))), 2, color, thickness=2)
        #             # cv2.putText(img, COCO_KP_NAMES_SHORT[i], (int(round(x + 10)), int(round(y + 10))),
        #             #             cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), thickness=1)
        # cv2.imshow('', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        labels_out = torch.zeros((nl, labels.shape[-1] + 1))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.img_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

    @staticmethod
    def collate_fn4(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        n = len(shapes) // 4
        img4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]

        ho = torch.tensor([[0.0, 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0.0, 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, 0.5, 0.5, 0.5, 0.5]])  # scale
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4
            if random.random() < 0.5:
                im = F.interpolate(
                    img[i].unsqueeze(0).float(),
                    scale_factor=2.0,
                    mode="bilinear",
                    align_corners=False,
                )[0].type(img[i].type())
                l = label[i]
            else:
                im = torch.cat(
                    (
                        torch.cat((img[i], img[i + 1]), 1),
                        torch.cat((img[i + 2], img[i + 3]), 1),
                    ),
                    2,
                )
                l = (
                    torch.cat(
                        (
                            label[i],
                            label[i + 1] + ho,
                            label[i + 2] + wo,
                            label[i + 3] + ho + wo,
                        ),
                        0,
                    )
                    * s
                )
            img4.append(im)
            label4.append(l)

        for i, l in enumerate(label4):
            l[:, 0] = i  # add target image index for build_targets()

        return torch.stack(img4, 0), torch.cat(label4, 0), path4, shapes4


# Ancillary functions --------------------------------------------------------------------------------------------------
def load_image(self, i):
    # loads 1 image from dataset index 'i', returns im, original hw, resized hw
    path = self.img_files[i]
    im = cv2.imread(path)  # BGR
    assert im is not None, "Image Not Found " + path
    h0, w0 = im.shape[:2]  # orig hw
    r = self.img_size / max(h0, w0)  # ratio
    if r != 1:  # if sizes are not equal
        im = cv2.resize(
            im,
            (int(w0 * r), int(h0 * r)),
            interpolation=cv2.INTER_AREA
            if r < 1 and not self.augment
            else cv2.INTER_LINEAR,
        )
    return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized


def load_mosaic(self, index, kp_bbox=None):
    # loads images in a 4-mosaic

    labels4, segments4 = [], []
    s = self.img_size
    yc, xc = [
        int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border
    ]  # mosaic center x, y
    indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img4
        if i == 0:  # top left
            img4 = np.full(
                (s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8
            )  # base image with 4 tiles
            x1a, y1a, x2a, y2a = (
                max(xc - w, 0),
                max(yc - h, 0),
                xc,
                yc,
            )  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = (
                w - (x2a - x1a),
                h - (y2a - y1a),
                w,
                h,
            )  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(
                labels[:, 1:], w, h, padw, padh
            )  # normalized xywh to pixel xyxy format
            segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
        labels4.append(labels)
        segments4.extend(segments)

    # Concat/clip labels
    labels4 = np.concatenate(labels4, 0)
    for x in (labels4[:, 1:], *segments4):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    # img4, labels4 = replicate(img4, labels4)  # replicate

    # Augment
    img4, labels4, segments4 = copy_paste(
        img4, labels4, segments4, p=self.hyp["copy_paste"]
    )
    img4, labels4 = random_perspective(
        img4,
        labels4,
        segments4,
        degrees=self.hyp["degrees"],
        translate=self.hyp["translate"],
        scale=self.hyp["scale"],
        shear=self.hyp["shear"],
        perspective=self.hyp["perspective"],
        border=self.mosaic_border,
        kp_bbox=kp_bbox,
    )  # border to remove

    return img4, labels4


def load_mosaic9(self, index):
    # loads images in a 9-mosaic

    labels9, segments9 = [], []
    s = self.img_size
    indices = [index] + random.choices(self.indices, k=8)  # 8 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img9
        if i == 0:  # center
            img9 = np.full(
                (s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8
            )  # base image with 4 tiles
            h0, w0 = h, w
            c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
        elif i == 1:  # top
            c = s, s - h, s + w, s
        elif i == 2:  # top right
            c = s + wp, s - h, s + wp + w, s
        elif i == 3:  # right
            c = s + w0, s, s + w0 + w, s + h
        elif i == 4:  # bottom right
            c = s + w0, s + hp, s + w0 + w, s + hp + h
        elif i == 5:  # bottom
            c = s + w0 - w, s + h0, s + w0, s + h0 + h
        elif i == 6:  # bottom left
            c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
        elif i == 7:  # left
            c = s - w, s + h0 - h, s, s + h0
        elif i == 8:  # top left
            c = s - w, s + h0 - hp - h, s, s + h0 - hp

        padx, pady = c[:2]
        x1, y1, x2, y2 = [max(x, 0) for x in c]  # allocate coords

        # Labels
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(
                labels[:, 1:], w, h, padx, pady
            )  # normalized xywh to pixel xyxy format
            segments = [xyn2xy(x, w, h, padx, pady) for x in segments]
        labels9.append(labels)
        segments9.extend(segments)

        # Image
        img9[y1:y2, x1:x2] = img[y1 - pady :, x1 - padx :]  # img9[ymin:ymax, xmin:xmax]
        hp, wp = h, w  # height, width previous

    # Offset
    yc, xc = [
        int(random.uniform(0, s)) for _ in self.mosaic_border
    ]  # mosaic center x, y
    img9 = img9[yc : yc + 2 * s, xc : xc + 2 * s]

    # Concat/clip labels
    labels9 = np.concatenate(labels9, 0)
    labels9[:, [1, 3]] -= xc
    labels9[:, [2, 4]] -= yc
    c = np.array([xc, yc])  # centers
    segments9 = [x - c for x in segments9]

    for x in (labels9[:, 1:], *segments9):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    # img9, labels9 = replicate(img9, labels9)  # replicate

    # Augment
    img9, labels9 = random_perspective(
        img9,
        labels9,
        segments9,
        degrees=self.hyp["degrees"],
        translate=self.hyp["translate"],
        scale=self.hyp["scale"],
        shear=self.hyp["shear"],
        perspective=self.hyp["perspective"],
        border=self.mosaic_border,
    )  # border to remove

    return img9, labels9


def create_folder(path="./new"):
    # Create folder
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder


def flatten_recursive(path="../datasets/coco128"):
    # Flatten a recursive directory by bringing all files to top level
    new_path = Path(path + "_flat")
    create_folder(new_path)
    for file in tqdm(glob.glob(str(Path(path)) + "/**/*.*", recursive=True)):
        shutil.copyfile(file, new_path / Path(file).name)


def extract_boxes(
    path="../datasets/coco128", labels_dir="labels"
):  # from utils.datasets import *; extract_boxes()
    # Convert detection dataset into classification dataset, with one directory per class
    path = Path(path)  # images dir
    shutil.rmtree(path / "classifier") if (
        path / "classifier"
    ).is_dir() else None  # remove existing
    files = list(path.rglob("*.*"))
    n = len(files)  # number of files
    for im_file in tqdm(files, total=n):
        if im_file.suffix[1:] in IMG_FORMATS:
            # image
            im = cv2.imread(str(im_file))[..., ::-1]  # BGR to RGB
            h, w = im.shape[:2]

            # labels
            lb_file = Path(img2label_paths([str(im_file)], labels_dir=labels_dir)[0])
            if Path(lb_file).exists():
                with open(lb_file, "r") as f:
                    lb = np.array(
                        [x.split() for x in f.read().strip().splitlines()],
                        dtype=np.float32,
                    )  # labels

                for j, x in enumerate(lb):
                    c = int(x[0])  # class
                    f = (
                        (path / "classifier")
                        / f"{c}"
                        / f"{path.stem}_{im_file.stem}_{j}.jpg"
                    )  # new filename
                    if not f.parent.is_dir():
                        f.parent.mkdir(parents=True)

                    b = x[1:] * [w, h, w, h]  # box
                    # b[2:] = b[2:].max()  # rectangle to square
                    b[2:] = b[2:] * 1.2 + 3  # pad
                    b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(int)

                    b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
                    b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                    assert cv2.imwrite(
                        str(f), im[b[1] : b[3], b[0] : b[2]]
                    ), f"box failure in {f}"


def autosplit(
    path="../datasets/coco128/images",
    weights=(0.9, 0.1, 0.0),
    annotated_only=False,
    labels_dir="labels_dir",
):
    """Autosplit a dataset into train/val/test splits and save path/autosplit_*.txt files
    Usage: from utils.datasets import *; autosplit()
    Arguments
        path:            Path to images directory
        weights:         Train, val, test weights (list, tuple)
        annotated_only:  Only use images with an annotated txt file
    """
    path = Path(path)  # images dir
    files = sum(
        [list(path.rglob(f"*.{img_ext}")) for img_ext in IMG_FORMATS], []
    )  # image files only
    n = len(files)  # number of files
    random.seed(0)  # for reproducibility
    indices = random.choices(
        [0, 1, 2], weights=weights, k=n
    )  # assign each image to a split

    txt = [
        "autosplit_train.txt",
        "autosplit_val.txt",
        "autosplit_test.txt",
    ]  # 3 txt files
    [(path.parent / x).unlink(missing_ok=True) for x in txt]  # remove existing

    print(
        f"Autosplitting images from {path}"
        + ", using *.txt labeled images only" * annotated_only
    )
    for i, img in tqdm(zip(indices, files), total=n):
        if (
            not annotated_only
            or Path(img2label_paths([str(img)], labels_dir="labels_dir")[0]).exists()
        ):  # check label
            with open(path.parent / txt[i], "a") as f:
                f.write(
                    "./" + img.relative_to(path.parent).as_posix() + "\n"
                )  # add image to txt file


def dataset_stats(
    path="coco128.yaml",
    autodownload=False,
    verbose=False,
    profile=False,
    hub=False,
    labels_dir="labels",
):
    """Return dataset statistics dictionary with images and instances counts per split per class
    To run in parent directory: export PYTHONPATH="$PWD/yolov5"
    Usage1: from utils.datasets import *; dataset_stats('coco128.yaml', autodownload=True)
    Usage2: from utils.datasets import *; dataset_stats('../datasets/coco128_with_yaml.zip')
    Arguments
        path:           Path to data.yaml or data.zip (with data.yaml inside data.zip)
        autodownload:   Attempt to download dataset if not found locally
        verbose:        Print stats dictionary
    """

    def round_labels(labels):
        # Update labels to integer class and 6 decimal place floats
        return [[int(c), *[round(x, 4) for x in points]] for c, *points in labels]

    def unzip(path):
        # Unzip data.zip TODO: CONSTRAINT: path/to/abc.zip MUST unzip to 'path/to/abc/'
        if str(path).endswith(".zip"):  # path is data.zip
            assert Path(path).is_file(), f"Error unzipping {path}, file not found"
            assert (
                os.system(f"unzip -q {path} -d {path.parent}") == 0
            ), f"Error unzipping {path}"
            dir = path.with_suffix("")  # dataset directory
            return (
                True,
                str(dir),
                next(dir.rglob("*.yaml")),
            )  # zipped, data_dir, yaml_path
        else:  # path is data.yaml
            return False, None, path

    def hub_ops(f, max_dim=1920):
        # HUB ops for 1 image 'f'
        im = Image.open(f)
        r = max_dim / max(im.height, im.width)  # ratio
        if r < 1.0:  # image too large
            im = im.resize((int(im.width * r), int(im.height * r)))
        im.save(im_dir / Path(f).name, quality=75)  # save

    zipped, data_dir, yaml_path = unzip(Path(path))
    with open(check_file(yaml_path), errors="ignore") as f:
        data = yaml.safe_load(f)  # data dict
        if zipped:
            data["path"] = data_dir  # TODO: should this be dir.resolve()?
    check_dataset(data, autodownload)  # download dataset if missing
    hub_dir = Path(data["path"] + ("-hub" if hub else ""))
    stats = {"nc": data["nc"], "names": data["names"]}  # statistics dictionary
    for split in "train", "val", "test":
        if data.get(split) is None:
            stats[split] = None  # i.e. no test set
            continue
        x = []
        dataset = LoadImagesAndLabels(
            data[split], labels_dir=labels_dir
        )  # load dataset
        for label in tqdm(dataset.labels, total=dataset.n, desc="Statistics"):
            x.append(np.bincount(label[:, 0].astype(int), minlength=data["nc"]))
        x = np.array(x)  # shape(128x80)
        stats[split] = {
            "instance_stats": {"total": int(x.sum()), "per_class": x.sum(0).tolist()},
            "image_stats": {
                "total": dataset.n,
                "unlabelled": int(np.all(x == 0, 1).sum()),
                "per_class": (x > 0).sum(0).tolist(),
            },
            "labels": [
                {str(Path(k).name): round_labels(v.tolist())}
                for k, v in zip(dataset.img_files, dataset.labels)
            ],
        }

        if hub:
            im_dir = hub_dir / "images"
            im_dir.mkdir(parents=True, exist_ok=True)
            for _ in tqdm(
                ThreadPool(NUM_THREADS).imap(hub_ops, dataset.img_files),
                total=dataset.n,
                desc="HUB Ops",
            ):
                pass

    # Profile
    stats_path = hub_dir / "stats.json"
    if profile:
        for _ in range(1):
            file = stats_path.with_suffix(".npy")
            t1 = time.time()
            np.save(file, stats)
            t2 = time.time()
            x = np.load(file, allow_pickle=True)
            print(
                f"stats.npy times: {time.time() - t2:.3f}s read, {t2 - t1:.3f}s write"
            )

            file = stats_path.with_suffix(".json")
            t1 = time.time()
            with open(file, "w") as f:
                json.dump(stats, f)  # save stats *.json
            t2 = time.time()
            with open(file, "r") as f:
                x = json.load(f)  # load hyps dict
            print(
                f"stats.json times: {time.time() - t2:.3f}s read, {t2 - t1:.3f}s write"
            )

    # Save, print and return
    if hub:
        print(f"Saving {stats_path.resolve()}...")
        with open(stats_path, "w") as f:
            json.dump(stats, f)  # save stats.json
    if verbose:
        print(json.dumps(stats, indent=2, sort_keys=False))
    return stats
