import glob
import random
import os
import sys

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from utils.augmentations import horisontal_flip


class ImageDataset(Dataset):
    """Add the functions to the original Dataset class"""

    def pad_to_square(self, img, pad_value):
        """
        Args:
            img: image in format c, h, w
            pad_value: pixel value used to pad the image

        Returns:
            img_padded: padded square image in foramt c, h, w, h=w
            pad: amount of padding in pixel integers on 4 sides.
        """
        c, h, w = img.shape
        dim_diff = np.abs(h - w)
        # (upper / left) padding and (lower / right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
        # Add padding
        img_padded = F.pad(img, pad, "constant", value=pad_value)
        return img_padded, pad

    def adjust_bbox_to_pad(self, img_padded, boxes, pad, w_factor, h_factor):
        """
        Args:
            img_padded: padded square image in c, h, w
            boxes: labels in format (class, x1, y1, x2, y2)
            pad: amountof padding in integers on 4 sides
            w_factor: unpadded image scaling factor
            h_factor: unpadded image scaling factor

        Returns:
            boxes
        """
        _, padded_h, padded_w = img_padded.shape
        # Extract coordinates for unpadded + unscaled image
        x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
        y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
        x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
        y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
        # Adjust for added padding
        x1 += pad[0]
        y1 += pad[2]
        x2 += pad[1]
        y2 += pad[3]
        # Returns (x, y, w, h)
        boxes[:, 1] = ((x1 + x2) / 2) / padded_w
        boxes[:, 2] = ((y1 + y2) / 2) / padded_h
        boxes[:, 3] *= w_factor / padded_w
        boxes[:, 4] *= h_factor / padded_h
        return boxes

    def resize(self, img, size):
        img = F.interpolate(img.unsqueeze(0), size=size, mode="nearest").squeeze(0)
        return img

    def random_resize(self, imgs, min_size=288, max_size=448):
        new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
        imgs = F.interpolate(imgs, size=new_size, mode="nearest")
        return imgs

    def ps_to_ds(self, imgs, labels):
        """convert labels of a batch from percentage scale to diagram scale
        Args:
            imgs: (B, C, H, W) in the format of (x, y, w, h)
            labels: (# of labels in batch,  6)
        Returns:
        """
        for i in range(imgs.shape[0]):
            img_i = imgs[i, ::]
            labels_i = labels[labels[:, 0] == i]
            labels_i[:, [2, 4]] *= img_i.shape[2]
            labels_i[:, [3, 5]] *= img_i.shape[1]
            labels[labels[:, 0] == i] = labels_i
        return labels

    def xywh2xyxy(self, boxes):
        """boxes in (batch_i, label, x, y, w, h)"""
        a = boxes[:,2:4]
        b = boxes[:,4:]
        boxes[:, 2:4] = a-b/2
        boxes[:, 4:] = a+ b/2
        return boxes


class ImageFolder(ImageDataset):
    """Load all the images in a folder, labels not required"""
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = self.resize(img, self.img_size)
        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(ImageDataset):
    """
    Dataset with path to all images provided in a list.
    The following rules have to be true for data to load:
        - Images are located in a folder called <images>
        - Images are in either .jpg or .png format
        - labels for the images are located in a folder called <labels>
        - labels are text files.
        - Each image has its own annotaion file
    """

    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True, xyxy=False):
        self.img_files = []
        self.label_files = []
        self.img_size = img_size
        self.max_objects = 100
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.load_list(list_path)
        self.xyxy = xyxy

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        """
            imgs: c, h, w
            targets: class, x, y, w, h
        """
        #-----------------Image------------------
        img = self._load_image(index)
        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = self.pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape
        #-----------------Label------------------
        label_path = self.label_files[index % len(self.img_files)].rstrip()
        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            boxes = self.adjust_bbox_to_pad(img, boxes, pad, w_factor, h_factor)
            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes
        # Apply augmentations
        if self.augment:
            #TODO
            pass
        return img, targets

    def collate_fn(self, batch):
        """Format a batch of data
            imgs: batch_size, c, h, w
            targets: batch_num, class, x, y, w, h
        """
        imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        #! One underlying assumption is that all training images are labeled
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        # Stack the targets of the batch
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([self.resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        # Label post processing
        # Convert from xywh label to xyxy label if needed
        if self.xyxy:
            targets = self.xywh2xyxy(targets)
        return imgs, targets

    def load_list(self, list_path):
        """Load the list containing the paths to images"""
        # Load paths to images
        with open(list_path, "r") as file:
            self.img_files = file.readlines()
        # Load annotation for each image
        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]

    def _load_image(self, index):
        """Load an individual image based on index"""
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))
        return img
