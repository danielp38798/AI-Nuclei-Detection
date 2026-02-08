from collections import defaultdict

import torch
import transforms as reference_transforms


from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision import tv_tensors, models
from torchvision.transforms.v2 import functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
def plot(imgs, row_title=None, out_file=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            boxes = None
            masks = None
            if isinstance(img, tuple):
                img, target = img
                if isinstance(target, dict):
                    boxes = target.get("boxes")
                    masks = target.get("masks")
                elif isinstance(target, tv_tensors.BoundingBoxes):
                    boxes = target
                else:
                    raise ValueError(f"Unexpected target type: {type(target)}")
            img = F.to_image(img)
            if img.dtype.is_floating_point and img.min() < 0:
                # Poor man's re-normalization for the colors to be OK-ish. This
                # is useful for images coming out of Normalize()
                img -= img.min()
                img /= img.max()

            img = F.to_dtype(img, torch.uint8, scale=True)
            if boxes is not None:
                img = draw_bounding_boxes(img, boxes, colors="yellow", width=3)
            if masks is not None:
                img = draw_segmentation_masks(img, masks.to(torch.bool), colors=["green"] * masks.shape[0], alpha=.35)
                # draw polygonal masks using cv2 findContours
                masks_np = masks.cpu().numpy() # (N, H, W)
                #masks_np = np.moveaxis(masks_np, 0, -1) # (H, W, N)
                masks_np = np.ascontiguousarray(masks_np)
                img = img.permute(1, 2, 0).numpy()
                for mask in masks_np:
                    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
                img = torch.from_numpy(img).permute(2, 0, 1)


            ax = axs[row_idx, col_idx]
            ax.imshow(img.permute(1, 2, 0).numpy(), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
    file_path = os.path.join(os.getcwd(), out_file)
    plt.savefig(file_path)
    print(f"Saved to {file_path}")
    plt.close

def get_modules(use_v2):
    # We need a protected import to avoid the V2 warning in case just V1 is used
    if use_v2:
        import torchvision.transforms.v2
        import torchvision.tv_tensors

        return torchvision.transforms.v2, torchvision.tv_tensors
    else:
        return reference_transforms, None


class DetectionPresetTrain:
    # Note: this transform assumes that the input to forward() are always PIL
    # images, regardless of the backend parameter.
    def __init__(
        self,
        *,
        data_augmentation,
        hflip_prob=0.5,
        mean=(123.0, 117.0, 104.0),
        backend="pil",
        use_v2=False,
    ):

        T, tv_tensors = get_modules(use_v2)

        transforms = []
        backend = backend.lower()
        if backend == "tv_tensor":
            transforms.append(T.ToImage())
        elif backend == "tensor":
            transforms.append(T.PILToTensor())
        elif backend != "pil":
            raise ValueError(f"backend can be 'tv_tensor', 'tensor' or 'pil', but got {backend}")

        if data_augmentation == "hflip":
            transforms += [T.RandomHorizontalFlip(p=hflip_prob)]
        elif data_augmentation == "lsj":
            # output size is 1024x1024
            transforms += [
                T.ScaleJitter(target_size=(1024, 1024), antialias=True),
                # TODO: FixedSizeCrop below doesn't work on tensors!
                reference_transforms.FixedSizeCrop(size=(1024, 1024), fill=mean),
                T.RandomHorizontalFlip(p=hflip_prob),
            ]
        elif data_augmentation == "multiscale":
            transforms += [
                T.RandomShortestSize(min_size=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800), max_size=1333),
                T.RandomHorizontalFlip(p=hflip_prob),
            ]
        elif data_augmentation == "ssd":
            fill = defaultdict(lambda: mean, {tv_tensors.Mask: 0}) if use_v2 else list(mean)
            transforms += [
                T.RandomPhotometricDistort(),
                T.RandomZoomOut(fill=fill),
                T.RandomIoUCrop(),
                T.RandomHorizontalFlip(p=hflip_prob),
            ]
        elif data_augmentation == "ssdlite":
            transforms += [
                T.RandomIoUCrop(),
                T.RandomHorizontalFlip(p=hflip_prob),
            ]
        elif data_augmentation == "maskrcnn":
            transforms += [
                #T.RandomPhotometricDistort(),
                #T.RandomZoomOut(fill=mean),
                T.RandomHorizontalFlip(p=hflip_prob),
                T.PILToTensor(),
                T.ToDtype(torch.float, scale=True),
                T.Normalize(mean=[0.1444, 0.1444, 0.1444], std=[0.1280, 0.1280, 0.1280])
            ]
        elif data_augmentation == "advanced_maskrcnn":
             transforms += [
                T.RandomPhotometricDistort(),
                T.RandomIoUCrop(),
                T.RandomHorizontalFlip(p=hflip_prob),
                #T.RandomVerticalFlip(p=hflip_prob),
                T.RandomRotation(degrees=(-15,15)),  # Random rotation by ±15 degrees
                #T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)), #Applies random affine transformations including translation and scaling.
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), #Randomly change the brightness, contrast, saturation and hue of an image.
                #T.GaussianBlur(kernel_size=3), #Gaussian blur with kernel size 3
                #T.RandomResizedCrop(size=(400, 400), scale=(0.8, 1.0), ratio=(0.9, 1.1)), #Random crop to (400, 400) size with scale and ratio
                T.PILToTensor(),
                T.ToDtype(torch.float, scale=True),
                T.Normalize(mean=[0.1444, 0.1444, 0.1444], std=[0.1280, 0.1280, 0.1280])
            ]
        elif data_augmentation == "none":
            # only normalize
            transforms += [
                T.PILToTensor(),
                T.ToDtype(torch.float, scale=True),
                T.Normalize(mean=[0.1444, 0.1444, 0.1444], 
                            std=[0.1280, 0.1280, 0.1280])]

        else:
            raise ValueError(f'Unknown data augmentation policy "{data_augmentation}"')

        if backend == "pil":
            # Note: we could just convert to pure tensors even in v2.
            transforms += [T.ToImage() if use_v2 else T.PILToTensor()]

        transforms += [T.ToDtype(torch.float, scale=True)]

        if use_v2:
            transforms += [
                T.ConvertBoundingBoxFormat(tv_tensors.BoundingBoxFormat.XYXY),
                T.SanitizeBoundingBoxes(),
                T.ToPureTensor(),
            ]

        self.transforms = T.Compose(transforms)
        #print("TRAIN TRANSFORMS: ", self.transforms)

    def __call__(self, img, target):

        transformed_img, transformed_target = self.transforms(img, target)

        # uncomment/comment the following line to visualize/hide the transformed image and target
        # plot([(img, target), (transformed_img, transformed_target)], row_title=["original", "transformed"], out_file=f"transformed_img_{time.time()}.png")
        return transformed_img, transformed_target


class DetectionPresetEval:
    def __init__(self, backend="pil", use_v2=False):
        T, _ = get_modules(use_v2)
        transforms = []
        backend = backend.lower()
        if backend == "pil":
            print("Using PIL")
            # Note: we could just convert to pure tensors even in v2?
            transforms += [
                T.PILToTensor(),
                T.ToDtype(torch.float, scale=True),
                T.Normalize(mean=[0.1444, 0.1444, 0.1444], std=[0.1280, 0.1280, 0.1280]),
                T.ToImage() if use_v2 else T.PILToTensor()]
        elif backend == "tensor":
            transforms += [T.PILToTensor(),
                            T.ToDtype(torch.float, scale=True),
                            T.Normalize(mean=[0.1444, 0.1444, 0.1444], std=[0.1280, 0.1280, 0.1280]),
                           ]
        elif backend == "tv_tensor":
            transforms += [T.ToImage(), 
                            T.ToDtype(torch.float, scale=True),
                            T.Normalize(mean=[0.1444, 0.1444, 0.1444], std=[0.1280, 0.1280, 0.1280])
                            ]
        else:
            raise ValueError(f"backend can be 'tv_tensor', 'tensor' or 'pil', but got {backend}")
        
        transforms += [T.ToDtype(torch.float, scale=True)]

        if use_v2:
            transforms += [T.ToPureTensor()]

        self.transforms = T.Compose(transforms)
        #print("EVAL TRANSFORMS: ", self.transforms)

    def __call__(self, img, target):
        return self.transforms(img, target)
