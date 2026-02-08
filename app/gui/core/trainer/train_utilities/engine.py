import math
import sys
import time
import torch
import torchvision.models.detection.mask_rcnn
import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset

import os
import numpy as np
import matplotlib.pyplot as plt


from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision import tv_tensors, models
from torchvision.transforms.v2 import functional as F
import cv2
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
                    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
                img = torch.from_numpy(img).permute(2, 0, 1)


            ax = axs[row_idx, col_idx]
            ax.imshow(img.permute(1, 2, 0).numpy(), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
    plt.savefig(out_file)

class EarlyStopping:
    def __init__(self, patience=5, delta=0, path='early_stopping_checkpoint.pth'):
        """
        Implement early stopping of the training process.
        Args:
            patience (int): Number of epochs to wait before stopping the training process.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than delta, will count as no improvement.
                            Example: if delta=0.01, the training process will stop if the validation loss does not decrease by at least 0.01.
                                        if delta=0, the training process will stop if the validation loss does not decrease by any amount, 
                                        i.e. the training process will stop if the validation loss does not decrease at all.
            path (str): Path to save the model checkpoint when the validation loss decreases.
                
        Smaller delta values can effectively increase the patience of the EarlyStopping mechanism.
        In the EarlyStopping callback, the counter is incremented when the current validation loss is not decreasing by an amount greater than the delta 
        compared to the best validation loss. If the delta is smaller, it means that smaller improvements in validation loss are considered significant, and thus the counter is reset less frequently.
        Consequently, smaller delta values lead to a more conservative EarlyStopping behavior, as the training process will continue for a 
        longer duration without improvement in the validation loss. This effectively increases the "patience" of the EarlyStopping mechanism, allowing the 
        training process more time to find better parameter settings before stopping.
        
        """
        
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.storage_path = path

    def __call__(self, val_loss, model, optimizer, epoch, lr_scheduler):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model, optimizer, epoch, lr_scheduler)
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model, optimizer, epoch, lr_scheduler)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer, epoch, lr_scheduler):
        '''Saves model when validation loss decreases.'''
        if val_loss < self.val_loss_min:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            self.val_loss_min = val_loss
            # You can save the model state dict or anything you want here
            checkpoint = {
                'val_loss_min': val_loss,
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict()
            }
            es_ckpt_path = os.path.join(self.storage_path, 'early_stopping_checkpoint.pth')
            torch.save(checkpoint, es_ckpt_path)
            # save model
            es_model_path = os.path.join(self.storage_path, 'early_stopping_model.pth')
            torch.save(model.state_dict(), es_model_path)
        print(f'Current best loss: {self.val_loss_min:.6f}')



def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
    #for idx, (images, targets) in enumerate(data_loader):

        #if idx == 0:
            #print(f"images: {images}")
            #print(f"targets: {targets}")
            #plot([(images[-1],targets[-1])], out_file=f"disk/AI_nuclei_detection/trainer/detection/sample_{time.time()}.png")
            #sys.exit()

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator

@torch.no_grad()
def validate(model, data_loader, device, print_freq, scaler=None):
    """
    Evaluate the model on the validation set.
    Sets the model to training mode to get the losses but doesn't perform backpropagation.
    This assures that the weights are not updated during validation.
    """

    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("loss", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Validation:"


    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        #  check if any of targets does not have any bounding boxes
        # if so, remove the target and the corresponding image from the batch
        new_images = []
        new_targets = []
        for img, target in zip(images, targets):
            if "boxes" in target:
                new_images.append(img)
                new_targets.append(target)
        images = new_images
        targets = new_targets

        if len(images) == 0 or len(targets) == 0:
            continue
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())


        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)

    return metric_logger