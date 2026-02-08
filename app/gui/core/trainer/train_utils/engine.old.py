import math
import sys
import time
import numpy as np
import cv2
import torch
import torchvision.models.detection.mask_rcnn

from gui.core.trainer.train_utils.coco_utils import *
from gui.core.trainer.train_utils.coco_eval import CocoEvaluator
from gui.core.trainer.train_utils import utils
from gui.core.trainer.train_utils.utils import MetricLogger, SmoothedValue, ImageTiler, reduce_dict

import matplotlib.patches as patches
import matplotlib.pyplot as plt

def is_tensor_empty(tensor):
    return tensor.numel() == 0

#scaler = torch.cuda.amp.GradScaler()


def plot_image_with_annotations( patch, target):
    """
    Plot the image patch along with bounding boxes and masks.

    Parameters:
    - patch (torch.Tensor): Image patch tensor with shape (C, H, W).
    - annotations (dict): Dictionary containing 'boxes', 'labels', and 'masks'.
    """

    # Ensure the image tensor has the correct shape (H, W, C)
    patch = patch.permute(1, 2, 0).numpy().astype(np.uint8)

    bboxes = target['boxes']
    masks = target['masks']
    labels = target['labels']
    for idx, bbox in enumerate(bboxes):
        xmin, ymin, xmax, ymax = bbox
        cv2.rectangle(patch, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)
        #cv2.putText(patch, str(labels[idx]), (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    for mask in masks:
        mask = mask.numpy()
        # overlay mask on patch

        # Find contours of the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Draw contours with reduced opacity on the original image
        contours_image = np.zeros_like(patch)
        cv2.drawContours(contours_image, contours, -1, (0, 255, 0), 2)

        # Blend the drawn contours with the original image
        alpha = 0.2  # Set the desired opacity value (adjust as needed)
        patch = cv2.addWeighted(patch, 1, contours_image, alpha, 0)

    # Create a figure and axis
    fig, ax = plt.subplots(1)
    ax.imshow(patch)

    # Set plot title
    ax.set_title('Image with Annotations')

    plt.savefig('patch_1.png')
    #print(f"Plot saved at patch_1.png")


def train_one_epoch_old(model, optimizer, data_loader, device, epoch, writer, perform_patch_training, print_freq, scaler=None):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    early_stopping = None
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

        #early_stopping = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, 
                        #verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

    losses_dict = {
            'loss_classifier': [],
            'loss_box_reg': [],
            'loss_mask': [],
            'loss_objectness': [],
            'loss_rpn_box_reg': [],
            'loss': []
    }
    
    average_losses_dict = {
            'loss_classifier': [],
            'loss_box_reg': [],
            'loss_mask': [],
            'loss_objectness': [],
            'loss_rpn_box_reg': [],
            'loss': []
    }

    if perform_patch_training:
        print(f"\nPerforming patch training ...\n")
        for batch in metric_logger.log_every(data_loader, print_freq, header):
            all_loss_dicts = []
            #print(f"batch: {batch}")
            patches = batch[0]
            targets = batch[1]
            loss_classifier_list = []
            loss_box_reg_list = []
            loss_mask_list = []
            loss_objectness_list = []
            loss_rpn_box_reg_list = []
            total_image_loss_list = []

            for patch, target in zip(patches, targets):

                #plot_image_with_annotations(patch, target)
                if is_tensor_empty(target["boxes"]):
                    #print("no boxes - skipping patch")
                    continue
                if is_tensor_empty(target["masks"]):
                    #print("no masks - skipping patch")
                    continue

                patch = patch.to(device)
                patch_target = {k: v.to(device) for k, v in target.items()}
                #print(f"current tile shape: {tile.shape}")
                #with torch.cuda.amp.autocast(enabled=scaler is not None):
                loss_dict = model([patch], [patch_target])
                losses = sum(loss for loss in loss_dict.values())
                #print(f"loss_dict: {loss_dict}")

                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = reduce_dict(loss_dict)
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                #loss_value = losses_reduced.item()
            
                loss_classifier = loss_dict['loss_classifier'].item()
                loss_box_reg= loss_dict['loss_box_reg'].item()
                loss_mask = loss_dict['loss_mask'].item()
                loss_objectness = loss_dict['loss_objectness'].item()
                loss_rpn_box_reg = loss_dict['loss_rpn_box_reg'].item()
                
                # calculate total loss
                loss_value = loss_classifier + loss_box_reg + loss_mask + loss_objectness + loss_rpn_box_reg
                total_image_loss_list.append(loss_value)
                
                #print("\n")
                #print(f"loss_classifier: {loss_classifier}")
                #print(f"loss_box_reg: {loss_box_reg}")
                #print(f"loss_mask: {loss_mask}")
                #print(f"loss_objectness: {loss_objectness}")
                #print(f"loss_rpn_box_reg: {loss_rpn_box_reg}")
                #print(f"total loss: {loss_value}")
                #print("\n")
                
                loss_classifier_list.append(loss_classifier)
                loss_box_reg_list.append(loss_box_reg)
                loss_mask_list.append(loss_mask)
                loss_objectness_list.append(loss_objectness)
                loss_rpn_box_reg_list.append(loss_rpn_box_reg)
        

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

                metric_logger.update(loss=loss_value, **loss_dict_reduced)
                metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            # store the average loss for this image
            losses_dict['loss_classifier'].append(np.mean(loss_classifier_list))
            losses_dict['loss_box_reg'].append(np.mean(loss_box_reg_list))
            losses_dict['loss_mask'].append(np.mean(loss_mask_list))
            losses_dict['loss_objectness'].append(np.mean(loss_objectness_list))
            losses_dict['loss_rpn_box_reg'].append(np.mean(loss_rpn_box_reg_list))
            losses_dict['loss'].append(np.mean(total_image_loss_list))
            print(f"total loss (averange over {len(patches)} patches): {losses_dict['loss']}")
            #print(losses_dict)
        
        # calculate average loss over all images
        for key in losses_dict.keys():
            average_losses_dict[key] = sum(losses_dict[key]) / len(losses_dict[key])
            
        print(f"average classifier loss over all images: {average_losses_dict['loss_classifier']}")
        print(f"average box reg loss over all images: {average_losses_dict['loss_box_reg']}")
        print(f"average mask loss over all images: {average_losses_dict['loss_mask']}")
        print(f"average objectness loss over all images: {average_losses_dict['loss_objectness']}")
        print(f"average rpn box reg loss over all images: {average_losses_dict['loss_rpn_box_reg']}")
        print(f"average loss over all images (TOTAL): {average_losses_dict['loss']}")

            
        return metric_logger, losses_dict, average_losses_dict

    else:

        print("\nComplete image training")

        for images, targets in metric_logger.log_every(data_loader, print_freq, header):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                #loss_dict = model(images, targets)
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_dict(loss_dict)
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

            if early_stopping is not None:
                early_stopping.step(loss_value)

            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            losses_dict['loss_classifier'].append(loss_dict_reduced['loss_classifier'].item())
            losses_dict['loss_box_reg'].append(loss_dict_reduced['loss_box_reg'].item())
            losses_dict['loss_mask'].append(loss_dict_reduced['loss_mask'].item())
            losses_dict['loss_objectness'].append(loss_dict_reduced['loss_objectness'].item())
            losses_dict['loss_rpn_box_reg'].append(loss_dict_reduced['loss_rpn_box_reg'].item())
            losses_dict['loss'].append(losses_reduced.item())

        # calculate average loss
        for key in losses_dict.keys():
            average_losses_dict[key] = sum(losses_dict[key]) / len(losses_dict[key])

    return metric_logger, losses_dict, average_losses_dict



def train_one_epoch(model, optimizer, lr_scheduler, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    # LinearLR scheduler
    """
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    """

    
    losses_dict = {
            'loss_classifier': [],
            'loss_box_reg': [],
            'loss_mask': [],
            'loss_objectness': [],
            'loss_rpn_box_reg': [],
            'loss': []
    }

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        #print(f"images: {images}")
        #print("images[0].shape: ", images[0].shape)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        #print(f"targets: {targets}")
        #print("targets[0]['boxes'].shape: ", targets[0]['boxes'].shape)
        if targets[0]['boxes'].shape[0] == 0:
            print("no boxes - skipping image")
            continue
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        
        loss_classifier = loss_dict['loss_classifier'].item()
        loss_box_reg= loss_dict['loss_box_reg'].item()
        loss_mask = loss_dict['loss_mask'].item()
        loss_objectness = loss_dict['loss_objectness'].item()
        loss_rpn_box_reg = loss_dict['loss_rpn_box_reg'].item()

        """
        
        print("\n")
        print(f"loss_classifier: {loss_classifier}")
        print(f"loss_box_reg: {loss_box_reg}")
        print(f"loss_mask: {loss_mask}")
        print(f"loss_objectness: {loss_objectness}")
        print(f"loss_rpn_box_reg: {loss_rpn_box_reg}")
        print("\n")
        print(f"total loss: {loss_value}")
        
        """
        
        # calculate total loss
        #loss_value = loss_classifier + loss_box_reg + loss_mask + loss_objectness + loss_rpn_box_reg
        
        #store the loss values
        losses_dict['loss_classifier'].append(loss_classifier)
        losses_dict['loss_box_reg'].append(loss_box_reg)
        losses_dict['loss_mask'].append(loss_mask)
        losses_dict['loss_objectness'].append(loss_objectness)
        losses_dict['loss_rpn_box_reg'].append(loss_rpn_box_reg)
        losses_dict['loss'].append(loss_value)

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
    
    # calculate average loss
    average_losses_dict = {}
    for key in losses_dict.keys():
        average_losses_dict[key] = sum(losses_dict[key]) / len(losses_dict[key])


    return metric_logger, average_losses_dict

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

        #print(f"images: {images}")  
        #print("images[0].shape: ", images[0].shape) 
        #print(f"targets: {targets}")    
        
        if targets[0]['boxes'].shape[0] == 0:
            print("no boxes - skipping image")
            continue

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


### old functions ###

@torch.inference_mode()
def evaluate_old(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
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

    coco_results = {}
    
    # Check if there are results for bbox
    if 'bbox' in coco_evaluator.coco_eval:
        # store the results in a dict that can be used by TensorBoard
        results_bbox = {}
        results_bbox['AP'] = coco_evaluator.coco_eval['bbox'].stats[0]
        results_bbox['AP_50'] = coco_evaluator.coco_eval['bbox'].stats[1]
        results_bbox['AP_75'] = coco_evaluator.coco_eval['bbox'].stats[2]
        results_bbox['AP_small'] = coco_evaluator.coco_eval['bbox'].stats[3]
        results_bbox['AP_medium'] = coco_evaluator.coco_eval['bbox'].stats[4]
        results_bbox['AP_large'] = coco_evaluator.coco_eval['bbox'].stats[5]
        results_bbox['AR_1'] = coco_evaluator.coco_eval['bbox'].stats[6]
        results_bbox['AR_10'] = coco_evaluator.coco_eval['bbox'].stats[7]
        results_bbox['AR_100'] = coco_evaluator.coco_eval['bbox'].stats[8]
        results_bbox['AR_small'] = coco_evaluator.coco_eval['bbox'].stats[9]
        results_bbox['AR_medium'] = coco_evaluator.coco_eval['bbox'].stats[10]
        results_bbox['AR_large'] = coco_evaluator.coco_eval['bbox'].stats[11]
        coco_results['bbox'] = results_bbox
    else:
        print("No results found for bbox evaluation.")

    # Check if there are results for seg
    if 'segm' in coco_evaluator.coco_eval:
        results_seg = {}
        results_seg['AP'] = coco_evaluator.coco_eval['segm'].stats[0]
        results_seg['AP_50'] = coco_evaluator.coco_eval['segm'].stats[1]
        results_seg['AP_75'] = coco_evaluator.coco_eval['segm'].stats[2]
        results_seg['AP_small'] = coco_evaluator.coco_eval['segm'].stats[3]
        results_seg['AP_medium'] = coco_evaluator.coco_eval['segm'].stats[4]
        results_seg['AP_large'] = coco_evaluator.coco_eval['segm'].stats[5]
        results_seg['AR_1'] = coco_evaluator.coco_eval['segm'].stats[6]
        results_seg['AR_10'] = coco_evaluator.coco_eval['segm'].stats[7]
        results_seg['AR_100'] = coco_evaluator.coco_eval['segm'].stats[8]
        results_seg['AR_small'] = coco_evaluator.coco_eval['segm'].stats[9]
        results_seg['AR_medium'] = coco_evaluator.coco_eval['segm'].stats[10]
        results_seg['AR_large'] = coco_evaluator.coco_eval['segm'].stats[11]
        coco_results['segm'] = results_seg
    else:
        print("No results found for seg evaluation.")

    return coco_evaluator, coco_results


@torch.inference_mode()
def evaluate_old_2(model, data_loader, device, perform_patch_evaluation):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)


    results_bbox = {
                'AP': [],
                'AP_50': [],
                'AP_75': [],
                'AP_small': [],
                'AP_medium': [],
                'AP_large': [],
                'AR_1': [],
                'AR_10': [],
                'AR_100': [],
                'AR_small': [],
                'AR_medium': [],
                'AR_large': []
        }
        
    results_seg = {
                'AP': [],
                'AP_50': [],
                'AP_75': [],
                'AP_small': [],
                'AP_medium': [],
                'AP_large': [],
                'AR_1': [],
                'AR_10': [],
                'AR_100': [],
                'AR_small': [],
                'AR_medium': [],
                'AR_large': []
        }

    if perform_patch_evaluation:
        print(f"\nPerforming patch evaluation ...\n")
        for batch in metric_logger.log_every(data_loader, 100, header):
            
            patches = batch[0]
            targets = batch[1]
            
            patch_num = 0
            
            for patch, target in zip(patches, targets):
                
                #print(f"target: {target}")

                #plot_image_with_annotations(patch, target)
                if is_tensor_empty(target["boxes"]):
                    #print("no boxes - skipping patch")
                    continue
                if is_tensor_empty(target["masks"]):
                    #print("no masks - skipping patch")
                    continue
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                model_time = time.time()
                patch = patch.to(device)
                outputs = model([patch])
                
                outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
                model_time = time.time() - model_time

                # Add the results to the dictionary
                image_id = target["image_id"].item()
               
                # perform evaluation on the patch
                # check if there are boxes in the prediction
                if is_tensor_empty(outputs[0]["boxes"]):
                    #print("no boxes - skipping patch")
                    continue
                else:
                    evaluator_time = time.time()
                    res = {patch_num: output for output in outputs}
                    print(f"res: {res}")
     
                    coco_evaluator.update(res)
                    evaluator_time = time.time() - evaluator_time
                    metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
                    patch_num += 1
                    
    
            
            # gather the stats from all processes
            #metric_logger.synchronize_between_processes()
            #print("Averaged stats:", metric_logger)
            #coco_evaluator.synchronize_between_processes()

            # accumulate predictions from all patches
            coco_evaluator.accumulate()
            coco_evaluator.summarize()
            torch.set_num_threads(n_threads)
            
              # Check if there are results for bbox
            if 'bbox' in coco_evaluator.coco_eval and coco_evaluator.coco_eval['bbox'].stats is not None:
                
                print(f"stats: {coco_evaluator.coco_eval['bbox'].stats}")
                # store the results in a dict that can be used by TensorBoard
                results_bbox['AP'].append(coco_evaluator.coco_eval['bbox'].stats[0])
                results_bbox['AP_50'].append(coco_evaluator.coco_eval['bbox'].stats[1])
                results_bbox['AP_75'].append(coco_evaluator.coco_eval['bbox'].stats[2])
                results_bbox['AP_small'].append(coco_evaluator.coco_eval['bbox'].stats[3])
                results_bbox['AP_medium'].append(coco_evaluator.coco_eval['bbox'].stats[4])
                results_bbox['AP_large'].append(coco_evaluator.coco_eval['bbox'].stats[5])
                results_bbox['AR_1'].append(coco_evaluator.coco_eval['bbox'].stats[6])
                results_bbox['AR_10'].append(coco_evaluator.coco_eval['bbox'].stats[7])
                results_bbox['AR_100'].append(coco_evaluator.coco_eval['bbox'].stats[8])
                results_bbox['AR_small'].append(coco_evaluator.coco_eval['bbox'].stats[9])
                results_bbox['AR_medium'].append(coco_evaluator.coco_eval['bbox'].stats[10])
                results_bbox['AR_large'].append(coco_evaluator.coco_eval['bbox'].stats[11])

            else:
                print("No results found for bbox evaluation.")

            # Check if there are results for seg
            if 'segm' in coco_evaluator.coco_eval and coco_evaluator.coco_eval['segm'].stats is not None:
                results_seg['AP'].append(coco_evaluator.coco_eval['segm'].stats[0])
                results_seg['AP_50'].append(coco_evaluator.coco_eval['segm'].stats[1])
                results_seg['AP_75'].append(coco_evaluator.coco_eval['segm'].stats[2])
                results_seg['AP_small'].append(coco_evaluator.coco_eval['segm'].stats[3])
                results_seg['AP_medium'].append(coco_evaluator.coco_eval['segm'].stats[4])
                results_seg['AP_large'].append(coco_evaluator.coco_eval['segm'].stats[5])
                results_seg['AR_1'].append(coco_evaluator.coco_eval['segm'].stats[6])
                results_seg['AR_10'].append(coco_evaluator.coco_eval['segm'].stats[7])
                results_seg['AR_100'].append(coco_evaluator.coco_eval['segm'].stats[8])
                results_seg['AR_small'].append(coco_evaluator.coco_eval['segm'].stats[9])
                results_seg['AR_medium'].append(coco_evaluator.coco_eval['segm'].stats[10])
                results_seg['AR_large'].append(coco_evaluator.coco_eval['segm'].stats[11])

            else:
                print("No results found for seg evaluation.")
            
            
            print(f"bbox_AP_50: {results_bbox['AP_50']}")
            print(f"seg_AP_50: {results_seg['AP_75']}")
            #print(losses_dict)
        
        # calculate average loss over all images
        #average_results_bbox = {}
        #average_results_seg = {}
        #for key in results_bbox.keys():
            #average_results_bbox[key] = sum(results_bbox[key]) / len(results_bbox[key])
        #for key in results_seg.keys():
            #average_results_seg[key] = sum(results_seg[key]) / len(results_seg[key])
            
        average_results_bbox = {}
        for key in results_bbox.keys():
            average_results_bbox[key] = sum(sum(sublist) for sublist in results_bbox[key]) / len(results_bbox[key])
        average_results_seg = {}
        for key in results_seg.keys():
            average_results_seg[key] = sum(sum(sublist) for sublist in results_seg[key]) / len(results_seg[key])
        
        print(f"average bbox_AP_50 over all images: {average_results_bbox['AP_50']}")
        print(f"average seg_AP_50 over all images: {average_results_seg['AP_50']}")

        
        coco_results = {}
        coco_results['bbox'] = average_results_bbox
        coco_results['seg'] = average_results_seg
        
        return coco_evaluator, coco_results
    
