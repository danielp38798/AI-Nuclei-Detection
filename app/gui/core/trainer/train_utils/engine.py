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


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        checkpoint = {}
        checkpoint["model"] = self.model.state_dict()
        checkpoint["optimizer"]  = self.optimizer.state_dict()
        checkpoint["lr_scheduler"] = self.lr_scheduler.state_dict
        #torch.save(model.state_dict(), self.path)
        torch.save(checkpoint, self.path)
        if self.verbose:
            self.trace_func("EarlyStopping: Model saved.")
        self.val_loss_min = val_loss

def train_one_epoch(model, optimizer, lr_scheduler_list, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    """
    # linear lr scheduler turn off - when ReduceOnPlateau is in use!!
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )
    """

    losses_dict = {'loss_classifier': [],
                    'loss_box_reg': [],
                    'loss_mask': [],
                    'loss_objectness': [],
                    'loss_rpn_box_reg': [],
                    'loss': []}

    hyperparameters = {"lr" : None}
    if "momentum" in optimizer.param_groups[0]:
        hyperparameters["momentum"] = None
    if "weight_decay" in optimizer.param_groups[0]:
        hyperparameters["weight_decay"] = None

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        if targets[0]['boxes'].shape[0] == 0:
            print("no boxes - skipping image")
            continue
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(images, targets)
            #loss_dict = outputs
            #losses = sum(loss for loss in loss_dict.values())

            # updated version: also returns the detections -> modfication necessary for calculating val loss in validate
            loss_dict, detections = outputs
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        #store the loss values
        losses_dict['loss_classifier'].append(loss_dict['loss_classifier'].item())
        losses_dict['loss_box_reg'].append(loss_dict['loss_box_reg'].item())
        losses_dict['loss_mask'].append(loss_dict['loss_mask'].item())
        losses_dict['loss_objectness'].append(loss_dict['loss_objectness'].item())
        losses_dict['loss_rpn_box_reg'].append(loss_dict['loss_rpn_box_reg'].item())
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

            #perform gradient clipping 
            clipping_value = 5 # arbitrary value of your choosing
            torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)

            optimizer.step()

        #if lr_scheduler is not None:
            #lr_scheduler.step()
        
        if lr_scheduler_list is not None:
            for lr_scheduler in lr_scheduler_list:
                if lr_scheduler is not None and lr_scheduler.__class__.__name__ != "ReduceLROnPlateau":
                    lr_scheduler.step()
                #lr_scheduler.step()

        # log the hyperparameters
        hyperparameters['lr'] = optimizer.param_groups[0]["lr"]
        if "momentum" in optimizer.param_groups[0]:
            hyperparameters["momentum"] = optimizer.param_groups[0]["momentum"]
        if "weight_decay" in optimizer.param_groups[0]:
            hyperparameters["weight_decay"] = optimizer.param_groups[0]["weight_decay"]
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced, **hyperparameters)

        #metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    # calculate average loss
    average_losses_dict = {}
    for key in losses_dict.keys():
        if len(losses_dict[key]) > 0:
            average_losses_dict[key] = sum(losses_dict[key]) / len(losses_dict[key])

    #print(f"losses_dict: {losses_dict}")
    #print(f"average_losses_dict: {average_losses_dict}")

    return metric_logger, average_losses_dict, hyperparameters

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
    iou_types = ['bbox', 'segm'] #_get_iou_types(model)
    print(f"iou_types: {iou_types}")
    coco_evaluator = CocoEvaluator(coco, iou_types)

    #val_loss = 0
    losses_dict = {'loss_classifier': [],
                    'loss_box_reg': [],
                    'loss_mask': [],
                    'loss_objectness': [],
                    'loss_rpn_box_reg': [],
                    'loss': []}


    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        if targets[0]['boxes'].shape[0] == 0:
            print("no boxes - skipping image")
            continue

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images, targets)

        # updated version of the model also return the detection and losses to calc the val loss
        loss_dict, detections = outputs
        #print(f"loss_dict.keys(): {loss_dict.keys()}")
        outputs = detections
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

        # calculate validation loss
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        #store the loss values
        if "loss_classifier" in loss_dict.keys():
            losses_dict['loss_classifier'].append(loss_dict['loss_classifier'].item())
        if "loss_box_reg" in loss_dict.keys():
            losses_dict['loss_box_reg'].append(loss_dict['loss_box_reg'].item())
        if "loss_mask" in loss_dict.keys():
            losses_dict['loss_mask'].append(loss_dict['loss_mask'].item())
        if "loss_objectness" in loss_dict.keys():
            losses_dict['loss_objectness'].append(loss_dict['loss_objectness'].item())
        if "loss_rpn_box_reg" in loss_dict.keys():
            losses_dict['loss_rpn_box_reg'].append(loss_dict['loss_rpn_box_reg'].item())
        losses_dict['loss'].append(loss_value)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # calculate average loss
    average_losses_dict = {}
    for key in losses_dict.keys():
        if len(losses_dict[key]) > 0:
            average_losses_dict[key] = sum(losses_dict[key]) / len(losses_dict[key])

    #print(f"losses_dict (eval): {losses_dict}")
    #print(f"average_losses_dict (eval): {average_losses_dict}")

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    
    return coco_evaluator, average_losses_dict



@torch.no_grad()
def validate(model, data_loader, device):
    val_loss = 0
    model.train()
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        outputs = model(images, targets)
        loss_dict, detections = outputs
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        val_loss += losses_reduced
  
    validation_loss = val_loss/ len(data_loader)    
    # convert to float if instance is a tensor
    if isinstance(validation_loss, torch.Tensor):
        validation_loss = validation_loss.item()
    return validation_loss

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
    
