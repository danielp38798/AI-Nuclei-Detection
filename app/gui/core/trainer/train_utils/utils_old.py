import os
import json
import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import Image
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import numpy as np
from pathlib import Path
from collections import defaultdict
from itertools import accumulate
from copy import deepcopy
from random import shuffle
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import imgviz
import cv2
import math
import datetime
import errno
import time
from collections import defaultdict, deque
import torch.distributed as dist



# ---------------------- split data set ------------------------------- #
def read_json(json_path):
    with open(json_path, "r") as f:
        d = json.load(f)
    return d
def get_setname(cocodict, json_path):
    try:
        set_name = cocodict["info"]["description"]
        #print(f"Processing {set_name} (name from json info description)")
    except KeyError:
        json_path_p = Path(json_path)
        set_name = f"{json_path_p.parent.stem}_{json_path_p.stem}"
        #print(f"Processing {set_name} (name derived from json path)")
    return set_name
def read_coco_json(coco_json):
    coco_dict = read_json(coco_json)
    setname = get_setname(coco_dict, coco_json)
    return coco_dict, setname
def write_json(json_path, dic):
    with open(json_path, "w") as f:
        json.dump(dic, f)
    print(f"Wrote json to {json_path}")
def write_json_in_place(orig_coco_json, coco_dict, append_str="new", out_json=None):
    if out_json is None:
        orig_json_path = Path(orig_coco_json)
        out_json_path = (
            orig_json_path.parent / f"{orig_json_path.stem}_{append_str}.json"
        )
    else:
        out_json_path = Path(out_json)
    write_json(out_json_path, coco_dict)
    return str(out_json_path)
def default_coco_dict():
    new_dict = {
        "images": [],
        "annotations": [],
    }
    return new_dict
def split(coco_dict, ratios, names=None, do_shuffle=False, setname=""):
    assert sum(ratios) == 1.0, "Ratios given does not sum up to 1.0"
    if names:
        assert len(ratios) == len(names)
    else:
        names = [f"split{i}" for i in range(len(ratios))]

    total_imgs = len(coco_dict["images"])
    print(f"Total imgs: {total_imgs}")
    splits_num = [int(round(x * total_imgs)) for x in ratios]
    assert sum(splits_num) == total_imgs
    print(f"Splitting into {splits_num}")
    splits_num[0] -= 1
    splits_acc = list(accumulate(splits_num))
    assert splits_acc[-1] == total_imgs - 1

    if do_shuffle:
        shuffle(coco_dict["images"])

    split_coco_dicts = defaultdict(default_coco_dict)
    img_ids_maps = defaultdict(dict)
    oldimgid2name = {}
    split_idx = 0
    this_name = names[split_idx]
    this_split_images = split_coco_dicts[this_name]["images"]
    this_img_ids_map = img_ids_maps[this_name]
    for i, img_dict in enumerate(coco_dict["images"]):
        oldimgid2name[img_dict["id"]] = this_name

        new_img_dict = deepcopy(img_dict)
        new_img_id = len(this_split_images) + 1

        this_img_ids_map[img_dict["id"]] = new_img_id
        new_img_dict["id"] = new_img_id
        this_split_images.append(new_img_dict)

        if i >= splits_acc[split_idx]:
            if split_idx == len(splits_acc) - 1:
                break
            else:
                split_idx += 1
                this_name = names[split_idx]
                this_split_images = split_coco_dicts[this_name]["images"]
                this_img_ids_map = img_ids_maps[this_name]

    for annot_dict in coco_dict["annotations"]:
        name = oldimgid2name[annot_dict["image_id"]]
        new_annot_dict = deepcopy(annot_dict)
        new_annot_dict["id"] = len(split_coco_dicts[name]["annotations"]) + 1
        new_annot_dict["image_id"] = img_ids_maps[name][annot_dict["image_id"]]
        split_coco_dicts[name]["annotations"].append(new_annot_dict)

    for name, dic in split_coco_dicts.items():
        if "info" in coco_dict:
            dic["info"] = deepcopy(coco_dict["info"])
            dic["info"]["description"] = f"{setname}_{name}"
        if "licenses" in coco_dict:
            dic["licenses"] = deepcopy(coco_dict["licenses"])
        if "categories" in coco_dict:
            dic["categories"] = deepcopy(coco_dict["categories"])

    return split_coco_dicts

def split_from_file(cocojson, ratios, names=None, do_shuffle=False):
    print('Performing train / test split ...')
    coco_dict, setname = read_coco_json(cocojson)
    split_coco_dicts = split(
        coco_dict, ratios, names=names, do_shuffle=do_shuffle, setname=setname
    )
    out_json_paths = []
    for name, new_cocodict in split_coco_dicts.items():
        out_json_path = write_json_in_place(cocojson, new_cocodict, append_str=name)
        out_json_paths.append(out_json_path)
    print('Splitting data set done!\n')
    return out_json_paths
# ---------------------- split data set ------------------------------- #





# ----------------------  image utilities ------------------------------- #
def normalize_dataloader(loader, save_path):
    # Calculate mean and standard deviation for each channel across the entire dataset
    mean = 0.0
    std = 0.0
    total_samples = 0
    for batch in loader:
        # Get the images and labels
        images, _ = batch
        for images in images:
            # Get the number of images in the batch
            batch_samples = images.size(0)
            # Flatten the images into a single vector of pixels
            images = images.view(batch_samples, images.size(1), -1)
            # calculate the mean and std of the batch
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)
            total_samples += batch_samples

    mean /= total_samples
    std /= total_samples
    mean_array = mean.numpy()
    std_array = std.numpy()
    mean = np.nanmean(mean.numpy())
    std = np.nanmean(std.numpy())
    print(f"mean over all samples: {mean} \nstd over all samples: {std} \ntotal samples: {total_samples}")
    # Save mean and std to a file
    np.savez(save_path, mean_array=mean_array, mean_of_all_batches=mean, 
             std_array=std_array, std_of_all_batches=std) # mean=mean.numpy(), std=std.numpy())

    return mean, std

def normalize_inference(input_image, normalization_file):
    # Load mean and std from the file
    normalization_data = np.load(normalization_file)
    mean = torch.tensor(normalization_data['mean_of_all_batches'])
    std = torch.tensor(normalization_data['std_of_all_batches'])

    # Normalize the input image
    normalized_image = (input_image - mean.unsqueeze(1).unsqueeze(2)) / std.unsqueeze(1).unsqueeze(2)
    
    return normalized_image

def denormalize_inference(input_image, normalization_file):
    # Load mean and std from the file
    normalization_data = np.load(normalization_file)
    mean = torch.tensor(normalization_data['mean'])
    std = torch.tensor(normalization_data['std'])

    # Denormalize the input image
    denormalized_image = (input_image * std.unsqueeze(1).unsqueeze(2)) + mean.unsqueeze(1).unsqueeze(2)
    
    return denormalized_image

def get_normalization_values(filename):
    print('\nLoading mean and std of images from file ...')
    # load the mean and std from the file
    data = np.load(filename)
    mean = data['mean_of_all_batches']
    std = data['std_of_all_batches']
    print('\nLoading mean and std of images from file done!')
    print(f"mean: {data['mean_of_all_batches']}")
    print(f"std: {data['std_of_all_batches']}\n")
    return mean, std 



# ------------------------ devision into patches -------------------------------- #  
class ImageTiler:
    def __init__(self,patchsize=512, overlap=0.5):
        self.patchsize = patchsize
        self.overlap = overlap
            
    def get_target(self):
        dataDir = 'data'
        annFile = 'data/annotations/instances_default.json'
        coco = COCO(annFile)
        catIds = coco.getCatIds()
        imgIds = coco.getImgIds()
        np.random.seed(50)
        img_id = imgIds[np.random.randint(0, len(imgIds))]
        # load image
        img = coco.loadImgs(img_id)[0]
        image_path = '%s/images/%s' % (dataDir, img['file_name'])
        # load image
        I = Image.open(image_path)
        if I.mode != 'RGB':
            I = I.convert('RGB')
        annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
        coco_annotation = coco.loadAnns(annIds)
        num_objs = len(coco_annotation)
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2] # xmin + width
            ymax = ymin + coco_annotation[i]['bbox'][3] # ymin + height
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = []
        for i in range(num_objs):
            original_id = coco_annotation[i]['category_id']
            shifted_id = original_id #- 1  # subtract 1 to shift the class IDs (for 0-based indexing)
            labels.append(shifted_id)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # Tensorise img_id
        img_id = int(img_id) #torch.tensor([img_id], dtype=torch.int64) #must be int for the coco evaluator!
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        # create masks of the segmentation
        masks = []
        for i in range(num_objs):
            masks.append(coco.annToMask(coco_annotation[i]))
        masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        # Annotation is in dictionary format
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = img_id
        target["area"] = areas
        target["iscrowd"] = iscrowd
        target["masks"] = masks
        return I, target, coco, coco_annotation

    def is_bbox_in_patch(self, bounding_box, patch_bounds):
        upper_left, lower_right = patch_bounds
        x1, y1 = upper_left
        x2, y2 = lower_right
        xmin, ymin, xmax, ymax = bounding_box
        # check if the bounding box is in the patch
        if (xmin >= x1) and (xmax <= x2) and (ymin >= y1) and (ymax <= y2):
            return True
        else:
            return False
    
    def is_mask_in_patch(self, mask, patch_bounds):
        upper_left, lower_right = patch_bounds
        x1, y1 = upper_left
        x2, y2 = lower_right
        # check if one pixel of the mask is in the patch
        if np.any(mask[y1:y2, x1:x2]):
            return True
        else:
            return False
    

    def filter_bboxes(self, bboxes, labels, masks, patch_bounds):
        filtered_bboxes = []
        filtered_labels = []
        for bbox,label in zip(bboxes, labels):
            if self.is_bbox_in_patch(bounding_box=bbox, patch_bounds=patch_bounds):
                adjusted_bbox = self.adjust_bbox_to_patch(bounding_box=bbox, patch_bounds=patch_bounds)
                filtered_bboxes.append(adjusted_bbox)
                filtered_labels.append(label)
            else:
                continue

        return filtered_bboxes, filtered_labels
    
    def filter_masks(self, masks, patch_bounds):
        filtered_masks = []
        for mask in masks:
            if self.is_mask_in_patch(mask=mask, patch_bounds=patch_bounds):
                adjusted_mask = self.adjust_mask_to_patch(mask=mask, patch_bounds=patch_bounds)
                filtered_masks.append(adjusted_mask)
            else:
                continue

        return filtered_masks
    
    def filter_bboxes_and_masks(self, bboxes, labels, masks, patch_bounds):
        filtered_bboxes = []
        filtered_labels = []
        filtered_masks = []
        for bbox,label,mask in zip(bboxes, labels, masks):
            if self.is_bbox_in_patch(bounding_box=bbox, patch_bounds=patch_bounds):
                adjusted_bbox = self.adjust_bbox_to_patch(bounding_box=bbox, patch_bounds=patch_bounds)
                filtered_bboxes.append(adjusted_bbox)
                filtered_labels.append(label)
                adjusted_mask = self.adjust_mask_to_patch(mask=mask, patch_bounds=patch_bounds)
                filtered_masks.append(adjusted_mask)
            else:
                continue

        return filtered_bboxes, filtered_labels, filtered_masks

    def adjust_bbox_to_patch(self, bounding_box, patch_bounds):
        upper_left, lower_right = patch_bounds
        x1, y1 = upper_left
        x2, y2 = lower_right
        #xmin, ymin, w, h = bounding_box # pytorch format: xmin, ymin, ymax, ymax
        xmin, ymin, xmax, ymax = bounding_box # coco format: xmin, ymin, width, height
        # adjust the bounding box to the patch
        xmin = xmin - x1
        xmax = xmax - x1
        ymin = ymin - y1
        ymax = ymax - y1
        return [xmin, ymin, xmax, ymax]
    
    def adjust_bbox_to_image(self, bounding_box, patch_bounds):
        upper_left, lower_right = patch_bounds
        x1, y1 = upper_left
        x2, y2 = lower_right
        #xmin, ymin, w, h = bounding_box # pytorch format: xmin, ymin, ymax, ymax
        xmin, ymin, xmax, ymax = bounding_box
        # adjust the bounding box to the patch
        xmin = xmin + x1
        xmax = xmax + x1
        ymin = ymin + y1
        ymax = ymax + y1
        return [xmin, ymin, xmax, ymax]
    
    def adjust_mask_to_patch(self, mask, patch_bounds):
        upper_left, lower_right = patch_bounds
        x1, y1 = upper_left
        x2, y2 = lower_right
        # show the mask
        #plt.imshow(mask)
        #plt.title("mask before crop")
        #plt.show()
        
        #print(f"mask.shape before crop: {mask.shape}")
        # adjust the mask to the patch
        mask = mask[y1:y2, x1:x2]
        #print(f"mask.shape after crop: {mask.shape}")
        
        # show the mask
        #plt.imshow(mask)
        #plt.title("mask after crop")
        #plt.show()
        
        return mask 


    def display_patches_with_annotations(self, target_dict):
        # write bounding boxes into the image using cv2
        patched_images = []
        for patch_bounds, data in target_dict.items():
            patch, targets = data   
            #print(f"patch.shape: {patch.shape}")
            #print(f"type(patch): {type(patch)}")
            bboxes = targets['boxes']
            masks = targets['masks']
            labels = targets['labels']
            for idx, bbox in enumerate(bboxes):
                xmin, ymin, xmax, ymax = bbox
                cv2.rectangle(patch, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)
                cv2.putText(patch, str(labels[idx]), (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            for mask in masks:
                # overlay mask on patch
                patch[mask == 1] = (0, 255, 0)
                
            patched_images.append(patch)
        return patched_images
    
    def extract_patches(self, data, patchsize, overlap=None, stride=None, vox=False):
        '''
        Parameters
        ----------
        data : array to extract patches from; it can be 1D, 2D or 3D [W, H, D]. H: Height, W: Width, D: Depth,
               3D data includes images (RGB, RGBA, etc) or Voxel data.
        patchsize :  size of patch to extract from image only square patches can be
                    extracted for now.
        overlap (Optional): overlap between patched in percentage a float between [0, 1].
        stride (Optional): Step size between patches
        vox (Optional): Whether data is volumetric or not if set to true array will be cropped in last dimension too.

        Returns
        -------
        data_patches : a list containing extracted patches of images.
        indices : a list containing indices of patches in order, whihc can be used 
                at later stage for 'merging_patches'.

        '''
        dims = data.shape
        if len(dims)==1:        
            width = data.shape[0]
            maxWindowSize = patchsize
            windowSizeX = maxWindowSize
            windowSizeX = min(windowSizeX, width)
        elif len(dims)==2: 
            height = data.shape[0]
            width = data.shape[1]
            maxWindowSize = patchsize
            windowSizeX = maxWindowSize
            windowSizeY = maxWindowSize
            windowSizeX = min(windowSizeX, width)
            windowSizeY = min(windowSizeY, height)
        elif len(dims)==3:
            height = data.shape[0]
            width = data.shape[1]
            depth = data.shape[2]
            maxWindowSize = patchsize
            windowSizeX = maxWindowSize
            windowSizeY = maxWindowSize
            windowSizeZ = maxWindowSize
            windowSizeX = min(windowSizeX, width)
            windowSizeY = min(windowSizeY, height)
            windowSizeZ = min(windowSizeZ, depth)
        if stride is not None:
            if len(dims)==1:
                stepSizeX = stride
            elif len(dims)==2:
                stepSizeX = stride
                stepSizeY = stride
            elif len(dims)==3:
                stepSizeX = stride
                stepSizeY = stride
                stepSizeZ = stride               
        elif overlap is not None:
            overlapPercent = overlap
            if len(dims)==1:
                windowSizeX = maxWindowSize     
                # If the input data is smaller than the specified window size,
                # clip the window size to the input size on both dimensions
                windowSizeX = min(windowSizeX, width)
                # Compute the window overlap and step size
                windowOverlapX = int(math.floor(windowSizeX * overlapPercent))
                stepSizeX = windowSizeX - windowOverlapX   
            elif len(dims)==2:
                windowSizeX = maxWindowSize
                windowSizeY = maxWindowSize
                # If the input data is smaller than the specified window size,
                # clip the window size to the input size on both dimensions
                windowSizeX = min(windowSizeX, width)
                windowSizeY = min(windowSizeY, height)
                # Compute the window overlap and step size
                windowOverlapX = int(math.floor(windowSizeX * overlapPercent))
                windowOverlapY = int(math.floor(windowSizeY * overlapPercent))
                stepSizeX = windowSizeX - windowOverlapX
                stepSizeY = windowSizeY - windowOverlapY   
            elif len(dims)==3:
                windowSizeX = maxWindowSize
                windowSizeY = maxWindowSize
                windowSizeZ = maxWindowSize
                # If the input data is smaller than the specified window size,
                # clip the window size to the input size on both dimensions
                windowSizeX = min(windowSizeX, width)
                windowSizeY = min(windowSizeY, height)
                windowSizeZ = min(windowSizeZ, depth)
                # Compute the window overlap and step size
                windowOverlapX = int(math.floor(windowSizeX * overlapPercent))
                windowOverlapY = int(math.floor(windowSizeY * overlapPercent))
                windowOverlapZ = int(math.floor(windowSizeZ * overlapPercent))
                stepSizeX = windowSizeX - windowOverlapX
                stepSizeY = windowSizeY - windowOverlapY                
                stepSizeZ = windowSizeZ - windowOverlapZ                
        else:
            if len(dims)==1:
                stepSizeX = 1
            elif len(dims)==2:
                stepSizeX = 1
                stepSizeY = 1
            elif len(dims)==3:
                stepSizeX = 1
                stepSizeY = 1
                stepSizeZ = 1    
        if len(dims)==1:
            # Determine how many windows we will need in order to cover the input data
            lastX = width - windowSizeX
            xOffsets = list(range(0, lastX+1, stepSizeX))
            # Unless the input data dimensions are exact multiples of the step size,
            # we will need one additional row and column of windows to get 100% coverage
            if len(xOffsets) == 0 or xOffsets[-1] != lastX:
                xOffsets.append(lastX)

            indices = []
            patch_dictionary = {}
            
            #for xOffset in xOffsets:
            for x, xOffset in enumerate(xOffsets):
                if len(data.shape) >= 3:
                    patch = data[(slice(xOffset, xOffset+windowSizeX, None))]
                    patch_dictionary[(x,0)][0] = patch
                    patch_dictionary[(y,0)][1] = (xOffset, xOffset+windowSizeX)
                    
                else:
                    patch = data[(slice(xOffset, xOffset+windowSizeX))]
                    patch_dictionary[(xOffset, xOffset+windowSizeX)] = patch
                    
                indices.append((xOffset, xOffset+windowSizeX))
                
        elif len(dims)==2:
            # Determine how many windows we will need in order to cover the input data
            lastX = width - windowSizeX
            lastY = height - windowSizeY
            xOffsets = list(range(0, lastX+1, stepSizeX))
            yOffsets = list(range(0, lastY+1, stepSizeY))
            # Unless the input data dimensions are exact multiples of the step size,
            # we will need one additional row and column of windows to get 100% coverage
            if len(xOffsets) == 0 or xOffsets[-1] != lastX:
                xOffsets.append(lastX)
            if len(yOffsets) == 0 or yOffsets[-1] != lastY:
                yOffsets.append(lastY)
            indices = []
            patch_dictionary = {}
            
           #for xOffset in xOffsets:
            for y, yOffset in enumerate(yOffsets):
                #for yOffset in yOffsets:
                for x, xOffset in enumerate(xOffsets):  
                    if len(data.shape) >= 3:
                        patch = data[(slice(yOffset, yOffset+windowSizeY, None),
                                                slice(xOffset, xOffset+windowSizeX, None))]
                        #patch_dictionary[(yOffset, yOffset+windowSizeY, xOffset, xOffset+windowSizeX)] = patch
                        patch_dictionary[(x, y)][0] = patch
                        patch_dictionary[(x, y)][1] = (yOffset, yOffset+windowSizeY, xOffset, xOffset+windowSizeX)
                    else:
                        patch = data[(slice(yOffset, yOffset+windowSizeY),
                                                slice(xOffset, xOffset+windowSizeX))]
                        patch_dictionary[(yOffset, yOffset+windowSizeY, xOffset, xOffset+windowSizeX)] = patch
                    indices.append((yOffset, yOffset+windowSizeY, xOffset, xOffset+windowSizeX))            
        elif len(dims)==3:
            lastX = width - windowSizeX
            lastY = height - windowSizeY
            lastZ = depth - windowSizeZ
            xOffsets = list(range(0, lastX+1, stepSizeX))
            yOffsets = list(range(0, lastY+1, stepSizeY))
            zOffsets = list(range(0, lastZ+1, stepSizeZ))
            if len(xOffsets) == 0 or xOffsets[-1] != lastX:
                xOffsets.append(lastX)
            if len(yOffsets) == 0 or yOffsets[-1] != lastY:
                yOffsets.append(lastY)
            if len(zOffsets) == 0 or zOffsets[-1] != lastZ:
                zOffsets.append(lastZ)
            
            indices = []
            patch_dictionary = {(0,0):[None, None, None]}
            for x, xOffset in enumerate(xOffsets):
                for y, yOffset in enumerate(yOffsets):
                    patch_dictionary[(x, y)] = [None, None, None]  
            if not vox: # for images 
                for y, yOffset in enumerate(yOffsets):
                    for x, xOffset in enumerate(xOffsets):  
                        if len(data.shape) >= 3:
                            patch = data[(slice(yOffset, yOffset+windowSizeY, None),
                                                    slice(xOffset, xOffset+windowSizeX, None))]
                            patch_dictionary[(x, y)][0] = patch
                            patch_dictionary[(x, y)][1] = (yOffset, yOffset+windowSizeY, xOffset, xOffset+windowSizeX)
                            patch_dictionary[(x, y)][2] = (yOffset, xOffset)
                        else:
                            patch = data[(slice(yOffset, yOffset+windowSizeY),                                                  
                                                    slice(xOffset, xOffset+windowSizeX))]   
                            patch_dictionary[(x, y)][0] = patch
                            patch_dictionary[(x, y)][1] = (yOffset, yOffset+windowSizeY, xOffset, xOffset+windowSizeX)
                            patch_dictionary[(x, y)][2] = (yOffset, xOffset)
    
                        indices.append((yOffset, yOffset+windowSizeY, xOffset, xOffset+windowSizeX))
            if vox: # for volumetric data
                for z, zOffset in enumerate(zOffsets):
                    #for yOffset in yOffsets:
                    for y, yOffset in enumerate(yOffsets):  
                        #for zOffset in zOffsets:
                        for x, xOffset in enumerate(xOffsets):
                            if len(data.shape) >= 4:
                                patch = data[(slice(yOffset, yOffset+windowSizeY, None),
                                                        slice(xOffset, xOffset+windowSizeX, None),
                                                        slice(zOffset, zOffset+windowSizeZ, None))]
                                patch_dictionary[(x, y, z)][0] = patch
                                patch_dictionary[(x, y, z)][1] = (yOffset, yOffset+windowSizeY, xOffset, xOffset+windowSizeX, zOffset, zOffset+windowSizeZ)

                            else:
                                patch = data[(slice(yOffset, yOffset+windowSizeY),
                                                        slice(xOffset, xOffset+windowSizeX),
                                                        slice(zOffset, zOffset+windowSizeZ))]
                                patch_dictionary[(x, y, z)][0] = patch
                                patch_dictionary[(x, y, z)][1] = (yOffset, yOffset+windowSizeY, xOffset, xOffset+windowSizeX, zOffset, zOffset+windowSizeZ)
                                patch_dictionary[(x, y)][2] = (zOffset, yOffset, xOffset)
                                
                            indices.append((yOffset, yOffset+windowSizeY, xOffset, xOffset+windowSizeX, zOffset, zOffset+windowSizeZ))   
        
        return indices, patch_dictionary

    def merge_patches(self, data_patches, indices, mode='overwrite'):
        '''
        Parameters
        ----------
        data_patches : list containing image patches that needs to be joined, dtype=uint8
        indices : a list of indices generated by 'extract_patches' function of the format;
                    (yOffset, yOffset+windowSizeY, xOffset, xOffset+windowSizeX)
        mode : how to deal with overlapping patches;
                overwrite -> next patch will overwrite the overlapping area of the previous patch.
                max -> maximum value of overlapping area at each pixel will be written.
                min -> minimum value of overlapping area at each pixel will be written.
                avg -> mean/average value of overlapping area at each pixel will be written.
        Returns
        -------
        Stitched image.
        '''
        modes = ["overwrite", "max", "min", "avg"]
        if mode not in modes:
            raise ValueError(f"mode has to be either one of {modes}, but got {mode}")

        dims = len(indices[-1])
        
        if dims==2:
            orig_h = indices[-1][1]
        elif dims==4:
            orig_h = indices[-1][1]
            orig_w = indices[-1][3]
        elif dims==6:
            orig_h = indices[-1][1]
            orig_w = indices[-1][3]
            orig_d = indices[-1][5]
        
        ### There is scope here for rgb/hyperspectral volume (i.e. 4D -> 3 spatial and 1 spectral dimensions, simplest case is only 3 channles for the spectral dimension)
        rgb = True
        if len(data_patches[0].shape) == 2:
            rgb = False
        
        if mode == 'min':
            if dims == 2:
                empty_data = np.zeros((orig_h)).astype(np.float32) + np.inf # using float here is better
                
            elif dims==4:
                if rgb:
                    empty_data = np.zeros((orig_h, orig_w, data_patches[0].shape[-1])).astype(np.float32) + np.inf # using float here is better
                else:
                    empty_data = np.zeros((orig_h, orig_w)).astype(np.float32) + np.inf # using float here is better

            elif dims==6:
                empty_data = np.zeros((orig_h, orig_w, orig_d)).astype(np.float32) + np.inf # using float here is better
                
        else:
            if dims == 2:
                empty_data = np.zeros((orig_h)).astype(np.float32) # using float here is better
                
            elif dims==4:
                if rgb:
                    empty_data = np.zeros((orig_h, orig_w, data_patches[0].shape[-1])).astype(np.float32) # using float here is better
                else:
                    empty_data = np.zeros((orig_h, orig_w)).astype(np.float32) # using float here is better

            elif dims==6:
                empty_data = np.zeros((orig_h, orig_w, orig_d)).astype(np.float32) # using float here is better

        for i, indice in enumerate(indices):

            if mode == 'overwrite':

                if dims == 2:
                    empty_data[indice[0]:indice[1]] = data_patches[i]

                elif dims == 4:
                    if rgb:
                        empty_data[indice[0]:indice[1], indice[2]:indice[3], :] = data_patches[i]
                    else:
                        empty_data[indice[0]:indice[1], indice[2]:indice[3]] = data_patches[i]
                        
                elif dims==6:
                        empty_data[indice[0]:indice[1], indice[2]:indice[3], indice[4]:indice[5]] = data_patches[i]
                        
                        
            elif mode == 'max':

                if dims == 2:
                    empty_data[indice[0]:indice[1]] = np.maximum(data_patches[i], empty_data[indice[0]:indice[1]])
                elif dims == 4:
                    if rgb:
                        empty_data[indice[0]:indice[1], indice[2]:indice[3], :] = np.maximum(data_patches[i], empty_data[indice[0]:indice[1], indice[2]:indice[3], :])
                    else:
                        empty_data[indice[0]:indice[1], indice[2]:indice[3]] = np.maximum(data_patches[i], empty_data[indice[0]:indice[1], indice[2]:indice[3]])
                elif dims==6:
                        empty_data[indice[0]:indice[1], indice[2]:indice[3], indice[4]:indice[5]] = np.maximum(data_patches[i], empty_data[indice[0]:indice[1], indice[2]:indice[3], indice[4]:indice[5]])


            elif mode == 'min':
                if dims == 2:
                    empty_data[indice[0]:indice[1]] = np.minimum(data_patches[i], empty_data[indice[0]:indice[1]])
                elif dims == 4:
                    if rgb:
                        empty_data[indice[0]:indice[1], indice[2]:indice[3], :] = np.minimum(data_patches[i], empty_data[indice[0]:indice[1], indice[2]:indice[3], :])
                    else:
                        empty_data[indice[0]:indice[1], indice[2]:indice[3]] = np.minimum(data_patches[i], empty_data[indice[0]:indice[1], indice[2]:indice[3]])
                elif dims==6:
                        empty_data[indice[0]:indice[1], indice[2]:indice[3], indice[4]:indice[5]] = np.minimum(data_patches[i], empty_data[indice[0]:indice[1], indice[2]:indice[3], indice[4]:indice[5]])
                    
            elif mode == 'avg':

                if dims == 2:
                    empty_data[indice[0]:indice[1]] = np.where(empty_data[indice[0]:indice[1]] == 0,
                                                                                    data_patches[i], 
                                                                                    np.add(data_patches[i],empty_data[indice[0]:indice[1]])/2)
                elif dims == 4:
                    if rgb:
                        empty_data[indice[0]:indice[1], indice[2]:indice[3], :] = np.where(empty_data[indice[0]:indice[1], indice[2]:indice[3], :] == 0,
                                                                                            data_patches[i], 
                                                                                            np.add(data_patches[i],empty_data[indice[0]:indice[1], indice[2]:indice[3], :])/2)
                    else:
                        empty_data[indice[0]:indice[1], indice[2]:indice[3]] = np.where(empty_data[indice[0]:indice[1], indice[2]:indice[3]] == 0,
                                                                                        data_patches[i], 
                                                                                        np.add(data_patches[i],empty_data[indice[0]:indice[1], indice[2]:indice[3]])/2)
                elif dims==6:
                        empty_data[indice[0]:indice[1], indice[2]:indice[3], indice[4]:indice[5]] = np.where(empty_data[indice[0]:indice[1], indice[2]:indice[3], indice[4]:indice[5]] == 0,
                                                                                        data_patches[i], 
                                                                                        np.add(data_patches[i],empty_data[indice[0]:indice[1], indice[2]:indice[3], indice[4]:indice[5]])/2)

        return empty_data

    def split_image_and_annotations(self, image, target, patchsize, show_patches=False):
        """

        :param em: EMPatches object
        :param image: numpy array
        :param target: dictionary
        :param patchsize: int
        :return: patches, patch_indices, patch_target_dict
        
        """
        from PIL import ImageDraw
        indices, patch_dict = self.extract_patches(image, patchsize, overlap=self.overlap)
        patch_indices = indices
        max_columns = image.shape[0] // patchsize
        previous_position = (-1, -1)
        sorted_patch_dict = sorted(patch_dict.keys(), key=lambda x: (x[1]))
        patch_target_dict = {}
        for position in sorted_patch_dict:
            x, y = position 
            prev_x, prev_y = previous_position
            indices = patch_dict[position][2] # indices of the patch
            previous_position = position
            all_offsets = patch_dict[position][1]
            y1, y2, x1, x2 = all_offsets
            patch = patch_dict[position][0]
            if patch.dtype == np.uint8:
                   patch = patch.astype(np.float32) / 255.0
            patch_target_dict[((x1, y1), (x2, y2))] = patch  # store the patch in the patch_target_dict with the patch_bounds as key
            
        if target is not None:
            
            #print("performing image and target processing")
            bboxes = target['boxes'].numpy()
            labels = target['labels'].numpy()
            masks = target['masks'].numpy()
            image_id = target["image_idc:\Users\d_poi\Documents\Python Projects\23_12_23\train_utils\utils.py"]
            #print(f"masks.shape: {masks.shape}")
            
            new_patch_target_dict = {patch_bounds: [patch, {'boxes': [], 'labels': [], 'masks':[]}] for patch_bounds, patch in patch_target_dict.items()}
            for patch_bounds, patch in patch_target_dict.items():
                #filtered_bboxes, filtered_labels = self.filter_bboxes(bboxes=bboxes, labels=labels, masks=masks, patch_bounds=patch_bounds)
                #filtered_masks = self.filter_masks(masks=masks, patch_bounds=patch_bounds)
                filtered_bboxes, filtered_labels, filtered_masks = self.filter_bboxes_and_masks(bboxes=bboxes, labels=labels, masks=masks, patch_bounds=patch_bounds)
                if len(filtered_bboxes) == 0 or len(filtered_labels) == 0 or len(filtered_masks) == 0:
                    #print(f"no bboxes, labels or masks in patch: {patch_bounds}")
                    continue
                else:
                    #print(f"found {len(filtered_bboxes)} bboxes, {len(filtered_labels)} labels and {len(filtered_masks)} masks in patch: {patch_bounds}")
                
                    if patch.dtype == np.uint8:
                        patch = patch.astype(np.float32) / 255.0 # store the patch in the patch_target_dict with the patch_bounds as key as float32 with values in the range [0, 1]
                        
                    if show_patches:
                        # transform the patch to PIL image
                        patch_plot = Image.fromarray((patch * 255).astype(np.uint8))
                        # create a draw object
                        draw = ImageDraw.Draw(patch_plot)
                        # plot the bboxes
                        for bbox in filtered_bboxes:
                            xmin, ymin, xmax, ymax = bbox
                            draw.rectangle([xmin, ymin, xmax, ymax], outline='red', width=2)
                        # combine all masks into a single image
                        combined_mask = np.max(np.array(filtered_masks), axis=0)
                        # create a mask image and draw the combined mask on it
                        mask_image = Image.fromarray((combined_mask * 255).astype(np.uint8))
                        mask_draw = ImageDraw.Draw(mask_image)
                        mask_draw.bitmap((0, 0), mask_image, fill='white')
                        # composite the original image and the mask image
                        result = Image.blend(patch_plot, mask_image.convert('RGB'), alpha=0.2)
                        # show the result
                        #result.show()
                        #result.save("patches.png")
                    
                    new_patch_target_dict[patch_bounds] = [patch, {'boxes': filtered_bboxes, 'labels': filtered_labels, 'masks': filtered_masks}]
        else: # only extract patches from the image if in inference mode
            #print("performing image processing (inferencing)")
            new_patch_target_dict = {}
            for patch_bounds, patch in patch_target_dict.items():
                if patch.dtype == np.uint8:
                   patch = patch.astype(np.float32) / 255.0
                new_patch_target_dict[patch_bounds] = [patch, None] #return None as target if in inference mode
            
        return patch_indices, new_patch_target_dict

    def prepare_patch_anno_dict_for_torch(self, patch_anno_dict):
        # convert the patch_anno_dict to a format that can be used during training
        # patch_anno_dict: {(x1, y1), (x2, y2)}: [patch, {'boxes': filtered_bboxes, 'labels': filtered_labels, 'masks': filtered_masks}]
        # filtered bboxes: [xmin, ymin, xmax, ymax] are in numpy format and need to be converted to torch tensor
        # filtered labels: [label1, label2, ...] are in numpy format and need to be converted to torch tensor
        # filtered masks: [mask1, mask2, ...] are in numpy format and need to be converted to torch tensor
        for patch_bounds, data in patch_anno_dict.items():
            patch, targets = data
            
            if targets is None:
                #print(f"no annotations in patch: {patch_bounds} (inference mode)")
                #print(f"patch.shape: {patch.shape}")
                #  convert patch to torch tensor
                # bring patch into shape (C, H, W)
                patch = np.transpose(patch, (2, 0, 1))
                patch = torch.as_tensor(patch, dtype=torch.float32)
                #print(f"patch.shape after transpose: {patch.shape}")
                #print(f"patch.dtype: {patch.dtype}")
                #print(f"patch min: {patch.min()}")  
                #print(f"patch max: {patch.max()}")
                patch_anno_dict[patch_bounds] = [patch, None]
                continue
            else:
                #print(f"annotations in patch: {patch_bounds} (training mode)")
                bboxes = targets['boxes']
                labels = targets['labels']
                masks = targets['masks']
                #  convert patch to torch tensor
                #print(f"patch.shape: {patch.shape}")
                # bring patch into shape (C, H, W)
                patch = np.transpose(patch, (2, 0, 1))
                patch = torch.as_tensor(patch, dtype=torch.float32)
                #print(f"patch.shape after transpose: {patch.shape}")
                #print(f"patch.dtype: {patch.dtype}")
                #print(f"patch min: {patch.min()}")  
                #print(f"patch max: {patch.max()}")
                # convert bboxes to torch tensor
                #bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
                # faster: convert to np array first and then to torch tensor
                bboxes = np.array(bboxes, dtype=np.float32)
                bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
                # convert labels to torch tensor
                labels = torch.as_tensor(labels, dtype=torch.int64)
                # convert masks to torch tensor
                masks = np.array(masks, dtype=np.uint8)
                masks = torch.as_tensor(masks, dtype=torch.uint8)
                # update the patch_anno_dict
                patch_anno_dict[patch_bounds] = [patch, {'boxes': bboxes, 'labels': labels, 'masks': masks}]
            
        return patch_anno_dict
        
        

    def process_image(self, original_img, target, show_patches=False):
        
        # convert PIL image to numpy array
        original_img = np.array(original_img)
        #print(f"original_img.dtype: {original_img.dtype}")
        # Step 1: Split image and annotations into patches
        indices, patch_anno_dict = self.split_image_and_annotations(original_img, target=target, patchsize=self.patchsize, show_patches=False)
        
        if show_patches:
            store_path = os.path.join('tiles')
            if not os.path.exists(store_path):
                os.makedirs(store_path)
            # Step 2: Display the patches with the annotations
            patched_images = self.display_patches_with_annotations(patch_anno_dict)
            # Step 3: Merge the patches
            merged_img = self.merge_patches(patched_images, indices, mode='avg')
            plt.figure()
            plt.imshow(merged_img.astype(np.uint8))
            plt.title('Merge patches')
            plt.close()
            #plt.show()
            # store the merged image as an image file
            merged_img = Image.fromarray(merged_img.astype(np.uint8))
            current_time = time.strftime("%H_%M_%S")
            path = os.path.join(store_path, f'merged_img_{self.patchsize}_{current_time}.png')
            merged_img.save(path)
            plt.close()
            
            tiled= imgviz.tile(list(map(np.uint8, patched_images)),border=(255,0,0))
            plt.figure()
            plt.imshow(tiled)
            #plt.show()
            tiled = Image.fromarray(tiled.astype(np.uint8))
            current_time = time.strftime("%H_%M_%S")
            path = os.path.join(store_path, f'tiled_img_{self.patchsize}_{current_time}.png')
            tiled.save(path)
            plt.close()
        
        patch_anno_dict = self.prepare_patch_anno_dict_for_torch(patch_anno_dict)
                    
        return indices, patch_anno_dict


# ------------------------ old custom patch transform classes -------------------------------- #

class PatchTransform_v1:
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, images, targets):
        patched_images = []
        patched_targets = []

        for img, target in zip(images, targets):
            
            # if the image is not a PIL image convert it to PIL
            if not isinstance(img, Image.Image):
                img = F.to_pil_image(img)
            
            width, height = img.size

            # Determine number of patches based on patch_size
            num_patches_x = width // self.patch_size
            num_patches_y = height // self.patch_size

            for i in range(num_patches_x):
                for j in range(num_patches_y):
                    # Calculate patch coordinates
                    x_start = i * self.patch_size
                    y_start = j * self.patch_size
                    x_end = min((i + 1) * self.patch_size, width)
                    y_end = min((j + 1) * self.patch_size, height)

                    # Crop the patch from the image
                    patch_img = F.crop(img, y_start, x_start, y_end - y_start, x_end - x_start)

                    # Filter targets within the patch
                    filtered_targets = self.filter_targets(target, x_start, y_start, x_end, y_end)
                    if filtered_targets is None:
                        continue
                    patched_images.append(patch_img)
                    patched_targets.append(filtered_targets)
        
        # Convert PIL to a tensor
        patched_images = torch.stack([F.to_tensor(img) for img in patched_images])
        #print(f"patched_images.shape: {patched_images.shape}")
        
        return patched_images, patched_targets

    def filter_targets(self, target, x_start, y_start, x_end, y_end):
            masks = target['masks']
            labels = target['labels']
            boxes = target['boxes']

            filtered_masks = []
            filtered_labels = []
            filtered_boxes = []

            for i, box in enumerate(boxes):
                x_min, y_min, x_max, y_max = box.tolist()
                # Check if box coordinates intersect with patch coordinates
                if x_max > x_start and x_min < x_end and y_max > y_start and y_min < y_end:
                    # Adjust box coordinates to patch space
                    box[0] = max(x_min - x_start, 0)
                    box[1] = max(y_min - y_start, 0)
                    box[2] = min(x_max - x_start, x_end - x_start)
                    box[3] = min(y_max - y_start, y_end - y_start)

                    # Get mask coordinates within the patch
                    #mask_patch = masks[i][int(y_start):int(y_end), int(x_start):int(x_end)]
                    #mask_patch = mask_patch[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                    
                    # Get mask coordinates within the patch
                    #print(f"masks[i] {masks[i]}")
                    #print(f"masks[i].shape: {masks[i].shape}")
                    mask_patch = masks[i][int(y_start):int(y_end), int(x_start):int(x_end)]
                    #print(f"mask_patch.shape: {mask_patch.shape}")
                    #mask_patch = mask_patch[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                    #print(f"mask_patch.shape: {mask_patch.shape}")
                    # if mask_patch is empty, skip it
                    if mask_patch.shape[0] == 0 or mask_patch.shape[1] == 0:
                        continue
                    filtered_masks.append(mask_patch)
                    filtered_labels.append(labels[i])
                    filtered_boxes.append(box)
                    
            # create tensors from the filtered targets; only consider non-empty masks and corresponding labels and boxes
            if len(filtered_masks) == 0:
                #filtered_masks = np.zeros((1,1,1))
                #filtered_labels = np.zeros((1))
                #filtered_boxes = np.zeros((1,4))
                return None
            if len(filtered_boxes) == 0:
                return None
                #filtered_boxes = np.zeros((1,4))
            # stack masks
            filtered_masks = np.stack(filtered_masks)
            #print(f"filtered_masks.shape: {filtered_masks.shape}")
            filtered_masks = torch.as_tensor(filtered_masks, dtype=torch.uint8)
            filtered_labels = torch.tensor(filtered_labels)
            # stack boxes
            filtered_boxes = np.stack(filtered_boxes)
            filtered_boxes = torch.as_tensor(filtered_boxes, dtype=torch.float32)
  

            filtered_targets = {
                'masks': filtered_masks,
                'labels': filtered_labels,
                'boxes': filtered_boxes
            }
            
            #print(f"filtered_targets: {filtered_targets}")

            return filtered_targets
              




class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        if self.count == 0:
            return None
        return self.total / self.count

    @property
    def max(self):
        # avoid is an empty sequence error
        if len(self.deque) == 0:
            return 0
        else:
            return max(self.deque)
        
    @property
    def value(self):
        # avoid index out of range error
        if len(self.deque) == 0:
            return 0
        return self.deque[-1]
    
    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.inference_mode():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)")


def collate_fn(batch):
    return tuple(zip(*batch))


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


    def __init__(self, patch_size):
        self.patch_size = patch_size
        self.index = 0

    def __call__(self, image, target):
        # Teilen Sie das Bild und das Ziel in Patches auf
        patches, targets = self.split_into_patches(image, target)

        # Geben Sie den nächsten Patch und das entsprechende Ziel zurück
        patch = patches[self.index]
        target = targets[self.index]

        # Aktualisieren Sie den Index für den nächsten Aufruf
        self.index = (self.index + 1) % len(patches)

        return patch, target

    def split_into_patches(self, img, target):

        patched_image = []
        patched_targets = []
        
        print(f"img: {img}")
        print(f"target: {target}")
        # if the image is not a PIL image convert it to PIL
        if not isinstance(img, Image.Image):
            img = F.to_pil_image(img)
        
        width, height = img.size

        # Determine number of patches based on patch_size
        num_patches_x = width // self.patch_size
        num_patches_y = height // self.patch_size

        for i in range(num_patches_x):
            for j in range(num_patches_y):
                # Calculate patch coordinates
                x_start = i * self.patch_size
                y_start = j * self.patch_size
                x_end = min((i + 1) * self.patch_size, width)
                y_end = min((j + 1) * self.patch_size, height)

                # Crop the patch from the image
                patch_img = F.crop(img, y_start, x_start, y_end - y_start, x_end - x_start)

                # Filter targets within the patch
                filtered_targets = self.filter_targets(target, x_start, y_start, x_end, y_end)
                if filtered_targets is None:
                    continue
                patched_image.append(patch_img)
                patched_targets.append(filtered_targets)
        
        # Convert PIL to a tensor
        patched_images = torch.stack([F.to_tensor(img) for img in patched_image])
        #print(f"patched_images.shape: {patched_images.shape}")
        
        return [patched_images], [patched_targets]

    def filter_targets(self, target, x_start, y_start, x_end, y_end):
            masks = target['masks']
            labels = target['labels']
            boxes = target['boxes']

            filtered_masks = []
            filtered_labels = []
            filtered_boxes = []

            for i, box in enumerate(boxes):
                x_min, y_min, x_max, y_max = box.tolist()
                # Check if box coordinates intersect with patch coordinates
                if x_max > x_start and x_min < x_end and y_max > y_start and y_min < y_end:
                    # Adjust box coordinates to patch space
                    box[0] = max(x_min - x_start, 0)
                    box[1] = max(y_min - y_start, 0)
                    box[2] = min(x_max - x_start, x_end - x_start)
                    box[3] = min(y_max - y_start, y_end - y_start)

                    # Get mask coordinates within the patch
                    #mask_patch = masks[i][int(y_start):int(y_end), int(x_start):int(x_end)]
                    #mask_patch = mask_patch[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                    
                    # Get mask coordinates within the patch
                    #print(f"masks[i] {masks[i]}")
                    #print(f"masks[i].shape: {masks[i].shape}")
                    mask_patch = masks[i][int(y_start):int(y_end), int(x_start):int(x_end)]
                    #print(f"mask_patch.shape: {mask_patch.shape}")
                    #mask_patch = mask_patch[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                    #print(f"mask_patch.shape: {mask_patch.shape}")
                    # if mask_patch is empty, skip it
                    if mask_patch.shape[0] == 0 or mask_patch.shape[1] == 0:
                        continue
                    filtered_masks.append(mask_patch)
                    filtered_labels.append(labels[i])
                    filtered_boxes.append(box)
                    
            # create tensors from the filtered targets; only consider non-empty masks and corresponding labels and boxes
            if len(filtered_masks) == 0:
                #filtered_masks = np.zeros((1,1,1))
                #filtered_labels = np.zeros((1))
                #filtered_boxes = np.zeros((1,4))
                return None
            if len(filtered_boxes) == 0:
                return None
                #filtered_boxes = np.zeros((1,4))
            # stack masks
            filtered_masks = np.stack(filtered_masks)
            #print(f"filtered_masks.shape: {filtered_masks.shape}")
            filtered_masks = torch.as_tensor(filtered_masks, dtype=torch.uint8)
            filtered_labels = torch.tensor(filtered_labels)
            # stack boxes
            filtered_boxes = np.stack(filtered_boxes)
            filtered_boxes = torch.as_tensor(filtered_boxes, dtype=torch.float32)
  

            filtered_targets = {
                'masks': filtered_masks,
                'labels': filtered_labels,
                'boxes': filtered_boxes
            }
            
            #print(f"filtered_targets: {filtered_targets}")

            return filtered_targets