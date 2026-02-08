import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision.datasets import CocoDetection
import torch
from pycocotools.coco import COCO
import os
import numpy as np

import torch
import torchvision
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.models.detection import MaskRCNN
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet50_Weights, ResNet101_Weights
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision import datasets
import glob
import re
from pprint import pprint as pp
from pycocotools.coco import COCO
import json
import numpy as np
from pycocotools import mask as maskUtils

from matplotlib.patches import Patch

from matplotlib import font_manager
font_dirs = [r'disk/AI_nuclei_detection/GUI/cmu-serif']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
# set font
plt.rcParams['font.family'] = 'CMU Serif'
plt.rcParams['font.size'] = 12

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def load_checkpoint(ckpt_dir, model, device):
    """
    Load latest checkpoint if available
    """
    # Get the latest checkpoint
    ckpts = glob.glob(os.path.join(ckpt_dir, "*.pth"))
    ckpts.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))
    if len(ckpts) > 0:
        #ckpt_path = ckpts[-2] # early stopping checkpoint
        ckpt_path = ckpts[-3] # latest checkpoint
        print(f"Loading checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        #model.load_state_dict(checkpoint['model_state_dict'])
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint['epoch']
            print(f"Loaded checkpoint: {ckpt_path} at epoch {epoch}")
        elif "model" in checkpoint:
            model.load_state_dict(checkpoint['model'])
            epoch = checkpoint['epoch']
            print(f"Loaded checkpoint: {ckpt_path} at epoch {epoch}")
        else:
           model = checkpoint
           epoch = None

    else:
        print("No checkpoints found.")
    return model

def get_dataloader(dataset, batch_size, shuffle, num_workers):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)

def collate_fn(batch):
    return tuple(zip(*batch))


def plot(imgs, row_title=None, **imshow_kwargs):
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
                    scores = target.get("scores")
                    if scores is not None:
                        scores = [f"{s.numpy():.2f}" for s in scores]
                    #labels = target.get("labels")   

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

                # e.g. if masks.shape = (44,1,256,256) -> (44,256,256)
                if len(masks.shape) == 4:
                    masks = masks.squeeze(1)

                img = draw_segmentation_masks(img, masks.to(torch.bool), colors=["green"] * masks.shape[0], alpha=.65)
            

            ax = axs[row_idx, col_idx]
            ax.imshow(img.permute(1, 2, 0).numpy(), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
    plt.savefig("preditction_and_gt.png")

def nms(boxes, masks, labels, scores, iou_threshold=0.5):
    """
    Perform non-maximum suppression (NMS) on the predicted boxes and masks.
    """
    # Convert boxes to (x_min, y_min, x_max, y_max) format
    #boxes = torchvision.ops.box_convert(boxes, in_fmt='xywh', out_fmt='xyxy')

    # Perform NMS
    keep = torchvision.ops.nms(boxes, scores, iou_threshold)

    # Filter boxes, masks, labels, and scores
    boxes = boxes[keep]
    masks = masks[keep]
    labels = labels[keep]
    scores = scores[keep]

    return boxes, masks, labels, scores

def plot_ground_truth_and_prediction(img, target, prediction, out_file):
    """
    plots the ground truth (bounding boxes and masks) and the prediction (bounding boxes and masks) on the same image
    """
    # copy the image to the cpu
    img = img[0].cpu()

    # Plot the image and overlay the ground truth and the prediction
    target = [{k: v.to(torch.device('cpu')) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in target]
    target = target[0]
    target['masks'] = target['masks'] > 0.5

    prediction = [{k: v.to(torch.device('cpu')) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in prediction]
    prediction = prediction[0]
    prediction['masks'] = prediction['masks'] > 0.5


    img = F.to_image(img)
    if img.dtype.is_floating_point and img.min() < 0:
        # Poor man's re-normalization for the colors to be OK-ish. This
        # is useful for images coming out of Normalize()
        img -= img.min()
        img /= img.max()

    img = F.to_dtype(img, torch.uint8, scale=True)
    input_img = img


    target_masks = target.get("masks")	
    target_boxes = target.get("boxes")
    target_labels = target.get("labels")
    target_text_labels = [str(l.numpy()) for l in target_labels]
    print(f"target_labels: {target_labels}")
    target_text_colors = [] 
    for l in target_labels:
        if l == 1:
            target_text_colors.append("blue")
        else:
            target_text_colors.append("orange")
    print(f"target_text_colors: {target_text_colors}")

    prediction_masks = prediction.get("masks")
    prediction_boxes = prediction.get("boxes")
    prediction_scores = prediction.get("scores")
    prediction_labels = prediction.get("labels")
    print(f"prediction_labels prior to thresholding: {prediction_labels}")
    print(f"prediction_scores prior to thresholding: {prediction_scores}")

    # only show predictions with a score > 0.5
    confidence_threshold = 0.5
    if prediction_scores is not None:

 

        # implement nms if necessary
        prediction_boxes, prediction_masks, prediction_labels, prediction_scores = nms(prediction_boxes, prediction_masks, prediction_labels, prediction_scores, iou_threshold=0.5)

        prediction_boxes = prediction_boxes[prediction_scores > confidence_threshold]
        prediction_masks = prediction_masks[prediction_scores > confidence_threshold]
        prediction_labels = prediction_labels[prediction_scores > confidence_threshold]
        prediction_scores = prediction_scores[prediction_scores > confidence_threshold]
        print(f"prediction_labels after thresholding: {prediction_labels}")
        print(f"prediction_scores after thresholding: {prediction_scores}")

        pred_text_colors = []
        for l in prediction_labels:
            if l == 1:
                pred_text_colors.append("blue")
            else:
                pred_text_colors.append("orange")
        pred_text_labels = [str(l.numpy()) for l in prediction_labels]
        

    # create a new plot which shows the image and the ground truth and the prediction in two subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    

    # first subplot
    img_for_target = input_img
    axs[0].imshow(img_for_target.permute(1, 2, 0).numpy())
    axs[0].set_title("Ground Truth")
    axs[0].set_axis_off()

    if target_boxes is not None:
        #img_for_target = draw_bounding_boxes(img_for_target, target_boxes, colors="orange", width=2)
        if target_labels is not None:
            img_for_target = draw_bounding_boxes(img_for_target, target_boxes, colors=target_text_colors, width=2, labels=target_text_labels)
    if target_masks is not None:
        if len(target_masks.shape) == 4:
            target_masks = target_masks.squeeze(1)
        #img_for_target = draw_segmentation_masks(img_for_target, target_masks, colors=["green"] * target_masks.shape[0], alpha=.45)
        if target_labels is not None:
            img_for_target = draw_segmentation_masks(img_for_target, target_masks, colors=target_text_colors, alpha=.35)
    axs[0].imshow(img_for_target.permute(1, 2, 0).numpy())
    axs[0].set_title("Ground Truth")


    # second subplot
    image_for_prediction = input_img
    axs[1].imshow(image_for_prediction.permute(1, 2, 0).numpy())
    axs[1].set_title("Prediction")
    if prediction_boxes is not None:
        #image_for_prediction = draw_bounding_boxes(image_for_prediction, prediction_boxes, colors="yellow", width=2)
        if pred_text_labels is not None:
            image_for_prediction = draw_bounding_boxes(image_for_prediction, prediction_boxes, colors=pred_text_colors, width=2, labels=pred_text_labels)
    if prediction_masks is not None:
        if len(prediction_masks.shape) == 4:
            prediction_masks = prediction_masks.squeeze(1)
        #image_for_prediction = draw_segmentation_masks(image_for_prediction, prediction_masks, colors=["blue"] * prediction_masks.shape[0], alpha=.45)
        if pred_text_labels is not None:
            image_for_prediction = draw_segmentation_masks(image_for_prediction, prediction_masks, colors=pred_text_colors, alpha=.35)
    if prediction_scores is not None:
        for i, score in enumerate(prediction_scores):
            if score > confidence_threshold:
                #axs[1].text(prediction_boxes[i][0], prediction_boxes[i][1], f"{score:.2f}", fontsize=8, color="white", bbox=dict(facecolor='black', alpha=0.4))
                # put the score in the top right corner of the bounding box
                # derive score box width from the length of the score and the font size
                score_box_width = len(f"{score:.2f}") * 5
                score_box_height = 12 # fixed height
                x_top_right = prediction_boxes[i][2] - score_box_width # x coordinate of the top right corner of the bounding box
                y_top_right = prediction_boxes[i][1] + score_box_height # y coordinate of the top right corner of the bounding box
                
                # skip the score if the text is outside the image
                if x_top_right > image_for_prediction.shape[2] or y_top_right > image_for_prediction.shape[1]:
                    continue

                axs[1].text(x_top_right, y_top_right, f"{score:.2f}", fontsize=7, color="white", 
                            bbox=dict(facecolor='black', alpha=0.4, boxstyle="round,pad=0.1"))
                                      

    axs[1].imshow(image_for_prediction.permute(1, 2, 0).numpy())
    axs[1].set_title("Prediction")
    axs[1].set_axis_off()
    

    # add a legend to the plot
    color_handles = [Patch(color="blue", label="decondensed nucleus"), Patch(color="orange", label="nucleus")]

    plt.legend(["decondensed_nucleus", "nucleus"], handles=color_handles, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    subplot_file = os.path.basename(out_file).replace(".png", "_subplots.png")
    plt.savefig(subplot_file)
    subplot_file = os.path.basename(out_file).replace(".png", "_subplots.pdf")
    plt.savefig(subplot_file)
    

def calculate_confusion_matrix(ground_truth_annotations, predicted_annotations, num_classes):
    # Initialize confusion matrix
    confusion_matrix = [[0] * num_classes for _ in range(num_classes)]

    """
    fields of the confusion matrix:
    - True Positives (TP): These are the cases where the model correctly predicts the positive class.
    - False Positives (FP): These are the cases where the model incorrectly predicts the positive class when the actual class is negative.
    - True Negatives (TN): These are the cases where the model correctly predicts the negative class as negative.
    - False Negatives (FN): These are the cases where the model incorrectly predicts the negative class as positive.
    
                            |--actual values --|
    predicted values -------|                  |
                            decond/     cond/
                            positive  negative
    decond/positive        |   TP   |   FP     |
    cond/negative          |   FN   |   TN     |

    Example: Confusion matrix: [[0, 6318], [0, 49005]]'
    
    1. The model correctly predicts the "decond" class as "decond" (True Negatives, TN) 0 times.
    2. The model incorrectly predicts the "decond" class as "cond" (False Positives, FP) 6318 times.
    3. The model correctly predicts the "cond" class as "cond" (True Positives, TP) 49005 times.
    4. The model incorrectly predicts the "cond" class as "decond" (False Negatives, FN) 0 times.

    """

    # Load ground truth and predicted annotations

    #if isinstance(ground_truth_annotations, list):
        #ground_truth_annotations = load_json(ground_truth_annotations)
    coco_gt = COCO(ground_truth_annotations)
    coco_pred = COCO()
    coco_pred.dataset = load_json(predicted_annotations)[0]
    # Get image IDs
    img_ids = coco_gt.getImgIds()


    # Iterate over images
    for img_id in img_ids:
        # Get ground truth annotations for the image
        gt_ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        gt_anns = coco_gt.loadAnns(gt_ann_ids)

        # Get predicted annotations for the image
        pred_ann_ids = coco_pred.getAnnIds(imgIds=img_id)
        pred_anns = coco_pred.loadAnns(pred_ann_ids)

        # Extract category IDs for ground truth and predicted annotations
        gt_cat_ids = [ann['category_id'] for ann in gt_anns]
        pred_cat_ids = [ann['category_id'] for ann in pred_anns]

        fp_count = 0
        fn_count = 0
        tp_count = 0

        # Iterate over ground truth annotations
        for gt_cat_id in gt_cat_ids:
            # Check if the predicted annotations contain the same category
            if gt_cat_id in pred_cat_ids:
                # Increment true positive count
                tp_count += 1
                confusion_matrix[gt_cat_id - 1][gt_cat_id - 1] += 1 # this is the top left corner of the confusion matrix
            else:
                # Increment false negative count
                fn_count += 1
                confusion_matrix[gt_cat_id - 1][-1] += 1 # this is the 

        # Iterate over predicted annotations
        for pred_cat_id in pred_cat_ids:
            # Check if the ground truth annotations contain the same category
            if pred_cat_id not in gt_cat_ids:
                # Increment false positive count
                fp_count += 1
                confusion_matrix[-1][pred_cat_id - 1] += 1

    return confusion_matrix



def plot_confusion_matrix(confusion_matrix, class_names, figsize=(10, 8), cmap='Blues', out_file="confusion_matrix.png"):
    """
    Plot confusion matrix as a heatmap.

    Parameters:
        confusion_matrix (list of lists): Confusion matrix to be plotted.
        class_names (list of str): List of class names.
        figsize (tuple): Size of the figure (width, height).
        cmap (str): Colormap to be used for the heatmap.

    Returns:
        None
    """
    # Convert confusion matrix to numpy array
    confusion_matrix = np.array(confusion_matrix)

    # Normalize the confusion matrix
    confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

    # Plot heatmap
    plt.figure(figsize=figsize)
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    plt.title('Confusion Matrix')
    plt.colorbar()

    # Add labels
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Add text annotations
    thresh = confusion_matrix.max() / 2.
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            plt.text(j, i, format(confusion_matrix[i, j], '.2f'),
                     ha="center", va="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    #plt.show()
    plt.savefig(out_file)


def convert_to_coco_format(predictions, image_ids, categories, out_file='predictions_coco_format.json'):
    """
    Convert predictions of Mask R-CNN into COCO format.

    Parameters:
        predictions (list of dict): List of prediction dictionaries containing masks, scores, etc.
        image_ids (list): List of image IDs corresponding to the predictions.
        categories (list of dict): List of category dictionaries containing category IDs and names.

    Returns:
        coco_annotations (list): List of COCO format annotations.
    """
    coco_annotations = []

    for img_id, prediction in zip(image_ids, predictions):
        img_annotations = []

        for mask, score, box, label in zip(prediction['masks'], prediction['scores'], prediction['boxes'], prediction['labels']):
            # Convert mask to RLE
            mask = mask[0].mul(255).byte().cpu().numpy()
            mask = np.asfortranarray(mask)
            rle = maskUtils.encode(mask)
            segmentation = rle["counts"].decode("utf-8") # Convert bytes to string

            # Create annotation dictionary
            annotation = {
                'image_id': img_id,
                'category_id': label.item(),
                #'segmentation': rle,
                # json cannot serialize type bytes so we convert it to a list
                'segmentation': segmentation,
                'score': score.item(),
                'bbox': box.tolist()
            }

            img_annotations.append(annotation)

        coco_annotations.extend(img_annotations)

    # Save as JSON file
    with open(out_file, 'w') as f:
        json.dump(coco_annotations, f)
    print(f'Annotations saved to {out_file}')





def main():
    # Assuming COCO dataset is correctly located in 'data/coco'
    root = r"disk/AI_nuclei_detection/train_data/new_balanced_data/sliced_coco"
    annFile = r"disk/AI_nuclei_detection/train_data/new_balanced_data/sliced_coco/annotations/instances_val2017.json"

    # Load the dataset
    dataset = CocoDetection(root=f'{root}/val2017',
                            annFile=annFile,
                            transform=T.Compose([T.ToTensor()]))#,
                                                #T.Normalize(mean=[0.1444, 0.1444, 0.1444],
                                                #                std=[0.1280, 0.1280, 0.1280])
                                                #]))
    dataset = datasets.wrap_dataset_for_transforms_v2(dataset, target_keys=["masks", "boxes", "labels"])

    # Initialize the model (Assuming you've a trained model)
    # For demonstration, we load a pre-trained model.
    #model = get_model_with_backbone(num_classes=3, backbone_name="resnet50", pretrained=False, trainable_layers=3)
    model = torchvision.models.get_model(
        "maskrcnn_resnet50_fpn", weights=None, weights_backbone=ResNet50_Weights.IMAGENET1K_V1, num_classes=3)
    # set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the model
    checkpoint_dir = r"disk/AI_nuclei_detection/trainer/detection/run_20240325-163220"
    model = load_checkpoint(ckpt_dir=checkpoint_dir, model=model, device=device)
    # send the model to the device
    model.to(device)
    # set the model to evaluation mode
    model.eval()

    # Select a random image from the dataset
    random_index = np.random.randint(len(dataset))
    print(f"Random index: {random_index}")
    img, target = dataset[random_index]

    # Move the image to the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Get the prediction
    with torch.no_grad():
        img = [img.to(device)]

            # extract data attribute for mask and bounding boxes if value is of instance tv.Tensor
        new_target_list = []

        # in each dict of the list check the masks and boxes
        new_target = target
        if "boxes" in target:
            converted_bboxes = target['boxes'].data
            converted_masks = target['masks'].data
            new_target['boxes'] = converted_bboxes
            new_target['masks'] = converted_masks

            new_target_list.append(new_target)

            targets = new_target_list
            target_for_model = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            #losses, prediction = model(img, target_for_model)
            prediction = model(img, target_for_model)
            #print(f"validation loss: {losses}")

            # convert the prediction to coco format and store
            # the predictions in a json file
            #predictions = [{'masks': ..., 'labels': ..., 'scores': ..., 'boxes': ...}]  # Replace ... with your data
            image_ids = [1]
            categories = [{'id': 1, 'name': 'decondensed_nucleus'}, {'id': 2, 'name': 'nucleus'}]  # Replace with your category IDs and names
            out_file = os.path.join(checkpoint_dir, "predictions_coco_format.json")
            convert_to_coco_format(prediction, image_ids, categories, out_file=out_file)

            # read the json file and calculate the confusion matrix
            ground_truth_annotations = annFile
            predicted_annotations = out_file
            num_classes = 2
            confusion_matrix = calculate_confusion_matrix(ground_truth_annotations, predicted_annotations, num_classes)
            pp(f"Confusion matrix: {confusion_matrix}")
            class_names = ["decondensed_nucleus", "nucleus"]
            out_file = os.path.join(checkpoint_dir, "confusion_matrix.png")
            plot_confusion_matrix(confusion_matrix, class_names, out_file=out_file)#
            print(f"Plotting the image and the prediction")
            #plot([(img, target), (img, prediction)], row_title=["Ground Truth", "Prediction"])
            #plot([(img, target)], row_title=["Ground Truth"])
            #plot([(img, prediction)], row_title=["Prediction"])
            out_file = os.path.join(checkpoint_dir, "preditction_and_gt.png")
            plot_ground_truth_and_prediction(img, targets, prediction, out_file=out_file)
        else:
            print("No boxes in target")



    

if __name__ == "__main__":
    main()

"""

If you have a confusion matrix where one class ("cond") is correctly predicted while the other class ("decond") is consistently predicted as the negative class, here's how you can interpret it:

True Positives (TP): These are the cases where the model correctly predicts the "cond" class. These would be represented as 1s in the confusion matrix for the "cond" class.

False Positives (FP): These are the cases where the model incorrectly predicts the "cond" class when the actual class is "decond". Since you mentioned that the left quarters belonging to "decond" have values of 0, this indicates that there are no false positives for the "cond" class.

True Negatives (TN): These are the cases where the model correctly predicts the "decond" class as "decond". Since you mentioned that the left quarters belonging to "decond" have values of 0, this indicates that all predictions for "decond" in those quarters are correct.

False Negatives (FN): These are the cases where the model incorrectly predicts the "decond" class as "cond". Since you mentioned that the right quarters belonging to "cond" have values of 1, this indicates that there are false negatives for the "decond" class.

Based on this interpretation:

The model performs well in predicting the "cond" class, as indicated by the presence of 1s in the corresponding quarters of the confusion matrix.
However, the model struggles to predict the "decond" class, as indicated by the presence of 1s in the quarters corresponding to "decond" in the confusion matrix.
In summary, the values you described in the confusion matrix suggest that the model is effective in predicting one class ("cond") but is less effective in predicting the other class ("decond"), potentially leading to false negatives for the "decond" class.
"""