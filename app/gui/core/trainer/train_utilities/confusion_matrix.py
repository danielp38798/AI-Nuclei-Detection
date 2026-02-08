from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import json
import os

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def calculate_confusion_matrix(ground_truth_annotations, predicted_annotations, num_classes):
    # Initialize confusion matrix
    confusion_matrix = [[0] * num_classes for _ in range(num_classes)]

    # Load ground truth and predicted annotations
    print(ground_truth_annotations)
    #if isinstance(ground_truth_annotations, list):
        #ground_truth_annotations = load_json(ground_truth_annotations)


    coco_gt = COCO()
    coco_gt.dataset = load_json(ground_truth_annotations)
    coco_pred = COCO()
    coco_pred.dataset = load_json(predicted_annotations)
    
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

        # Iterate over ground truth annotations
        for gt_cat_id in gt_cat_ids:
            # Check if the predicted annotations contain the same category
            if gt_cat_id in pred_cat_ids:
                # Increment true positive count
                confusion_matrix[gt_cat_id - 1][gt_cat_id - 1] += 1
            else:
                # Increment false negative count
                confusion_matrix[gt_cat_id - 1][-1] += 1

        # Iterate over predicted annotations
        for pred_cat_id in pred_cat_ids:
            # Check if the ground truth annotations contain the same category
            if pred_cat_id not in gt_cat_ids:
                # Increment false positive count
                confusion_matrix[-1][pred_cat_id - 1] += 1

    return confusion_matrix



def plot_confusion_matrix(confusion_matrix, class_names, figsize=(10, 8), cmap='Blues'):
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
    plt.show()



# Example usage
#ground_truth_annotations_file = 'path_to_ground_truth_annotations.json'
#predicted_annotations_file = 'path_to_predicted_annotations.json'
num_classes = 2  # Number of classes in COCO dataset
ground_truth_annotations_file = os.path.join(os.getcwd(), "dummy_ground_truth_annotations.json")
predicted_annotations_file = os.path.join(os.getcwd(), "dummy_predicted_annotations.json")

# Calculate confusion matrix

confusion_matrix = calculate_confusion_matrix(ground_truth_annotations=predicted_annotations_file, 
                                              predicted_annotations=predicted_annotations_file, 
                                              num_classes=num_classes)
print(confusion_matrix)


# Example usage
class_names = ['Class 1', 'Class 2']
plot_confusion_matrix(confusion_matrix, class_names)
