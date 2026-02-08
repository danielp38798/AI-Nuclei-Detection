# Import necessary libraries

from gui.core.detection_module.detectron2.utils.logger import setup_logger
setup_logger()

# Import necessary components from gui.core.detection_module.detectron2
from gui.core.detection_module.detectron2.engine import DefaultTrainer, DefaultPredictor
from gui.core.detection_module.detectron2.config import get_cfg
from gui.core.detection_module.detectron2.data.datasets import register_coco_instances
from gui.core.detection_module.detectron2.evaluation import COCOEvaluator, inference_on_dataset
from gui.core.detection_module.detectron2.data import build_detection_test_loader
from gui.core.detection_module.detectron2.data import MetadataCatalog, DatasetCatalog
from gui.core.detection_module.detectron2.utils.visualizer import Visualizer, ColorMode
import os
import cv2
import matplotlib.pyplot as plt

import torch

torch.random.manual_seed(0)

def get_weights_from_output_dir(output_dir):
    
    #yaml_file = os.path.join(output_dir, "config.yaml")
    weight_file = os.path.join(output_dir, "model_best.pth")
    if not os.path.exists(weight_file):
        print(f"Weight file {weight_file} does not exist.")
    return weight_file

# Function to set up the configuration
def setup_cfg(output_dir,yaml_file):
    cfg = get_cfg()
    weight_file = get_weights_from_output_dir(output_dir)
    cfg.merge_from_file(yaml_file)
    cfg.MODEL.WEIGHTS = weight_file

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Adjust this to match the number of classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set the testing threshold for this model
    cfg.INPUT.MIN_SIZE_TEST = 400
    cfg.INPUT.MAX_SIZE_TEST = 400

    cfg.INPUT.FORMAT = 'BGR'
    cfg.TEST.AUG.ENABLED = False
    cfg.TEST.AUG.MIN_SIZES = (400,)
    cfg.TEST.AUG.MAX_SIZE = 400
    cfg.TEST.AUG.FLIP = False
    return cfg

# Function to create custom metadata
def create_custom_metadata():
    # Create metadata
    metadata = MetadataCatalog.get("custom_dataset_val")
    #metadata.thing_classes = ["class1"]  # Replace with your class names
    return metadata

# Function to visualize predictions and ground truth side by side
def visualize_predictions_side_by_side(predictor, cfg, metadata, dataset_name, output_dir):
    dataset_dicts = DatasetCatalog.get(dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    for d in dataset_dicts:
        img = cv2.imread(d["file_name"])
        
        # Ground truth visualization
        v_gt = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.8, font_size_scale=0.4, instance_mode=ColorMode.IMAGE_BW)
        v_gt = v_gt.draw_dataset_dict(d)
        gt_img = v_gt.get_image()[:, :, ::-1]

        # Prediction visualization
        outputs = predictor(img)
        v_pred = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.8, font_size_scale=0.4, instance_mode=ColorMode.IMAGE_BW)
        v_pred = v_pred.draw_instance_predictions(outputs["instances"].to("cpu"))
        pred_img = v_pred.get_image()[:, :, ::-1]

        # Plot ground truth and prediction side by side
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[0].imshow(gt_img)
        ax[0].set_title("Ground Truth")
        ax[0].axis('off')
        ax[1].imshow(pred_img)
        ax[1].set_title("Prediction")
        ax[1].axis('off')
        
        # Save the plot
        combined_output_path = os.path.join(output_dir, f"comparison_{os.path.basename(d['file_name'])}")
        plt.savefig(combined_output_path)
        plt.close()

# Function to evaluate the model
def evaluate_model(output_dir, yaml_path):
    
    # Register the custom COCO dataset
    register_coco_instances("custom_dataset_val", {}, "./data/coco/val2017/instances_val2017_coco.json", "./data/coco/val2017")

    # Setup the configuration
    cfg = setup_cfg(output_dir, yaml_path)

    # Create custom metadata
    metadata = create_custom_metadata()

    # Create evaluator
    evaluator = COCOEvaluator("custom_dataset_val", cfg, False, output_dir=os.path.join(output_dir, "eval"))   
    
    # Create data loader
    val_loader = build_detection_test_loader(cfg, "custom_dataset_val")

    # Create the predictor
    predictor = DefaultPredictor(cfg)

    # Evaluate the model
    print("Evaluating model...")
    metrics = inference_on_dataset(predictor.model, val_loader, evaluator)

    print("Evaluation results:")
    print(metrics)

    # Visualize predictions and ground truth side by side
    print("Visualizing predictions...")
    visualize_predictions_side_by_side(predictor, cfg, metadata, "custom_dataset_val", os.path.join(output_dir, "visualizations"))

if __name__ == "__main__":
    import time
    output_dir = f"./output_{time.strftime('%Y%m%d_%H%M%S')}"
    
    
    #yaml_path = "./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    yaml_path = "./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml"
    
    #yaml_path = "./configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
    #yaml_path = "./configs/COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml"

    
    evaluate_model(output_dir,yaml_path)
