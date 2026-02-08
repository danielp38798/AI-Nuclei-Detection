#!/usr/bin/env python
"""
Mask R-CNN evaluation script based on the scripts provided in detectron2 git repository: https://github.com/facebookresearch/detectron2
Running this script will perform the evaluation of the trained Detectron2 model on the custom dataset. 
"""


# Import necessary libraries
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# Import necessary components from detectron2
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
import os
import cv2
import matplotlib.pyplot as plt
from sahi.postprocess.combine import NMMPostprocess, NMSPostprocess
from sahi.predict import get_prediction
from sahi.auto_model import AutoDetectionModel
import torch
import json

from detectron2.structures import BoxMode
from pycocotools import mask as mask_utils
import numpy as np


POSTPROCESS_NAME_TO_CLASS = {
    "NMM": NMMPostprocess,
    "NMS": NMSPostprocess,
}



# Function to set up the configuration
def setup_cfg(config_path="./configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml", train_output_dir=r"./output_run_27_06_24/"):
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    #cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_best.pth")  # Path to the trained model weights
    #cfg.merge_from_file("./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    #cfg.MODEL.WEIGHTS = r"./output/model_0039999.pth"
    cfg.MODEL.WEIGHTS = os.path.join(train_output_dir, "best_checkpoint_bbox.pth")  # Path to the trained model weights
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Adjust this to match the number of classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # Set the testing threshold for this model
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {cfg.MODEL.DEVICE}")
    
    cfg.INPUT.MIN_SIZE_TEST = 400
    cfg.INPUT.MAX_SIZE_TEST = 400
    #cfg.INPUT.MIN_SIZE_TRAIN = 400
    #cfg.INPUT.MAX_SIZE_TRAIN = 400
    

    # dump config to yaml file
    yaml_cfg = detectron2.config.CfgNode.dump(cfg)
    detection_module_cfg_file = os.path.join(train_output_dir, "config_infer.yaml")
    print(f"Saving config to {detection_module_cfg_file}")
    
    with open(detection_module_cfg_file, 'w') as fp:
        fp.write(yaml_cfg)

    return cfg

# Function to create custom metadata
def create_custom_metadata():
    # Create metadata
    metadata = MetadataCatalog.get("custom_dataset_val")
    metadata.thing_classes = ["nucleus"]  # Replace with your class names
    metadata.thing_colors = [[0, 255, 0]]  # Replace with your class colors

    return metadata

# Function to visualize predictions and ground truth side by side
def visualize_predictions_side_by_side(predictor, yaml_cfg, config_path, metadata, dataset_name, output_dir):
    dataset_dicts = DatasetCatalog.get(dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    # Prepare CSV logging
    csv_log_path = os.path.join(output_dir, "detection_log.csv")
    with open(csv_log_path, "w") as log_file:
        #log_file.write("Image, Ground Truth Annotation Count, Model Prediction Count\n")
        log_file.write("Image,Ground Truth Annotation Count,Model Prediction Count,Ground Truth Area,Model Prediction Area,Ground Truth Area Normalized,Model Prediction Area Normalized\n")

    for d in dataset_dicts:
    
        img = cv2.imread(d["file_name"])
        print(f"Processing image: {d['file_name']}")
        
        # Get the number of instances in the ground truth
        num_gt_instances = len(d["annotations"])
        print(f"Number of ground truth instances in image {d['file_name']}: {num_gt_instances}")
        
        # Ground truth visualization
        v_gt = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.8, font_size_scale=0.8, instance_mode=ColorMode.SEGMENTATION)
        
        # create the same color palette as in the training
        
        v_gt = v_gt.draw_dataset_dict(d,show_labels=False)
        gt_img = v_gt.get_image()[:, :, ::-1]

        # Post processing
        #postprocess_constructor = POSTPROCESS_NAME_TO_CLASS["NMM"]
        postprocess_constructor = POSTPROCESS_NAME_TO_CLASS["NMS"]
      
        print(f"Model weights: {yaml_cfg.MODEL.WEIGHTS}")
        print(f"Config path: {config_path}")
        detection_model = AutoDetectionModel.from_pretrained(
                model_type='detectron2',
                model_path=yaml_cfg.MODEL.WEIGHTS,
                config_path=config_path,
                confidence_threshold=0.5,
                load_at_init=True,
            )
        detection_model.category_mapping = {"0": "nucleus"}
        detection_model.category_names = ["nucleus"]
        
        prediction_result = get_prediction(
            image=img,
            detection_model=detection_model,
            shift_amount=[0, 0],
            full_shape=(400, 400),

            #postprocess=NMMPostprocess(
                           # match_threshold=0.5,
                            #match_metric="IOS",
                        #class_agnostic=False,
            
            postprocess=NMSPostprocess(
                match_threshold=0.5,
                match_metric="IOU",
               class_agnostic=False
            ),
        )

        predictions_tmp_dir = os.path.join(output_dir, "predictions")
        image_filename = os.path.basename(d["file_name"])
        prediction_result.export_visuals(export_dir=predictions_tmp_dir,
                                    file_name=f"{os.path.splitext(os.path.basename(image_filename))[0]}_prediction",
                                    text_size=0.25, rect_th=2, hide_labels=True, hide_conf=True, color=(0, 255, 0))
        pred_img = prediction_result.results_dict["image"]

        # Get the number of detected instances
        pred_objects = prediction_result.object_prediction_list
        num_detected_instances = len(pred_objects)
        print(f"Number of detected instances in image {d['file_name']}: {num_detected_instances}")

        # Plot ground truth and prediction side by side
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[0].imshow(gt_img)
        ax[0].set_title("Ground Truth")
        ax[0].axis('off')
        ax[1].imshow(pred_img)
        ax[1].set_title("Prediction")
        ax[1].axis('off')
        
        # add the number of detected instances onderneath the images
        ax[0].text(0.5, -0.1, f"Annotation Count: {num_gt_instances}", ha='center', va='top', fontsize=12, transform=ax[0].transAxes)
        ax[1].text(0.5, -0.1, f"Detection Count: {num_detected_instances}", ha='center', va='top', fontsize=12, transform=ax[1].transAxes)
        
        
        gt_total_area = 0
        for ann in d["annotations"]:
            if "segmentation" in ann:
                rle = mask_utils.frPyObjects(ann["segmentation"], d["height"], d["width"])
                gt_total_area += mask_utils.area(rle).sum()

        pred_total_area = 0
        for obj in pred_objects:
            if obj.mask is not None and obj.mask.bool_mask is not None:
                pred_total_area += np.sum(obj.mask.bool_mask.astype(np.uint8))
                
        # for relative comparison normalize the area to the image area
        img_area = d["height"] * d["width"]
        gt_area_normalized = gt_total_area / img_area
        pred_area_normalized = pred_total_area / img_area


        # Log to CSV
        with open(csv_log_path, "a") as log_file:
            log_file.write(f"{d['file_name']},{num_gt_instances},{num_detected_instances},{gt_total_area},{pred_total_area},{gt_area_normalized},{pred_area_normalized}\n")
        
    
        # Save the plot
        combined_output_path = os.path.join(output_dir, f"comparison_{os.path.basename(d['file_name'])}")
        plt.savefig(combined_output_path)
        plt.close()

# Function to evaluate the model
def evaluate_model():
    """
    Evaluates the trained Detectron2 model on the custom dataset and visualizes predictions side by side with ground truth.
    """
    # Register the custom COCO dataset
    register_coco_instances("my_dataset_val2017", {}, 
                            r"./TRAINDATA/sliced_coco/annotations/instances_val2017.json", 
                            r"./TRAINDATA/sliced_coco/val2017")

    # Setup the configuration
    ##config = "mask_rcnn_R_50_C4_3x.yaml"
    #config = "mask_rcnn_R_50_DC5_3x.yaml"
    #config = "mask_rcnn_R_50_FPN_3x.yaml"
    
    #config = "mask_rcnn_R_101_C4_3x.yaml"
    #config = "mask_rcnn_R_101_DC5_3x.yaml"
    config = "mask_rcnn_R_101_FPN_3x.yaml"
    
    out_dir_name = "training_output_2025-05-05-15-54-38_mask_rcnn_R_101_FPN_3x"
    #config_path = f"./configs/COCO-InstanceSegmentation/
    config_path = os.path.join(os.getcwd(), "configs", "COCO-InstanceSegmentation", config)
    #config_path = os.path.join(os.getcwd(), "output_2025-04-09-15-38-33", "config.yaml")
    print(f"Config path: {config_path}")
    #train_output_dir = r"./output_2025-04-09-10-04-43/model_best.pth"
    train_output_dir = os.path.join(os.getcwd(), out_dir_name)
    print(f"Train output dir: {train_output_dir}")
    cfg = setup_cfg(config_path=config_path, train_output_dir=train_output_dir)

    # Create custom metadata
    metadata = create_custom_metadata()

    # Create evaluator
    evaluator = COCOEvaluator("my_dataset_val2017", cfg, False, output_dir=os.path.join(train_output_dir, "eval"))   
    
    # Create data loader
    val_loader = build_detection_test_loader(cfg, "my_dataset_val2017")

    # Create the predictor
    predictor = DefaultPredictor(cfg)

    # Evaluate the model
    print("Evaluating model...")
    metrics = inference_on_dataset(predictor.model, val_loader, evaluator)
    
    print("Evaluation results:")
    print(metrics)

    # Visualize predictions and ground truth side by side
    print("Visualizing predictions...")
    yaml_config_path = os.path.join(train_output_dir, "config_infer.yaml")
    visualize_predictions_side_by_side(predictor, cfg, yaml_config_path, metadata, "my_dataset_val2017", 
                                       os.path.join(train_output_dir, "visualizations"))

if __name__ == "__main__":
    evaluate_model()
