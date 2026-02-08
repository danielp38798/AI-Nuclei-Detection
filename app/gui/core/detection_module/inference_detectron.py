# Script to perform inference using the trained model

# Import necessary libraries
import gui.core.detection_module.detectron2
from gui.core.detection_module.detectron2.utils.logger import setup_logger
setup_logger()

# Import necessary components from gui.core.detection_module.detectron2
from gui.core.detection_module.detectron2.engine import DefaultPredictor
from gui.core.detection_module.detectron2.config import get_cfg
from gui.core.detection_module.detectron2.data import MetadataCatalog
from gui.core.detection_module.detectron2.utils.visualizer import Visualizer, ColorMode
import cv2
import os

# Function to set up the configuration and predictor
def setup_predictor():
    cfg = get_cfg()
    cfg.merge_from_file("./configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Adjust this to match the number of classes
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # Path to the trained model weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set the testing threshold for this model

    predictor = DefaultPredictor(cfg)
    return predictor, cfg

# Function to create custom metadata
def create_custom_metadata():
    # Create metadata
    metadata = MetadataCatalog.get("custom_dataset")
    metadata.thing_classes = ["nucleus"]  # Replace with your class names
    return metadata

# Function to perform inference on an image
def perform_inference(predictor, cfg, metadata, image_path, output_path):
    img = cv2.imread(image_path)
    outputs = predictor(img)
    
    # Visualize the predictions
    v = Visualizer(img[:, :, ::-1],
                   metadata=metadata,
                   scale=0.8,
                   instance_mode=ColorMode.IMAGE_BW)  # Remove the colors of unsegmented pixels
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    
    # Save the output image
    result = v.get_image()[:, :, ::-1]
    cv2.imwrite(output_path, result)

# Main function to load the model and run inference
def main():
    # Setup the predictor
    predictor, cfg = setup_predictor()
    
    # Create custom metadata
    metadata = create_custom_metadata()
    
    # Define paths for input images and output directory
    input_images_dir = "./demo"
    output_images_dir = "./output/inference_test"
    os.makedirs(output_images_dir, exist_ok=True)

    # Perform inference on each image
    for image_filename in os.listdir(input_images_dir):
        if image_filename.endswith(".png"):
            image_path = os.path.join(input_images_dir, image_filename)
            output_path = os.path.join(output_images_dir, image_filename)
            perform_inference(predictor, cfg, metadata, image_path, output_path)
        

if __name__ == "__main__":
    main()
