# sahi test inference
import os
import cv2
import gui.core.detection_module.detectron2.config
from gui.core.sahi.auto_model import AutoDetectionModel
from gui.core.sahi.predict import get_sliced_prediction, predict, get_prediction
from gui.core.sahi.utils.file import download_from_url
from gui.core.sahi.utils.cv import read_image
from IPython.display import Image
from gui.core.json_settings import Settings
import torch
import glob
import time
from pycocotools import mask
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from pprint import pprint as pp
import torchvision.transforms.functional as F
#from ultralytics import YOLO
import gui.core.detection_module.detectron2
from gui.core.detection_module.detectron2.engine import DefaultPredictor
from gui.core.detection_module.detectron2.config import get_cfg
from gui.core.detection_module.detectron2.data import MetadataCatalog
from gui.core.detection_module.detectron2.utils.visualizer import Visualizer, ColorMode
import gc 
from typing import Any, Generator, List, Tuple
import sys
import pandas as pd
from pycocotools import mask
import matplotlib.patches as mpatches
from sklearn.neighbors import KDTree
import seaborn as sns
import networkx as nx



class JSONSerializer:
    def serialize_with_conversion(self, obj: Any, filepath: str) -> None:
        """
        Serialize the object to JSON, converting numpy arrays to lists of strings.

        Args:
            obj (Any): The object to serialize. Can be a numpy array, dictionary, list, tuple, set, or any other serializable object.
            filepath (str): The path to the file where the JSON content will be written.

        Returns:
            None
        """
        def convert(obj: Any) -> Any:
            if isinstance(obj, np.ndarray):
                # Convert array elements to strings
                return obj.astype(str).tolist()
            elif isinstance(obj, dict):
                # Recursively apply to dictionary items
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                # Apply to each item in the list
                return [convert(v) for v in obj]
            elif isinstance(obj, tuple):
                # Apply to each item in the tuple
                return tuple(convert(v) for v in obj)
            elif isinstance(obj, set):
                # Apply to each item in the set
                return {convert(v) for v in obj}
            else:
                return obj

        pp(f"obj: {obj}")
        converted_obj = convert(obj)
        pp(f"converted_obj: {converted_obj}")
        with open(filepath, 'w') as json_file:
            json.dump(converted_obj, json_file, indent=1)

    def deserialize(self, filepath: str) -> Any:
        """
        Deserialize JSON content from a file. Assumes numpy arrays were stored as lists of strings.

        Args:
            filepath (str): The path to the file from which the JSON content will be read.

        Returns:
            Any: The deserialized object.
        """
        with open(filepath, 'r') as json_file:
            return json.load(json_file)


class NumpyEncoder(json.JSONEncoder):
    """
    Special json encoder for numpy types.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    

def get_base_path() -> str:
    """
    Get the base path of the script or the bundled executable.

    Returns:
        str: The base path of the script or the bundled executable.
    """
    if getattr(sys, 'frozen', False):
        # If the application is run as a bundled executable, the PyInstaller
        # bootloader sets a sys._MEIPASS attribute to the path of the temp folder it
        # extracts its bundled files to.
        return sys._MEIPASS
    else:
        # Otherwise, just use the directory of the script being run
        return os.getcwd()

def map_detection_accuracy(accuracy: int) -> float:
    """
    Maps a detection accuracy value to a corresponding confidence threshold.

    Detection accuracy could be a value between 1 and 10:
    - 1 = lowest accuracy allowing the most cells to be detected even with low contrast.
    - 10 = highest accuracy allowing only the most distinct cells to be detected.

    Args:
        accuracy (int): The detection accuracy level, ranging from 1 to 10.

    Returns:
        float: The corresponding confidence threshold for the given detection accuracy.
    """
    # Map detection accuracy to confidence threshold up to 98 percent confidence
    confidence_scores = np.linspace(0.5, 0.98, 10).tolist()
    confidence_scores = [round(score, 2) for score in confidence_scores]
    detection_accuracy = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    detection_accuracy_map = dict(zip(detection_accuracy, confidence_scores))
    
    return detection_accuracy_map[accuracy]

def sanitise_coco_data(coco_data: dict) -> dict:
    """
    Remove invalid segmentations from a COCO file. 
    A segmentation polygon is invalid if it has less than 3 pairs of (x, y) coordinates.

    Args:
        coco_data (dict): The COCO data containing annotations to be sanitized.

    Returns:
        dict: The sanitized COCO data with invalid segmentations removed.
    """
    for annotation in coco_data['annotations']:
        segmentations = annotation['segmentation']
        sanitized_segmentations = []
        for segmentation in segmentations:
            if len(segmentation) >= 6:  # Each (x, y) pair takes 2 elements, so 3 pairs = 6 elements
                sanitized_segmentations.append(segmentation)
        annotation['segmentation'] = sanitized_segmentations
    return coco_data

class ModelInteractor:
    """
    Model interface for evaluation.
    This class handles the initialization and interaction with different AI models for object detection and segmentation.
    """

    def __init__(self, 
                    instances_dict: dict, 
                    output_dir: str,
                    physical_image_width: float, 
                    store_coco_files: bool, 
                    store_csv_file: bool, 
                    store_xlsx_file: bool):
        """
        Initialize the ModelInteractor with the given parameters.

        Args:
            instances_dict (dict): Dictionary mapping instance IDs to instance names.
            output_dir (str): Directory where output files will be stored.
            physical_image_width (float): Physical width of the image in millimeters.
            store_coco_files (bool): Whether to store COCO format files.
            store_csv_file (bool): Whether to store results in CSV format.
            store_xlsx_file (bool): Whether to store results in XLSX format.
        """
        self.settings = Settings().items
        self.results_dict = None

        model_name = self.settings["processing_settings"]["model_selection"]
        print(f"Model name: {model_name}")
        
        if "maskrcnn_resnet101_dc5" in model_name or "Accurate Model" in model_name:
            self.model_name = "maskrcnn_resnet101_dc5"
            print("Using maskrcnn_resnet101_dc5 model.")
        elif "maskrcnn_resnet50_c4" in model_name or "Small Model (R_50_C4)" in model_name:
            self.model_name = "maskrcnn_resnet50_c4"
            print("Using maskrcnn_resnet50_c4 model.")
        else:
            print("Model not found.")
            self.model_name = "maskrcnn_resnet101_dc5"

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.detection_accuracy = float(self.settings["processing_settings"]["detection_accuracy"])
        self.confidence_threshold = self.detection_accuracy
        print(f"Confidence threshold: {self.confidence_threshold}")

        self.script_dir = get_base_path()   # script directory
        self.ckpt_dir = os.path.join(self.script_dir, "application_resources", "models")   # checkpoint directory
        self.instances_dict = instances_dict
        self.physical_image_width = physical_image_width # in mm
        if isinstance(output_dir, tuple):
            self.output_dir = output_dir[0]
        else:
            self.output_dir = output_dir
            self.create_folder(self.output_dir)
        
        self.aux_files_dir = os.path.join(self.output_dir, "_aux_files")
        self.create_folder(self.aux_files_dir)
        
        self.image_analysis_dir = os.path.join(self.output_dir, "01_image_analysis")
        self.create_folder(self.image_analysis_dir)
        
        
        self.excel_results_dir = os.path.join(self.image_analysis_dir, "00_excel_results")
        self.create_folder(self.excel_results_dir)
        
        self.csv_results_dir = os.path.join(self.image_analysis_dir, "01_csv_results")
        self.create_folder(self.csv_results_dir)
        
        self.segmentation_results_dir = os.path.join(self.image_analysis_dir, "03_segmentation_results")
        self.create_folder(self.segmentation_results_dir)
        
        self.coco_results_dir = os.path.join(self.image_analysis_dir, "04_coco_results")
        self.create_folder(self.coco_results_dir)
        
        self.clusters_results_dir = os.path.join(self.image_analysis_dir, "05_clusters_results")
        self.create_folder(self.clusters_results_dir)


        self.store_coco_files = store_coco_files
        self.store_binary_masks = True
        self.store_csv_file = store_csv_file
        self.store_xlsx_file = store_xlsx_file
        
        self.instance_results_df = None
        self.instance_results_dict = None
        self.result = None

        self.detection_module_cfg = None

        # read model_hyperparameters.json in ./applications_resources/models
        self.model_hyperparameters_path = os.path.join(get_base_path(), "application_resources", "models", "model_hyperparameters.json")
        with open(self.model_hyperparameters_path, 'r') as fp:
            self.model_hyperparameters = json.load(fp)
            
        self.mask_rcnn_resnet50_c4_hyperparameters = self.model_hyperparameters["maskrcnn_resnet50_c4"]
        self.mask_rcnn_resnet101_dc5_hyperparameters = self.model_hyperparameters["maskrcnn_resnet101_dc5"]


        self.serialier = JSONSerializer()

        # setup the model
        if not hasattr(self, 'model'):
            self.initialize_model()

    def create_folder(self, folder_name: str) -> None:
        """
        Create a folder if it does not already exist.

        Args:
            folder_name (str): The path of the folder to create.

        Returns:
            None
        """
        if not os.path.exists(folder_name):
            os.makedirs(folder_name, exist_ok=True)

    def setup_detection_module_predictor(self, model_name: str = "maskrcnn_resnet101_dc5") -> None:
        """
        Set up the detection module predictor with the specified model configuration.

        This function initializes the configuration for the detection module predictor
        based on the provided model name. It loads the appropriate configuration file,
        sets the number of classes, loads the latest model weights, and sets the device
        for inference.

        Args:
            model_name (str): The name of the model to set up. Options include:
                - "maskrcnn_resnet101_dc5"
                - "maskrcnn_resnet50_c4"
        Returns:
            None
        """
        print("Setting up detection_module predictor...")
        cfg = get_cfg()
        if model_name == "maskrcnn_resnet50_c4":
            config_path = os.path.join(self.script_dir, "gui", "core", "detection_module", "configs", "COCO-InstanceSegmentation", "mask_rcnn_R_50_C4_3x.yaml")
        elif model_name == "maskrcnn_resnet101_dc5":
            config_path = os.path.join(self.script_dir, "gui", "core", "detection_module", "configs", "COCO-InstanceSegmentation", "mask_rcnn_R_101_DC5_3x.yaml")
        cfg.merge_from_file(config_path)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Adjust this to match the number of classes
        self.detection_module_checkpoint_dir = os.path.join(self.ckpt_dir, model_name)
        list_of_files = []
        for root, dirs, files in os.walk(self.detection_module_checkpoint_dir):
            for file in files:
                if file.endswith(".pth"):
                    list_of_files.append(os.path.join(root, file))
        
        latest_file = max(list_of_files, key=os.path.getctime) 
        cfg.MODEL.WEIGHTS = latest_file  # Path to the trained model weights
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # Set the testing threshold for this model
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        # dump config to yaml file
        yaml_cfg = gui.core.detection_module.detectron2.config.CfgNode.dump(cfg)
        self.detection_module_cfg_file = os.path.join(self.detection_module_checkpoint_dir, "config.yaml")
        
        with open(self.detection_module_cfg_file, 'w') as fp:
            fp.write(yaml_cfg)

        # store the config as a class attribute
        self.detection_module_cfg = cfg

    def run_detection_module_model(self, image_path: str, output_path: str) -> None:
        """
        Run the detection module model on a given image and save the output.

        This function reads an image from the specified path, performs inference using the detection module predictor,
        visualizes the predictions, and saves the output image to the specified path.

        Args:
            image_path (str): The path to the input image file.
            output_path (str): The path where the output image with visualized predictions will be saved.

        Returns:
            None
        """
        # Perform inference on the image
        img = cv2.imread(image_path)
        outputs = self.detection_module_predictor(img)
        
        # Visualize the predictions
        v = Visualizer(img[:, :, ::-1],
                        metadata=self.detection_module_metadata,
                        scale=0.8,
                        instance_mode=ColorMode.IMAGE_BW)  # Remove the colors of unsegmented pixels
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        
        # Save the output image
        result = v.get_image()[:, :, ::-1]
        cv2.imwrite(output_path, result)
        
    def initialize_model(self) -> None:
        """
        Initialize the AI model based on the specified model name.

        This function sets up the AI model for object detection and segmentation. It supports various models including
        Mask R-CNN with different backbones and YOLOv11. The function loads the appropriate model weights, sets the 
        confidence threshold, and prepares the model for inference.

        Args:
            None

        Returns:
            None
        """
        print("Initializing AI model...")

        if self.model_name == "maskrcnn_resnet101_dc5":
            self.setup_detection_module_predictor(model_name="maskrcnn_resnet101_dc5")
            self.detection_model = AutoDetectionModel.from_pretrained(
                model_type='detectron2',
                model_path=self.detection_module_cfg.MODEL.WEIGHTS,
                config_path=self.detection_module_cfg_file,
                confidence_threshold=self.confidence_threshold,
                device=self.device,
            )
            category_mapping = {str(k): v for k, v in self.instances_dict.items()}
            self.detection_model.category_mapping = category_mapping
            self.detection_model.category_names = list(self.instances_dict.values())

        elif self.model_name == "maskrcnn_resnet50_c4":
            self.setup_detection_module_predictor(model_name="maskrcnn_resnet50_c4")
            self.detection_model = AutoDetectionModel.from_pretrained(
                model_type='detectron2',
                model_path=self.detection_module_cfg.MODEL.WEIGHTS,
                config_path=self.detection_module_cfg_file,
                confidence_threshold=self.confidence_threshold,
                device=self.device,
                load_at_init=True,
            )
            category_mapping = {str(k): v for k, v in self.instances_dict.items()}
            self.detection_model.category_mapping = category_mapping
            self.detection_model.category_names = list(self.instances_dict.values())

        elif self.model_name == "yolov11":
            yolo_checkpoint_dir = os.path.join(self.ckpt_dir, "yolo")
            list_of_files = glob.glob(os.path.join(yolo_checkpoint_dir, '*.pt'))
            latest_file = max(list_of_files, key=os.path.getctime)
            self.model = YOLO(latest_file)

            self.detection_model = AutoDetectionModel.from_pretrained(
                model_type='yolov11',
                model=self.model,
                confidence_threshold=self.confidence_threshold,
                device=self.device,
                load_at_init=True,
            )

    
    def sliced_inference(self, image_file: str = None, image_filename: str = None, verbose: bool = True) -> None:
        """
        Perform sliced inference on a single image.

        This function performs inference on an image using a specified AI model. It supports various models including
        Mask R-CNN with different backbones and YOLOv11. The function saves hyperparameters, performs sliced inference,
        evaluates predictions, visualizes predicted bounding boxes and masks, and stores the results.

        Args:
            image_file (str): The path to the input image file.
            image_filename (str): The filename of the input image.
            verbose (bool): Whether to print detailed information during the process.

        Returns:
            instance_result
        """
        # Save hyperparameters to json file
        hyper_parameters_path = os.path.join(self.aux_files_dir, "hyperparameters.json")
        
        
        if self.model_name == "maskrcnn_resnet101_dc5":
            with open(hyper_parameters_path, 'w') as fp:
                json.dump(self.mask_rcnn_resnet101_dc5_hyperparameters, fp, indent=4)
        elif self.model_name == "maskrcnn_resnet50_c4":
            with open(hyper_parameters_path, 'w') as fp:
                json.dump(self.mask_rcnn_resnet50_c4_hyperparameters, fp, indent=4)
        elif self.model_name == "yolov11":
            with open(hyper_parameters_path, 'w') as fp:
                json.dump(self.yolov11_hyperparameters, fp, indent=4)

        
        if self.model_name in ["maskrcnn_resnet50_c4", "maskrcnn_resnet101_dc5"]:
            print(f"Running inference with {self.model_name} model...")
            if self.model_name == "maskrcnn_resnet50_c4":
                hyper_parameters = self.mask_rcnn_resnet50_c4_hyperparameters
            elif self.model_name == "maskrcnn_resnet101_dc5":
                hyper_parameters = self.mask_rcnn_resnet101_dc5_hyperparameters


            # Save hyperparameters to json file
            hyper_parameters_path = os.path.join(self.aux_files_dir, "hyperparameters.json")
            with open(hyper_parameters_path, 'w') as fp:
                json.dump(hyper_parameters, fp, indent=4)
        
            # Perform sliced inference
            self.result = get_sliced_prediction(
                image=image_file,
                detection_model=self.detection_model,
                slice_height=400,
                slice_width=400,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
                verbose=0,
                interim_dir=None,
                #postprocess_type="NMM", #NMS used per patch -> NMM for patch to image reconstruction
                #postprocess_match_metric = "IOU",
                #postprocess_match_threshold = 0.2, # default 0.5, 0.15 delivered good results
                #postprocess_class_agnostic=False,
            )

            # Visualize predicted bounding boxes and masks over the original image
            self.predictions_export_dir = os.path.join(self.image_analysis_dir, "02_model_predictions")
            self.create_folder(self.predictions_export_dir)
            self.result.export_visuals(export_dir=self.predictions_export_dir,
                                    file_name=f"{os.path.splitext(os.path.basename(image_filename))[0]}_prediction",
                                    text_size=0.25, rect_th=2, hide_labels=False, hide_conf=False)
            # Evaluate predictions
            clusters_export_dir = self.clusters_results_dir 
            instance_result = self.evaluate_predictions(image_filename=image_filename, verbose=verbose)
            self.plot_clusters_on_image(export_dir=clusters_export_dir, image_filename=image_filename, save=True)

            self.inspect_evaluation_results(image_filename=image_filename, show_connectivity=True, show_correlations=True)

            #print(f"[sliced_inference()] instance_result: {instance_result}")
            # Store the results dictionary
            if self.results_dict is None:
                self.results_dict = self.result.results_dict
            else:
                self.results_dict.update(self.result.results_dict)
            
            #print(f"[sliced_inference()] self.result.results_dict: {self.result.results_dict}")
            #print(f"[sliced_inference()]self.results_dict: {self.results_dict}")

            # Store color_dict and category_dict in a json file
            
            color_dict = self.result.results_dict["color_dict"]
            category_dict = self.result.results_dict["category_name_dict"]
            color_dict_path = os.path.join(self.aux_files_dir, "color_dict.json")
            category_dict_path = os.path.join(self.aux_files_dir, "category_name_dict.json")

            with open(color_dict_path, 'w') as fp:
                json.dump(color_dict, fp, indent=4)
            with open(category_dict_path, 'w') as fp:
                json.dump(category_dict, fp, indent=4)
        
        print(f"Sliced inference for image '{image_filename}' complete.")
        
        return instance_result
            
    def get_results_dict(self) -> dict:
        """
        Get the results as a dictionary.

        Returns:
            dict: The results dictionary containing image, file_name, output_dir, elapsed_time, color_dict, and category_dict.
        """
        return self.results_dict

    def get_instance_results_df(self) -> pd.DataFrame:
        """
        Get the inference result as a pandas DataFrame.

        Returns:
            pd.DataFrame: The inference results DataFrame.
        """
        return self.instance_results_df

    def get_instance_results_dict(self) -> dict:
        """
        Get the inference result as a dictionary.

        Returns:
            dict: The inference results dictionary.
        """
        return self.instance_results_dict

    def get_pixel_size(self) -> float:
        """
        Get the pixel size from image dimensions (in pixels) and physical dimensions of the microscope (in mm).

        Returns:
            float: The pixel size in millimeters.
        """
        return self.pixel_size

    def calculate_pixel_size(self, image_width: int) -> float:
        """
        Calculate the pixel size from image dimensions (in pixels) and physical dimensions of the microscope (in mm).

        Args:
            image_width (int): The width of the image in pixels.

        Returns:
            float: The calculated pixel size in millimeters.
        """
        if image_width:
            self.pixel_size = self.physical_image_width / image_width  # in mm
        return self.pixel_size

 
    def plot_clusters_on_image(self, export_dir: str = None, image_filename: str = None, alpha_mask: float = 0.5, save: bool = True):
        """
        Plot detected masks and clusters directly on the prediction image.
        Legend is oriented outside the image, top-left.
        """

        if not hasattr(self, "cell_details_df") or len(self.cell_details_df) == 0:
            print("No cell details found. Run evaluate_predictions() first.")
            return

        df_cells = self.cell_details_df.copy()
        object_prediction_list = self.result.object_prediction_list

        # --- Use the original prediction image ---
        if not hasattr(self.result, "image"):
            print("No prediction image found. Can't overlay masks.")
            return

        # Convert PIL Image to NumPy
        img = np.array(self.result.image)
        if img.ndim == 2:  # grayscale -> RGB
            img = np.stack([img]*3, axis=-1)
        img = img.astype(np.float32) / 255.0 if img.max() > 1 else img.astype(np.float32)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img)

        # --- Colors for clusters ---
        # clustered ids are those with cluster_id != 0 (isolated cells are cluster_id == 0)
        clustered_ids = df_cells[df_cells["cluster_id"] != 0]["cluster_id"].unique()
        rng = np.random.default_rng(42)
        self.cluster_colors = {int(cluster_id): rng.random(3) for cluster_id in clustered_ids}
        isolated_color = np.array([0.6, 0.6, 0.6])
        self.cluster_colors["isolated"] = isolated_color

        # --- Overlay masks ---
        for _, row in df_cells.iterrows():
            mask_i = object_prediction_list[int(row["mask_id"])].mask.bool_mask
            color = self.cluster_colors["isolated"] if int(row["cluster_id"]) == 0 else self.cluster_colors[int(row["cluster_id"])]
            for c in range(3):
                img[..., c] = np.where(mask_i, alpha_mask*color[c] + (1-alpha_mask)*img[..., c], img[..., c])

        ax.imshow(img)

        # --- Centroids & cluster labels ---
        for _, row in df_cells.iterrows():
            cx, cy = row["centroid_x"], row["centroid_y"]
            if np.isnan(cx) or np.isnan(cy):
                continue
            cluster_id = int(row["cluster_id"]) if not pd.isna(row["cluster_id"]) else 0
            color = self.cluster_colors["isolated"] if cluster_id == 0 else self.cluster_colors[cluster_id]
            ax.scatter(cx, cy, s=20, color=color, edgecolors='black', linewidth=0.5)
            #if not row["is_isolated"]:
            #    ax.text(cx, cy, str(cluster_id), color='white', fontsize=8, ha='center', va='center')

        ax.set_title(f"Detected Nuclei clusters — {image_filename}", fontsize=14)
        ax.axis("off")

        # --- Legend (outside, with isolated cluster first) ---
        patches = []
        # Put isolated cluster (cluster_id == 0) first in the legend if present
        num_isolated = len(df_cells[df_cells["cluster_id"] == 0])
        if num_isolated > 0:
            # Show as "Cluster 0 (Isolated) (N cells)"
            patches.append(mpatches.Patch(color=self.cluster_colors["isolated"], label=f"Cluster 0 (Isolated) ({num_isolated} cells)"))

        # Then add the detected clusters (ids > 0) in sorted order
        clustered_ids_sorted = sorted([int(x) for x in clustered_ids])
        for cluster_id in clustered_ids_sorted:
            num_cells = len(df_cells[df_cells["cluster_id"] == cluster_id])
            label = f"Cluster {cluster_id} ({num_cells} cells)"
            patches.append(mpatches.Patch(color=self.cluster_colors[cluster_id], label=label))

        # Adjust font size based on number of legend entries to prevent overflow
        num_entries = len(patches)
        legend_fontsize = min(8, max(6, 12 - num_entries * 0.5))  # Decrease font size as entries increase

        # Place legend to the right of the plot
        ax.legend(handles=patches, loc='center left', bbox_to_anchor=(1.02, 0.5), 
                 fontsize=legend_fontsize, frameon=True)

        plt.tight_layout()

        # --- Optional save ---
        if save and export_dir and image_filename:
            out_file = f"{os.path.splitext(os.path.basename(image_filename))[0]}_clusters.png"
            #output_path = os.path.join(export_dir, out_file)
            output_path = os.path.join(self.clusters_results_dir, out_file)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved overlay plot to: {output_path}")

    def evaluate_predictions(self, image_filename: str = None, verbose: bool = True) -> None:
        """Evaluate model predictions for a single image and compute per-image,
        per-class, per-cluster and per-object (cell) metrics.

        Summary
        -------
        This method analyses the instance segmentation predictions stored in
        ``self.result.object_prediction_list`` and computes the following:
        - Image-level metrics (image area in mm² and µm²).
        - Per-class aggregates (count, total area, relative area, densities).
        - Per-object (cell) descriptors (area, centroid, confidence and a set of
          geometrical, intensity and spatial features).
        - Cluster detection based on mask overlap and adaptive centroid proximity
          (produces cluster ids and cluster summaries including cluster area as
          percent of the image).

        Parameters
        ----------
        image_filename : str, optional
            The filename of the input image (used for output names), by default None
        verbose : bool, optional
            Whether to print detailed information during processing, by default False

        Outputs / Side effects
        -----------------------
        - Returns ``instance_results`` (a dictionary of per-class aggregates).
        - Sets the following attributes on ``self``:
            - ``instance_results_df`` : pandas.DataFrame with per-class summary
            - ``cell_details_df`` : pandas.DataFrame with one row per detected object
            - ``clusters_summary_df`` : pandas.DataFrame summarising detected clusters
        - Optionally writes CSV/XLSX files to configured output folders.

        Per-object (cell) fields computed
        ---------------------------------
        Each row in ``cell_details_df`` contains (name: unit / meaning):
        - mask_id: (int) index of the prediction in ``object_prediction_list``.
        - image_filename: (str) source image filename.
        - class_id, class_name: predicted class label.
        - area_px: (px) mask area in pixels (pycocotools area of RLE).
        - area_mm2: (mm²) = area_px * pixel_size².
        - area_um2: (µm²) = area_mm2 * 1e6.
        - area_relative: (%%) percent of full image area occupied by this cell.
        - centroid_x, centroid_y: (px) centroid coordinates (mean of mask pixels).
        - confidence: (float) model score for the object.
        - perimeter_px: (px) contour perimeter (OpenCV arcLength summed over contours).
        - perimeter_mm: (mm) = perimeter_px * pixel_size.
        - convex_area_px: (px) area of the convex hull of the mask (cv2.contourArea).
        - solidity: (unitless) = area_px / convex_area_px (measure of concavity).
        - major_axis_px, minor_axis_px: (px) fitted ellipse axes (if available, NaN otherwise).
        - eccentricity: (unitless) elliptical eccentricity (0 = circle, ->1 elongated).
        - mean_intensity_r/g/b: mean RGB channel intensities inside the mask (NaN if
          original image not available or size mismatch).
        - nn_distance_px, nn_distance_mm: distance to nearest neighbouring centroid.
        - local_density: local cell density estimated via KD-tree using k neighbours.
        - cluster_id: integer cluster assignment (0 = isolated cell, >0 cluster index).
        - num_neighbours: number of neighbors inside the assigned cluster.
        - is_isolated: boolean = (cluster_id == 0).

        Cluster detection algorithm
        ---------------------------
        - Creates a boolean adjacency matrix between detected objects using two
          criteria:
            1. Relative mask overlap (overlap area / smaller object area) above
               ``overlap_threshold_rel`` (default 2%).
            2. Centroid distance below an adaptive threshold that depends on the
               sum of object radii, a buffer scaled by local median radius, and
               a density/confidence adjustment.
        - A graph is built from adjacency and connected components define clusters.
        - Isolated objects are assigned ``cluster_id = 0``; multi-object clusters
          get sequential ids starting at 1.
        - ``clusters_summary_df`` contains for each cluster: cluster_id, num_cells,
          total_area_um2, area_pct_of_image and mean_confidence.

        Notes and edge-cases
        --------------------
        - All area/length conversions use ``self.pixel_size`` (mm / pixel). If
          pixel size is missing or zero, mm/µm conversions may be invalid.
        - Intensity measures require ``self.result.image`` to be available and
          the same size as the masks; otherwise mean intensities are NaN.
        - Ellipse fitting requires >=5 contour points; when insufficient points
          are present, axis lengths and eccentricity are set to NaN.
        - Convex-area-based solidity uses a safe division (0 when convex area is 0).
        - Single-cell images: nearest-neighbour distance is NaN.
        - Small or noisy masks may produce spurious small areas; consider a
          minimum area filter upstream if desired.

        Performance
        -----------
        - Texture / intensity computations are O(n_pixels_in_masks) and the KD-tree
          queries are O(n log n). If many objects are present, consider batching
          or computing expensive features on request only.
        """

        print(f"Evaluating predictions for image '{image_filename}'...")
        object_prediction_list = self.result.object_prediction_list
        start_time = time.time()

        pixel_size = self.get_pixel_size()  # mm per pixel
        image_area_mm2 = self.result.image_width * self.result.image_height * pixel_size**2
        image_area_um2 = image_area_mm2 * 1e6
        total_area_mm2 = 0

        # --- Initialize per-class summary ---
        instance_results = {
            id: {
                'filename': image_filename,
                'image_area_mm2': image_area_mm2,
                'image_area_um2': image_area_um2,
                'class_name': self.instances_dict[id],
                'count': 0,
                'area_mm2': 0,
                'area_um2': 0,
                'area_relative': 0,
                'density_per_mm2': 0,
                'density_per_um2': 0
            }
            for id in self.instances_dict.keys()
        }

        # --- Store individual object results ---
        individual_cells = []

        for idx, object_prediction in enumerate(object_prediction_list):
            object_prediction = object_prediction.deepcopy()
            if object_prediction.bbox is not None and object_prediction.mask is not None:
                if object_prediction.score.value >= self.confidence_threshold:
                    cat_id = object_prediction.category.id

                    bool_mask = object_prediction.mask.bool_mask
                    rle = mask.encode(np.asfortranarray(bool_mask))
                    area_px = mask.area(rle)
                    area_mm2 = area_px * pixel_size**2
                    area_um2 = area_mm2 * 1e6
                    total_area_mm2 += area_mm2
                    area_relative = (area_mm2 / image_area_mm2) * 100  # percent of full image area

                    # --- Centroid coordinates ---
                    ys, xs = np.nonzero(bool_mask)
                    cx, cy = (int(xs.mean()), int(ys.mean())) if len(xs) > 0 else (np.nan, np.nan)

                    # Update class totals
                    instance_results[cat_id]['count'] += 1
                    instance_results[cat_id]['area_mm2'] += area_mm2

                    # Store individual cell info
                    individual_cells.append({
                        "mask_id": idx,
                        "image_filename": image_filename,
                        "class_id": cat_id,
                        "class_name": self.instances_dict[cat_id],
                        "area_px": area_px,
                        "area_mm2": area_mm2,
                        "area_um2": area_um2,
                        "area_relative": area_relative, # percent of full image area
                        "centroid_x": cx,
                        "centroid_y": cy,
                        "confidence": np.round(object_prediction.score.value, 3), 
                    })
        
        #check if area_relative is not bigger than 100%
        sum_of_area_relative = sum([cell['area_relative'] for cell in individual_cells])
        if sum_of_area_relative > 100:
            print(f"Warning: Sum of area_relative for all cells exceeds 100% ({sum_of_area_relative}%). Check pixel size and image dimensions.")

        # ---- Aggregate statistics per class ----
        total_area_mm2 = round(total_area_mm2, 6)
        total_area_um2 = total_area_mm2 * 1e6

        for id in self.instances_dict.keys():
            count = instance_results[id]['count']
            area_mm = round(instance_results[id]['area_mm2'], 8)
            area_um = area_mm * 1e6
            rel_area = (area_mm / image_area_mm2) * 100  # percent of full image area
            density_mm2 = round((count / image_area_mm2), 1)
            density_um2 = density_mm2 / 1e6

            instance_results[id].update({
                'area_mm2': area_mm,
                'area_um2': area_um,
                'area_relative': rel_area,
                'density_per_mm2': density_mm2,
                'density_per_um2': density_um2
            })

        df_summary = pd.DataFrame.from_dict(instance_results, orient='index')
        df_cells = pd.DataFrame(individual_cells)




        # --- Detect overlaps / conglomerates ---
        if len(df_cells) > 0:
            num_cells = len(df_cells)
            # Use relative overlap threshold (% of smaller cell's area)
            overlap_threshold_rel = 0.02  # 2% overlap threshold
            adjacency = np.zeros((num_cells, num_cells), dtype=bool)
            
            # Preload masks and compute additional features
            all_masks = [object_prediction_list[i].mask.bool_mask for i in range(num_cells)]
            centroids = df_cells[["centroid_x", "centroid_y"]].to_numpy()
            
            # Calculate cell features
            radii = np.sqrt(df_cells["area_px"].to_numpy() / np.pi)
            median_radius = np.median(radii)
            areas = df_cells["area_px"].to_numpy()
            confidences = df_cells["confidence"].to_numpy()
            
            # Adaptive distance thresholds based on local cell properties
            buffer_scale = 3 #0.5 4 # Scale factor for buffer
            adaptive_buffer = buffer_scale * median_radius  
            
            # Calculate local cell density using KD-tree
            tree = KDTree(centroids)
            k = min(20, num_cells)  # Use up to 20 nearest neighbors
            local_densities = []
            
            for i in range(num_cells):
                distances, _ = tree.query(centroids[i:i+1], k=k) # search for k nearest neighbors
                # consider the highest distance (to the k-th neighbor) as radius
                density = k / (np.pi * distances[0][-1]**2) if distances[0][-1] > 0 else float('inf')
                local_densities.append(density)
            
            local_densities = np.array(local_densities)
            median_density = np.median(local_densities)
            df_cells["local_density"] = local_densities

            # --- Additional per-cell features: perimeter, solidity, eccentricity, axis lengths, mean intensities, NN distance ---
        
            try:
                # get image array for intensity measurements
                img = np.array(self.result.image)
                if img.ndim == 2:  # grayscale -> RGB-like
                    img = np.stack([img] * 3, axis=-1)
                # ensure image is HxWxC
                img_h, img_w = img.shape[:2]
            except Exception:
                img = None

            perimeters = []
            convex_areas = []
            solidities = []
            major_axes = []
            minor_axes = []
            eccentricities = []
            mean_R = []
            mean_G = []
            mean_B = []

            # compute nearest-neighbour distances (px)
            if num_cells > 1:
                dists_all, _ = tree.query(centroids, k=2)
                # second column is nearest other point
                nn_distances = dists_all[:, 1]
            else:
                nn_distances = np.array([np.nan] * num_cells)

            for i in range(num_cells):
                try:
                    mask_i = all_masks[i].astype(np.uint8)
                    # find contours
                    contours, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
   
                    if len(contours) > 0:
                        # combine all contour points for shape fits
                        all_pts = np.vstack(contours)
                        # perimeter (sum of contour perimeters)
                        perim = sum([cv2.arcLength(c, True) for c in contours])
                        # convex hull area
                        try:
                            hull = cv2.convexHull(all_pts)
                            convex_area_px = cv2.contourArea(hull)
                        except Exception:
                            convex_area_px = 0

                        # ellipse fit for major/minor axis
                        if all_pts.shape[0] >= 5:
                            try:
                                ellipse = cv2.fitEllipse(all_pts)
                                (_, _), (MA, ma), _ = ellipse
                                major = max(MA, ma)
                                minor = min(MA, ma)
                                ecc = np.sqrt(max(0.0, 1.0 - (minor / major) ** 2)) if major > 0 else np.nan
                            except Exception:
                                major, minor, ecc = np.nan, np.nan, np.nan
                        else:
                            major, minor, ecc = np.nan, np.nan, np.nan
                    else:
                        perim, convex_area_px, major, minor, ecc = 0.0, 0.0, np.nan, np.nan, np.nan

                    # solidity
                    area_px = df_cells.iloc[i]["area_px"] if "area_px" in df_cells.columns else 0
                    solidity = (area_px / convex_area_px) if convex_area_px and convex_area_px > 0 else 0.0

                    # mean RGB intensities inside mask 
                    if img is not None and mask_i.shape[0] == img_h and mask_i.shape[1] == img_w:
                        try:
                            masked_pixels = img[mask_i.astype(bool)]
                            if masked_pixels.size > 0:
                                mR = float(np.mean(masked_pixels[:, 0]))
                                mG = float(np.mean(masked_pixels[:, 1]))
                                mB = float(np.mean(masked_pixels[:, 2]))
                            else:
                                mR = mG = mB = np.nan
                        except Exception:
                            mR = mG = mB = np.nan
                    else:
                        mR = mG = mB = np.nan

                except Exception:
                    perim, convex_area_px, solidity, major, minor, ecc = 0.0, 0.0, 0.0, np.nan, np.nan, np.nan
                    mR = mG = mB = np.nan

                perimeters.append(perim)
                convex_areas.append(convex_area_px)
                solidities.append(solidity)
                major_axes.append(major)
                minor_axes.append(minor)
                eccentricities.append(ecc)
                mean_R.append(mR)
                mean_G.append(mG)
                mean_B.append(mB)

            # add computed columns to df_cells
            df_cells["perimeter_px"] = perimeters
            df_cells["perimeter_mm"] = df_cells["perimeter_px"] * pixel_size
            df_cells["convex_area_px"] = convex_areas
            df_cells["solidity"] = solidities
            df_cells["major_axis_px"] = major_axes
            df_cells["minor_axis_px"] = minor_axes
            df_cells["eccentricity"] = eccentricities
            df_cells["mean_intensity_r"] = mean_R
            df_cells["mean_intensity_g"] = mean_G
            df_cells["mean_intensity_b"] = mean_B

            # nearest neighbour distances
            df_cells["nn_distance_px"] = nn_distances
            df_cells["nn_distance_mm"] = df_cells["nn_distance_px"] * pixel_size

            for i in range(num_cells):
                for j in range(i + 1, num_cells):
                    mask_i = all_masks[i]
                    mask_j = all_masks[j]

                    # Check relative mask overlap
                    if mask_i.shape == mask_j.shape:
                        overlap_area = np.sum(np.logical_and(mask_i, mask_j))
                        min_area = min(areas[i], areas[j])
                        overlap_fraction = overlap_area / min_area if min_area > 0 else 0
                    else:
                        overlap_fraction = 0

                    # Check centroid distance with adaptive threshold
                    distance = np.linalg.norm(centroids[i] - centroids[j]) # euclidean distance
                    base_threshold = radii[i] + radii[j]
                    
                    # Adjust threshold based on local density
                    density_factor = np.sqrt(median_density / max(local_densities[i], local_densities[j])) # denser areas -> smaller threshold
                    density_adjusted_buffer = adaptive_buffer * density_factor
                    
                    # Confidence-based adjustment
                    conf_factor = min(confidences[i], confidences[j]) # higher confidence -> larger threshold
                    distance_threshold = base_threshold + density_adjusted_buffer * conf_factor

                    # Merge based on centroid distance, mask overlap, local density and confidence
                    if overlap_fraction > overlap_threshold_rel or distance <= distance_threshold:
                        adjacency[i, j] = adjacency[j, i] = True

            # Build graph and detect clusters
            G = nx.Graph()
            G.add_nodes_from(range(num_cells))
            for i in range(num_cells):
                for j in range(num_cells):
                    if adjacency[i, j]: # if masks overlap or centroids close enough
                        G.add_edge(i, j)

            # NetworkX recognizes connected components: A component is a group of nodes that are directly or indirectly connected via edges.
            # Each component corresponds to a cluster.
            components = list(nx.connected_components(G))
            cell_cluster_info = np.zeros(num_cells, dtype=int)
            cell_neighbor_count = np.zeros(num_cells, dtype=int)

            # Assign cluster ids such that isolated cells get cluster_id = 0
            # and multi-cell clusters get sequential ids starting at 1
            next_cluster_id = 1
            for comp in components:
                if len(comp) == 1:
                    # isolated -> cluster id 0
                    for node in comp:
                        cell_cluster_info[node] = 0
                        cell_neighbor_count[node] = 0
                else:
                    # conglomerate -> assign next available id
                    for node in comp:
                        cell_cluster_info[node] = next_cluster_id
                        cell_neighbor_count[node] = len(comp) - 1
                    next_cluster_id += 1

            df_cells["cluster_id"] = cell_cluster_info
            df_cells["num_neighbours"] = cell_neighbor_count
            # is_isolated is now defined by cluster_id == 0 for clarity
            df_cells["is_isolated"] = (df_cells["cluster_id"] == 0)

            # Optional conglomerate summary: only include clusters with id > 0
            conglomerate_summary = []
            unique_cluster_ids = np.unique(cell_cluster_info)
            for cluster_id in unique_cluster_ids:
                #if cluster_id == 0:
                #    continue
                comp_indices = np.where(cell_cluster_info == cluster_id)[0].tolist()
                if len(comp_indices) > 0:
                    total_area = df_cells.loc[comp_indices, "area_um2"].sum()
                    avg_conf = df_cells.loc[comp_indices, "confidence"].mean()
                    # percentage of image area taken by this cluster (use image_area_um2 computed earlier)
                    if image_area_um2 and image_area_um2 > 0:
                        area_pct = round((total_area / image_area_um2) * 100, 4)
                    else:
                        area_pct = 0.0

                    conglomerate_summary.append({
                        "cluster_id": int(cluster_id),
                        "num_cells": len(comp_indices),
                        "total_area_um2": total_area,
                        "area_pct_of_image": area_pct,
                        "mean_confidence": round(avg_conf, 4)
                    })
            df_clusters = pd.DataFrame(conglomerate_summary)
        else:
            df_clusters = pd.DataFrame()

        # --- Store internally ---
        self.instance_results_df = df_summary
        self.cell_details_df = df_cells
        self.clusters_summary_df = df_clusters

        # --- Optional export ---
        output_dir = getattr(self, "csv_results_dir", None)
        if output_dir and image_filename:
            df_summary.to_csv(os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_filename))[0]}_summary.csv"), index=False)
            df_cells.to_csv(os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_filename))[0]}_instances_info.csv"), index=False)
            if not df_clusters.empty:
                df_clusters.to_csv(os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_filename))[0]}_clusters_summary.csv"), index=False)
        output_dir = getattr(self, "excel_results_dir", None)
        if output_dir and image_filename:
            cell_excel_name = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_filename))[0]}_instances_info.xlsx")
            self.write_to_excel(outpath=cell_excel_name, df=df_cells, sheetname="Detection Results")
            if not df_clusters.empty:
                cluster_excel_name = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_filename))[0]}_clusters_summary.xlsx")
                self.write_to_excel(outpath=cluster_excel_name, df=df_clusters, sheetname="Clusters Summary")
                
            
        # --- Verbose output ---

        elapsed_time = time.time() - start_time
        if verbose:
            num_clusters = sum(1 for comp in components if len(comp) > 1)
            num_isolated = sum(1 for comp in components if len(comp) == 1)
            print(f"\nEvaluation of predictions performed in {elapsed_time:.2f}s")
            print(f"Detected {num_clusters} clusterss with mask or centroid proximity")
            print(f"Detected {num_isolated} isolated cells")
            print(f"Saved results for {len(individual_cells)} objects.")

        return instance_results
    


    def inspect_evaluation_results(self,
                                   image_filename=None,
                                show_contours=True,
                                show_clusters=True,
                                show_metrics=True,
                                show_connectivity=False,
                                show_correlations=False,
                                max_objects=500):
        """
        Visualize and inspect the results computed by evaluate_predictions().

        Parameters
        ----------
        show_contours : bool
            Overlay contours of detected objects on the input image.
        show_clusters : bool
            Scatter plot of centroids colored by cluster ID.
        show_metrics : bool
            Show histograms/scatterplots of key geometric metrics.
        show_connectivity : bool
            Show graph of cluster connectivity (networkx view).

        max_objects : int
            Maximum number of objects to plot (to avoid overcrowding).
            
        Notes:

            Solidity: Measures how filled or compact a shape is — i.e., how close it is to being convex.
                        A perfectly solid shape (circle, ellipse, convex blob) → solidity ≈ 1.0
                        A shape with indentations or irregular boundaries (cluster, crescent, ring) → solidity < 1.0.
                        Helps identify overlapping cells or irregularly shaped clusters; those often have significantly lower solidity than isolated cells.

            Eccentricity: Ratio of the major to minor axis of the best-fit ellipse:
                        Describes how elongated a shape is.
                        A circle → eccentricity ≈ 0
                        A stretched ellipse or rod-like shape → eccentricity → 1. 
                        
                        Detects elongated or deformed cells, or merged objects that appear stretched.
                        
            Nearest-neighbour distance (vs. Area): The distance (in pixels) from each cell’s centroid to the nearest other centroid. Plotted against cell area (in µm²).
                        Points in the lower-left → small cells close together (dense packing).
                        Points in the upper-right → large, isolated objects. 
                        
                        Reveals spatial density and interaction patterns:
                            Low distances suggest close packing or aggregation.
                            High distances for large areas can indicate merged masks or isolated artifacts.

            Major vs. Minor Axis Length: major_axis_px → length of the longest diameter of the fitted ellipse.
                        major_axis_px → length of the longest diameter of the fitted ellipse.
                        minor_axis_px → shortest diameter.
                        Quickly checks aspect ratio consistency — clusters or stretched cells show distinct distributions.
                            Circular cells → points along the diagonal (major ≈ minor)
                            Elongated or irregular cells → deviate above or below that diagonal.
         
        """

        if not hasattr(self, "cell_details_df"):
            print("⚠️ No evaluated results found. Run evaluate_predictions() first.")
            return

        df_cells = self.cell_details_df.copy()
        df_clusters = getattr(self, "clusters_summary_df", pd.DataFrame())
        img = np.array(getattr(self.result, "image", None))

        print(f"\n🔍 Inspecting {len(df_cells)} detected objects and {len(df_clusters)} clusters...")

        # limit for performance
        if len(df_cells) > max_objects:
            print(f"Limiting visualization to first {max_objects} objects for clarity.")
            df_cells = df_cells.head(max_objects)

 
        # --- 1. Metric distributions ---
        if show_metrics:
            print("📈 Showing shape and spacing metrics...")
            fig, axes = plt.subplots(2, 2, figsize=(12,10))
            sns.histplot(df_cells["solidity"], bins=40, ax=axes[0,0], color="steelblue")
            axes[0,0].set_title("Solidity distribution")

            sns.histplot(df_cells["eccentricity"], bins=40, ax=axes[0,1], color="coral")
            axes[0,1].set_title("Eccentricity distribution")

            sns.scatterplot(x="area_um2", y="nn_distance_px", data=df_cells,
                            hue="cluster_id", palette="tab10", ax=axes[1,0], s=25)
            axes[1,0].set_title("Nearest-neighbour distance vs Area") # 

            sns.scatterplot(x="major_axis_px", y="minor_axis_px", data=df_cells,
                            ax=axes[1,1], color="gray", s=20)
            axes[1,1].set_title("Major vs Minor axis length")
            plt.tight_layout()
            image_file = f"{os.path.splitext(os.path.basename(image_filename))[0]}_metrics_distributions.png"
            plt.savefig(os.path.join(self.clusters_results_dir, image_file), dpi=300, bbox_inches='tight')
            print(f"Saved metrics distributions plot to: {os.path.join(self.clusters_results_dir, image_file)}")
            #plt.show()

        # --- 2. Connectivity graph overlaid on image ---
        if show_connectivity and "cluster_id" in df_cells.columns:
            print("🕸️ Showing cluster connectivity graph over image...")
            
            # prepare figure
            fig, ax = plt.subplots(figsize=(10,10))
            if img is not None:
                ax.imshow(img)

            # draw edges between centroids within same cluster
            for cluster_id in df_cells["cluster_id"].unique():
                if cluster_id <= 0:
                    continue
                members = df_cells[df_cells["cluster_id"] == cluster_id]
                centroids = members[["centroid_x", "centroid_y"]].values

                # draw edges between every pair in the cluster
                for i in range(len(centroids)):
                    for j in range(i+1, len(centroids)):
                        x0, y0 = centroids[i]
                        x1, y1 = centroids[j]
                        color = self.cluster_colors.get(cluster_id, (0,1,0))
                        ax.plot([x0, x1], [y0, y1], color=color, alpha=0.3, linewidth=1.2)  

            # Create a consistent color mapping
            unique_clusters = df_cells["cluster_id"].unique()
            color_dict = {}
            for cluster_id in unique_clusters:
                if cluster_id == 0:
                    color_dict[cluster_id] = self.cluster_colors.get("isolated", (0.5,0.5,0.5))
                else:
                    color_dict[cluster_id] = self.cluster_colors.get(cluster_id, (0,1,0))
            
            # loop over the masks and plot them first (so they're behind the centroids)
            for idx, row in df_cells.iterrows():
                mask_id = row["mask_id"]
                cluster_id = row["cluster_id"]
                color = color_dict[cluster_id]
                
                object_prediction = self.result.object_prediction_list[mask_id]
                bool_mask = object_prediction.mask.bool_mask
                
                # plot the mask as alpha overlay
                alpha = 0.1 # higher alpha for higher visibility
                ys, xs = np.nonzero(bool_mask)
                ax.scatter(xs, ys, color=color, alpha=alpha)
                
                # draw contour
                if show_contours:
                    contours, _ = cv2.findContours(bool_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    for contour in contours:
                        contour = contour.squeeze()
                        if contour.ndim == 1:
                            continue
                        ax.plot(contour[:,0], contour[:,1], color=color, linewidth=1)
            
            # draw the centroids on top with the same colors
            for cluster_id in unique_clusters:
                cluster_cells = df_cells[df_cells["cluster_id"] == cluster_id]
                ax.scatter(cluster_cells["centroid_x"], cluster_cells["centroid_y"],
                          color=color_dict[cluster_id], s=30, 
                          edgecolor="black", linewidth=0.3,
                          label=f"Cluster {cluster_id}" if cluster_id > 0 else "Isolated")
                

            plt.tight_layout()
            

            # save figure
            image_file = f"{os.path.splitext(os.path.basename(image_filename))[0]}_cluster_connectivity_overlay.png"
            plt.savefig(os.path.join(self.clusters_results_dir, image_file), dpi=300, bbox_inches='tight')
            print(f"Saved overlaid connectivity graph to: {os.path.join(self.clusters_results_dir, image_file)}")
            plt.close(fig)

        print("✅ Inspection complete.")




    def write_json_results(self, results_dict: dict, out_path: str) -> None:
        """
        Write results to a JSON file.

        Args:
            results_dict (dict): The results dictionary to be written to the JSON file.
            out_path (str): The output directory path where the JSON file will be saved.
        Returns:
            None
        """
        with open(out_path, 'w') as fp:
            json.dump(results_dict, fp, indent=4, cls=NumpyEncoder)

    def write_csv_results(self, df: pd.DataFrame, outpath: str) -> None:
        """
        Write results to a CSV file.

        Args:
            df (pd.DataFrame): The DataFrame containing the results to be written to the CSV file.

        Returns:
            None
        """
        df.to_csv(outpath)

    def write_to_excel(self, outpath: str= f"instances_results.xlsx", df: pd.DataFrame = None, sheetname: str = "Sheet1") -> None:
        """
        Write the given DataFrame to an Excel file with formatting.

        This function writes the provided DataFrame to an Excel file, applies formatting to the header cells,
        and adjusts the column widths based on the content. It also creates a table with a specified style.

        Args:
            df (pd.DataFrame): The DataFrame containing the results to be written to the Excel file.

        Returns:
            None
        """
        dfs = {f'{sheetname}': df}  # can send a dict of dataframes to to_excel as well
        #filename_xlsx = f"instances_results.xlsx"
        #outpath = os.path.join(self.output_dir, filename_xlsx)
        writer = pd.ExcelWriter(outpath, engine='xlsxwriter')
        for sheetname, df in dfs.items():  # loop through `dict` of dataframes
            df.to_excel(writer, sheet_name=sheetname, index=False)  # send df to writer

            worksheet = writer.sheets[sheetname]  # pull worksheet object
            # workbook is an instance of the whole book - used i.e. for cell format assignment 
            workbook = writer.book

            header_cell_format = workbook.add_format()
            header_cell_format.set_align('center')
            header_cell_format.set_align('vcenter')

            # create list of dicts for header names 
            #  (columns property accepts {'header': value} as header name)
            col_names = [{'header': col_name} for col_name in df.columns]

            # add table with coordinates: first row, first col, last row, last col; 
            #  header names or formatting can be inserted into dict 
            worksheet.add_table(0, 0, df.shape[0], df.shape[1]-1, {
                'columns': col_names,
                # 'style' = option Format as table value and is case sensitive 
                # (look at the exact name into Excel)
                'style': 'Table Style Medium 10'
            })

            # determine max col width 
            for idx, col in enumerate(df):  # loop through all columns
                series = df[col]
                max_len = max((
                    series.astype(str).map(len).max(),  # len of largest item
                    len(str(series.name))  # len of column name/header
                    )) + 3  # adding a little extra space
                worksheet.set_column(idx, idx, max_len)  # set column width

            # set header format
                
            # skip the loop completly if AutoFit for header is not needed
            for i, col in enumerate(col_names):
                # apply header_cell_format to cell on [row:0, column:i] and write text value from col_names in
                worksheet.write(0, i, col['header'], header_cell_format)
                
        writer.close()
        
    def save_coco_predictions(self, image_file: str = None) -> None:
        """
        Save COCO format predictions to a JSON file.

        This function converts the model predictions to COCO annotations and saves them to a JSON file.

        Args:
            image_file (str): The path to the input image file.

        Returns:
            None
        """
        # convert predictions to COCO annotations
        coco_list = self.result.to_coco_annotations(image_id=0)
        # save coco annotations to json file
        predictions_coco_export_dir = os.path.join(self.output_dir, "coco_results")
        self.create_folder(predictions_coco_export_dir)
        
        coco_json_path = os.path.join(predictions_coco_export_dir, f"{os.path.splitext(os.path.basename(image_file))[0]}_coco.json")
        with open(coco_json_path, 'w') as f:
            json.dump(coco_list, f)

    def get_coco_predictions_for_image(self, image_file: str = None, image_id: int = 0) -> list:
        """
        Get COCO format predictions for a single image.

        This function converts the model predictions to COCO annotations and adds an annotation ID to each annotation.

        Args:
            image_file (str): The path to the input image file.
            image_id (int): The ID of the image.

        Returns:
            list: A list of COCO annotations.
        """
        # convert predictions to COCO annotations
        coco_list = self.result.to_coco_annotations(image_id=image_id)
        # add annotation id to each annotation
        for i, coco_dict in enumerate(coco_list):
            coco_dict["id"] = i + 1

        return coco_list

    def save_binary_masks(self, image_file: str = None) -> None:
        """
        Save binary masks to disk.

        This function saves the binary masks of the predicted objects to disk in PNG format.

        Args:
            image_file (str): The path to the input image file.

        Returns:
            None
        """
        # save binary masks to disk
        image_basename = os.path.basename(image_file)
        binary_masks_export_dir = self.segmentation_results_dir
        mask_dir = os.path.join(binary_masks_export_dir, image_basename)
        self.create_folder(mask_dir)
        self.result.store_binary_masks(export_dir=mask_dir, file_name="cell", format="png")
        
    def run_inference(self, image_files: list, trial_dir: str, language: str = "eng", results_dict: dict = None, 
                      analysis_thread_should_stop: bool = None) -> Generator[tuple, None, None]:
        """
        Perform inference on a list of images, yielding process and status updates.

        Args:
            image_files (list): List of paths to the image files to be processed.
            trial_dir (str): Directory where trial-related files are stored.
            language (str): Language for status messages ("eng" for English, "de" for German). Default is "eng".
            results_dict (dict): Dictionary to store the results. Default is None.
            analysis_thread_should_stop (bool): Flag to indicate if the analysis thread should stop. Default is None.

        Yields:
            tuple: A tuple containing the image number, process value, status message, and results dictionary and eta at each iteration.
        """
        self.results_dict = results_dict
        if not hasattr(self, 'model'):
            self.initialize_model()
        #else:
            #print("Model already initialized.")
    
        inference_time = time.time()
    
        num_classes = len(self.instances_dict.keys())
        num_images = len(image_files)   
        date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        class_names = [self.instances_dict[id] for id in self.instances_dict.keys()]
        trial_name = os.path.basename(trial_dir)
        if self.results_dict is None:
            results_dict = {
                "trial": trial_name,
                "date": date,
                "num_images": num_images,
                "num_classes": num_classes, 
                "class_names": class_names,  # convert set to list
                "image_data": {os.path.basename(image_file): {} for image_file in image_files} # create empty dictionary for each image
            }
        else:
            results_dict = self.results_dict.copy() # copy the results dictionary, all keys and values

            if "class_names" not in results_dict.keys():
                results_dict["class_names"] = class_names
            if "trial" not in results_dict.keys():
                results_dict["trial"] = trial_name
            if "date" not in results_dict.keys():
                results_dict["date"] = date
            if "num_images" not in results_dict.keys():
                results_dict["num_images"] = num_images
            if "num_classes" not in results_dict.keys():
                results_dict["num_classes"] = num_classes
            
            if "image_data" not in results_dict.keys():
                results_dict["image_data"] = {os.path.basename(image_file): {} for image_file in image_files}
            else:
                for image_file in image_files:
                    if os.path.basename(image_file) not in results_dict["image_data"].keys():
                        results_dict["image_data"][os.path.basename(image_file)] = {}
        

        # write the results dictionary to a json file
        filename = "image_analysis_results.json"
        out_path = os.path.join(self.aux_files_dir, filename)
        self.write_json_results(results_dict, out_path)

        # get initial image dimensions from json file --> original image size
        #image_dims_json = os.path.join(trial_dir, "aux_files", "images_dims.json")
        image_dims_json = os.path.join(self.aux_files_dir, "images_dims.json")
        with open(image_dims_json, 'r') as f:
            image_dimensions = json.load(f)

        # Start the timer
        #start_time = time.time()
        ETA = 0
        if language == "eng":
            ETA_text = f"Calculating ETA ..."
        elif language == "de":
            ETA_text = "Berechne die verbleibende Zeit ..."

        # create empty coco file
        if self.store_coco_files:
            print("Creating COCO file...")
            #coco_export_dir = os.path.join(self.output_dir, "coco")
            coco_export_dir = self.coco_results_dir
            self.create_folder(coco_export_dir)
            coco_json_path = os.path.join(coco_export_dir, "coco_segmentation_results.json")
            coco_template = {
                "info": {},
                "licenses": [],
                "images": [],
                "annotations": [],
                "categories": []
            }
            coco_template["info"] = {
                "description": "COCO dataset for instance segmentation",
                "url": "http://cocodataset.org",
                "version": "1.0",
                "year": 2024,
                "contributor": "AI Model",
                "date_created": date
            }
            coco_template["licenses"] = [
                {
                    "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
                    "id": 1,
                    "name": "Attribution-NonCommercial-ShareAlike License"
                }
            ]
 
            categories = []
            for i, class_name in enumerate(class_names):
                category = {
                    "supercategory": "",
                    "id": i + 1,
                    "name": class_name,
                }
                categories.append(category)
            coco_template["categories"] = categories
            
            with open(coco_json_path, 'w') as f:
                json.dump(coco_template, f, indent=4)

        # Loop through all images
        mean_time_per_image = 0
        for image_num, image_file in enumerate(image_files):
            
            if self.settings["language"] == "eng":
                status = f"Processing image {image_num + 1} of {len(image_files)}: {os.path.basename(image_file)}"
            elif self.settings["language"] == "de":
                status = f"Analysiere Bild {image_num + 1} von {len(image_files)}: {os.path.basename(image_file)}"
                
            progress = int((image_num) / len(image_files) * 100)
            
            yield (image_num, progress, status, results_dict, None)

            start_time_image = time.time()

            if analysis_thread_should_stop == True:
                print("Analysis stopped.")
                optimize_VRAM_usage()
                    # clear unused variables
                if hasattr(self, 'detection_model'):
                    del self.detection_model
                if hasattr(self, 'model'):
                    del self.model
                if hasattr(self, 'detection_module_predictor'):
                    del self.detection_module_predictor
                if hasattr(self, 'detection_module_cfg'):
                    del self.detection_module_cfg
                if hasattr(self, 'result'):
                    del self.result
                break

            filename = os.path.basename(image_file)


            image_dims = image_dimensions[filename]
            # get the image dimensions
            if len(image_dims["initial"]) == 2 and len(image_dims["final"]) == 2:
                initial_image_width, initial_image_height = image_dims["initial"]
                final_image_width, final_image_height = image_dims["final"]
            elif len(image_dims["initial"]) == 3 and len(image_dims["final"]) == 3:
                initial_image_width, initial_image_height, _ = image_dims["initial"]
                final_image_width, final_image_height, _ = image_dims["final"]
            else:
                print("Invalid image dimensions.")
                initial_image_width, initial_image_height = 0, 0
                final_image_width, final_image_height = 0, 0

            # if the final image dimensions differ from the initial image dimensions,
            # the image has been resized and the pixel size has to be recalculated
            if initial_image_width != final_image_width or initial_image_height != final_image_height:
                pixel_size = self.calculate_pixel_size(final_image_width)
            else:
                pixel_size = self.calculate_pixel_size(initial_image_width)

            image = read_image(image_file)
            
            # perform sliced inference per image
            with torch.no_grad():
                instances_results_dict = self.sliced_inference(image_file=image, image_filename=filename, verbose=False)
            
            # get the instance results
            instances_results_df = self.get_instance_results_df()
           
            
            # ensure keys exist before updating
            if "image_data" not in results_dict:
                results_dict["image_data"] = {}

            if filename not in results_dict["image_data"]:
                results_dict["image_data"][filename] = {}

            # now safely update
            results_dict["image_data"][filename].update(instances_results_dict)
            
            filename = "image_analysis_results.json"
            out_path = os.path.join(self.aux_files_dir, filename)
            self.write_json_results(results_dict, out_path)

            # store the dataframe to csv
            if image_num == 0:
                final_df = instances_results_df
            else:
                # concatenate all results to a single dataframe
                final_df = pd.concat([final_df, instances_results_df], axis=0)

            if self.store_csv_file:
                # save results to csv
                outpath = os.path.join(self.csv_results_dir, "00_image_analysis_results.csv")
                self.write_csv_results(final_df, outpath)

            # save results to xlsx
            if self.store_xlsx_file:
                #outpath = os.path.join(self.output_dir, "instances_results.xlsx")
                #self.write_to_excel(outpath, final_df)
                excel_output_dir = self.excel_results_dir
                self.create_folder(excel_output_dir)
                outfile = f"00_image_analysis_results.xlsx"
                outpath =  os.path.join(excel_output_dir, outfile)
                self.write_to_excel(outpath, final_df, sheetname="Image Analysis Results")
            
            # get the prediction in COCO format for the current image
            coco_predictions = self.get_coco_predictions_for_image(image_file=image, image_id=image_num+1)
            # update the coco json file
            if self.store_coco_files:
                if not os.path.exists(coco_json_path):
                    os.makedirs(coco_json_path)
                    print("Creating COCO file...")
                    
                    with open(coco_json_path, 'w') as f:
                        json.dump(coco_template, f, indent=4)
                
                with open(coco_json_path, 'r') as f:
                    coco_json = json.load(f)

                images_dict_for_coco = {"id": image_num+1, "file_name": filename, "height": final_image_height, "width": final_image_width}
                coco_json["images"].append(images_dict_for_coco)

                coco_json["annotations"].extend(coco_predictions)
            
                # sanitise the json coco file
                coco_json = sanitise_coco_data(coco_json)

                with open(coco_json_path, 'w') as f:
                    json.dump(coco_json, f, indent=4)
                    
                process_value = int((image_num) / len(image_files) * 100)
                
                # stop the timer
                end_time_image = time.time()
                time_taken_image = end_time_image - start_time_image
                if image_num == 0:
                    mean_time_per_image = time_taken_image
                else:
                    mean_time_per_image = (mean_time_per_image * image_num + time_taken_image) / (image_num + 1)
                # calculate the remaining images
                remaining_images = num_images - image_num
                # calculate the remaining time
                ETA = mean_time_per_image / (image_num + 1) * remaining_images
                ETA_minutes = int(ETA // 60)
                ETA_seconds = int(ETA % 60)
                ETA_text = f"{ETA_minutes} min {ETA_seconds} sec"
                
                if language == "eng":
                    #status = f"Processing image {image_num + 1} of {len(image_files)}: {filename} \n\n ETA (current trial): {ETA_text}"
                    status = f"Processing image {image_num + 1} of {len(image_files)}: {filename}"
                elif language == "de":
                    #status = f"Analysiere Bild {image_num + 1} von {len(image_files)}: {filename} \n\n Verbleibende Zeit (aktueller Versuch): {ETA_text}"
                    status = f"Analysiere Bild {image_num + 1} von {len(image_files)}: {filename}"

                yield (image_num, process_value, status, results_dict, time_taken_image) 

            # save binary masks
            if self.store_binary_masks:
                self.save_binary_masks(image_file=image_file)
            
            #process_value = int((image_num + 1) / len(image_files) * 100)
            
            #time_take_for_all_images = time.time() - inference_time # in seconds

            #if language == "eng":
                #status = f"Processed image {image_num + 1} of {len(image_files)}: {filename} \n ETA (per trial): {ETA_text}"
            #elif language == "de":
                #status = f"Bild {image_num + 1} von {len(image_files)} verarbeitet: {filename} \n Verbleibende Zeit (pro Versuch): {ETA_text}"

            # yield process and status updates
            #yield (image_num, process_value, status, results_dict, time_take_for_all_images)

            del image, self.result, instances_results_df, instances_results_dict
            torch.cuda.empty_cache()
            print(f"Image {filename} processed.")

        inference_time = time.time() - inference_time
        print(f"Total inference time: {round(inference_time, 2)} s")
    
        # free GPU cache after completion
        optimize_VRAM_usage()

        # clear all unused variables
        if hasattr(self, 'detection_model'):
            del self.detection_model
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'detection_module_predictor'):
            del self.detection_module_predictor
        if hasattr(self, 'detection_module_cfg'):
            del self.detection_module_cfg
        if hasattr(self, 'result'):
            del self.result
        if hasattr(self, 'instance_results_df'):
            del self.instance_results_df
        if hasattr(self, 'instance_results_dict'):
            del self.instance_results_dict
        if hasattr(self, 'results_dict'):
            del self.results_dict
        if hasattr(self, 'pixel_size'):
            del self.pixel_size


def optimize_VRAM_usage() -> None:
    """
    Optimize VRAM usage by clearing the cache.

    This function performs garbage collection and clears the CUDA cache to free up VRAM.
    It also prints the current VRAM usage after optimization.

    Returns:
        None
    """
    #print("Optimizing GPU usage...")
    # clear cache
    gc.collect()
    torch.cuda.empty_cache()
    #print("GPU memory usage after optimization:")
    #print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")