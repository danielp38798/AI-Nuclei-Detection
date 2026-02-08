import os
import cv2
import numpy as np
import json
from cellpose import models, plot
from PIL import Image
import matplotlib.pyplot as plt

def run_cellpose_on_tiffs(input_dir, output_dir):
    # Create output directories
    predictions_dir = os.path.join(output_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    coco_output_path = os.path.join(predictions_dir, "instances_default.json")

    # Find all TIFF files
    tiff_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.tif', '.tiff'))]

    # Initialize Cellpose model
    model = models.CellposeModel(gpu=True, model_type='nuclei')

    coco_annotations = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "nucleus", "supercategory": "object"}],
    }
    annotation_id = 1

    for idx, tiff_file in enumerate(tiff_files):
        tiff_path = os.path.join(input_dir, tiff_file)
        img = cv2.imread(tiff_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Failed to read {tiff_file}")
            continue

        # Run Cellpose prediction
        masks, flows, styles = model.eval(img, diameter=30, channels=[0,0])

        # Save prediction image
        pred_img_path = os.path.join(predictions_dir, f"pred_{tiff_file}")
        #plot.save_masks(img, masks, flows, pred_img_path)
        # create a figure to save
        fig = plt.figure(figsize=(12, 3))
        plot.show_segmentation(fig, img, masks, flows[0], channels=[0,0], file_name=pred_img_path)

        # Add image info to COCO
        image_id = idx + 1
        coco_annotations["images"].append({
            "id": image_id,
            "file_name": tiff_file,
            "width": img.shape[1],
            "height": img.shape[0]
        })

        # Extract contours and add to COCO annotations
        for mask_id in np.unique(masks):
            if mask_id == 0:
                continue
            mask = (masks == mask_id).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                segmentation = contour.flatten().tolist()
                if len(segmentation) < 6:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                coco_annotations["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "segmentation": [segmentation],
                    "bbox": [x, y, w, h],
                    "area": area,
                    "iscrowd": 0
                })
                annotation_id += 1

        # Save COCO annotations
        with open(coco_output_path, "w") as f:
            json.dump(coco_annotations, f)
        print(f"COCO annotations saved to {coco_output_path}")
        print(f"Prediction images saved to {predictions_dir}")

if __name__ == "__main__":
    input_dir = r"C:\Users\pod38798\Documents\Coding\Python\CELL_DETECTION_TRAINDATA\new_balanced_data\val2017"  # Change to your TIFF folder
    output_dir = os.path.join(input_dir, "cellose_predictions")              # Change to your desired output folder
    run_cellpose_on_tiffs(input_dir, output_dir)