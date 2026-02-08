import os
import cv2
import numpy as np
import json
from stardist.models import StarDist2D
from stardist.plot import render_label
from csbdeep.utils import normalize
from PIL import Image
import matplotlib.pyplot as plt

def run_stardist_on_tiffs(input_dir, output_dir):
    # Create output directories
    predictions_dir = os.path.join(output_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    coco_output_path = os.path.join(predictions_dir, "instances_default.json")

    # Find all TIFF files
    images_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.tif', '.tiff', '.png'))]
    print(f"Found {len(images_files)} image files in {input_dir}")

    # Initialize StarDist2D model
    model = StarDist2D.from_pretrained('2D_versatile_fluo')

    coco_annotations = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "nucleus", "supercategory": "object"}],
    }
    annotation_id = 1

    for idx, image_file in enumerate(images_files):
        image_path = os.path.join(input_dir, image_file)
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Failed to read {image_file}")
            continue

        # Convert to grayscale if needed
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Run StarDist prediction
        labels, _ = model.predict_instances(normalize(img))

        # Save prediction image
        pred_img_path = os.path.join(predictions_dir, f"comparison_{image_file.replace('.tiff', '.png').replace('.tif', '.png')}")
        instance_count = int(np.max(labels))
        fig = plt.figure(figsize=(18, 8))
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.title("input image")
        plt.subplot(1, 2, 2)
        plt.imshow(render_label(labels, img=img))
        plt.axis("off")
        plt.title("prediction + input overlay")
        fig.text(0.5, 0.02, f"Detected instances: {instance_count}", ha="center", va="bottom", fontsize=12)
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.savefig(pred_img_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # store just the image with predicted labels as png
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(render_label(labels, img=img))
        plt.axis("off")
        #plt.title("prediction")
        pred_label_img_path = os.path.join(predictions_dir, f"pred_{image_file.replace('.tiff', '.png').replace('.tif', '.png')}")
        fig.text(0.5, 0.02, f"Detected instances: {instance_count}", ha="center", va="bottom", fontsize=12)
        plt.savefig(pred_label_img_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        # Add image info to COCO
        image_id = idx + 1
        coco_annotations["images"].append({
            "id": image_id,
            "file_name": image_file,
            "width": img.shape[1],
            "height": img.shape[0]
        })

        # Extract contours from labels and add to COCO annotations
        for label_id in np.unique(labels):
            if label_id == 0:
                continue
            mask = (labels == label_id).astype(np.uint8)
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
        print(f"Processed {image_file}")
        
    print(f"COCO annotations saved to {coco_output_path}")
    print(f"Prediction images saved to {predictions_dir}")

if __name__ == "__main__":
    input_dir = r".\fullscale_images" 
    #input_dir = r"R:\10_Labs\BFM\20_Projekte\40_Promotionen\50_AI_CFD_Pointner\10_Work\20_Promotion\99_Sonstiges\30_AI_Nuclei_Detection\02_Code\new_balanced_data\sliced_coco\val2017"
    output_dir = os.path.join(input_dir, "stardist_predictions")              # Change to your desired output folder
    run_stardist_on_tiffs(input_dir, output_dir)