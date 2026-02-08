import pandas as pd
import os
import logging
import matplotlib.pyplot as plt
from AutoStatistics import AutoStatistics
from SeaPlot_v3 import SeaPlot

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compare_columns(csv_path, output_dir, column1, column2, x_label, y_label, max_y_val=100, save_plot=True):
    logging.info(f"Comparing: {column1} vs {column2}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    logging.info(f"Loaded data from {csv_path}")

    for col in [column1, column2]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the dataset.")

    df = df[[column1, column2]].dropna()
    column1 = df[column1]
    column2 = df[column2]

    ast_obj = AutoStatistics()
    stat_result_dict = ast_obj.test_significance(data1=column1, data2=column2, paired=False)
    print(stat_result_dict)

    if not pd.api.types.is_numeric_dtype(column1) or not pd.api.types.is_numeric_dtype(column2):
        raise TypeError("Both columns must contain numeric data.")

    plot_obj = SeaPlot()
    plt.figure(figsize=(5, 6))
    fig = plot_obj.totalBoxPlot(data1=column1, data2=column2, title=f"{x_label} vs {y_label}",
                                 xlabel=x_label, ylabel=y_label,
                                 ci=False, ci_alpha=0.95, max_y_val=max_y_val)

    plot_path = None
    if save_plot:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, f"{x_label}_vs_{y_label}.png")
        plt.savefig(plot_path, bbox_inches='tight')

    return {
        "test_used": stat_result_dict["test_used"],
        "plot_path": plot_path
    }


def aggregate_data_old(folders, count_column='Model Prediction Count', area_column='Model Prediction Area Normalized'):
    gt_counts, model_counts = [], []
    gt_areas, model_areas = [], []
    model_names = []

    for folder in folders:
        csv_path = os.path.join(folder, "detection_log.csv")
        if not os.path.exists(csv_path):
            logging.warning(f"CSV not found in {folder}")
            continue
        
        df = pd.read_csv(csv_path)
        logging.info(f"Processing {csv_path}")

        # Get GT and Model values
        try:
            gt_count = df['Ground Truth Annotation Count']
            pred_count = df[count_column]

            gt_area = df['Ground Truth Area Normalized']
            pred_area = df[area_column]
        except KeyError as e:
            logging.error(f"Missing expected column in {csv_path}: {e}")
            continue

        # For first file, keep GT, else just predictions
        if len(gt_counts) == 0:
            gt_counts = gt_count
            gt_areas = gt_area

        model_counts.append(pred_count.reset_index(drop=True))
        model_areas.append(pred_area.reset_index(drop=True))
        #model_names.append(os.path.basename(folder))
        # the the dir name parent dir name
        #model_name = os.path.basename(os.path.dirname(folder))
        import re

        folder_name = os.path.dirname(folder)
        pattern = r"training_output_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}_(.+)"
        match = re.search(pattern, folder_name)

        if match:
            extracted_name = match.group(1)
            print(extracted_name)  # Output: mask_rcnn_R_50_C4_3x
        model_names.append(extracted_name)

    # Concatenate into DataFrames
    df_counts = pd.concat([gt_counts.reset_index(drop=True)] + model_counts, axis=1)
    df_areas = pd.concat([gt_areas.reset_index(drop=True)] + model_areas, axis=1)

    df_counts.columns = ['GT'] + model_names
    df_areas.columns = ['GT'] + model_names
    
    print(df_counts.columns)
    print(df_areas.columns)

    order = [
        "GT",
        "mask_rcnn_R_50_C4_3x", "mask_rcnn_R_50_DC5_3x", "mask_rcnn_R_50_FPN_3x",
        "mask_rcnn_R_101_C4_3x", "mask_rcnn_R_101_DC5_3x", "mask_rcnn_R_101_FPN_3x"
    ]
    # Filter and sort columns based on the order
    df_counts = df_counts[order]
    df_areas = df_areas[order]


    return df_counts, df_areas


def aggregate_data(folders, count_column='Model Prediction Count', area_column='Model Prediction Area Normalized'):
    gt_counts, model_counts = [], []
    gt_areas, model_areas = [], []
    model_names = []
    file_names = None  # To store image file names

    for folder in folders:
        csv_path = os.path.join(folder, "detection_log.csv")
        if not os.path.exists(csv_path):
            logging.warning(f"CSV not found in {folder}")
            continue
        
        df = pd.read_csv(csv_path)
        logging.info(f"Processing {csv_path}")

        # Get GT and Model values
        try:
            gt_count = df['Ground Truth Annotation Count']
            pred_count = df[count_column]

            gt_area = df['Ground Truth Area Normalized']
            pred_area = df[area_column]
            if file_names is None and 'Image' in df.columns:
                file_names = df['Image']
                # only take the file name without path
                file_names = [os.path.basename(x) for x in file_names]
                
        except KeyError as e:
            logging.error(f"Missing expected column in {csv_path}: {e}")
            continue

        # For first file, keep GT, else just predictions
        if len(gt_counts) == 0:
            gt_counts = gt_count
            gt_areas = gt_area

        model_counts.append(pred_count.reset_index(drop=True))
        model_areas.append(pred_area.reset_index(drop=True))
        import re

        folder_name = os.path.dirname(folder)
        pattern = r"training_output_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}_(.+)"
        match = re.search(pattern, folder_name)

        if match:
            extracted_name = match.group(1)
        model_names.append(extracted_name)

    # Concatenate into DataFrames
    df_counts = pd.concat([gt_counts.reset_index(drop=True)] + model_counts, axis=1)
    df_areas = pd.concat([gt_areas.reset_index(drop=True)] + model_areas, axis=1)

    df_counts.columns = ['GT'] + model_names
    df_areas.columns = ['GT'] + model_names

    # Insert file names as first column if available
    if file_names is not None:
        df_counts.insert(0, 'Image', file_names)
        df_areas.insert(0, 'Image', file_names)

    order = [
        "Image",
        "GT",
        "mask_rcnn_R_50_C4", "mask_rcnn_R_50_DC5", "mask_rcnn_R_50_FPN",
        "mask_rcnn_R_101_C4", "mask_rcnn_R_101_DC5", "mask_rcnn_R_101_FPN", 
        "Cellpose", "StarDist"
    ]
    # Filter and sort columns based on the order
    df_counts = df_counts[[col for col in order if col in df_counts.columns]]
    df_areas = df_areas[[col for col in order if col in df_areas.columns]]

    return df_counts, df_areas



import json
import os
import numpy as np
from pycocotools import mask as mask_utils
import pandas as pd


def eval_coco_instances_to_csv(coco_pred_path, csv_out_path):
    """
    Evaluate COCO instance segmentation predictions and save results to CSV.
    Parameters:
    - coco_pred_path: Path to the COCO predictions JSON file.
    - csv_out_path: Path to save the output CSV file.
    The CSV will contain:
    - file_name: Name of the image file.
    - num_gt_instances: Number of ground truth instances.
    - num_detected_instances: Number of detected instances.
    - gt_total_area: Total area of ground truth instances.
    - pred_total_area: Total area of detected instances.
    - gt_area_normalized: Normalized ground truth area (by image area).
    - pred_area_normalized: Normalized predicted area (by image area).
    """
    # Load COCO predictions
    with open(coco_pred_path, "r") as f:
        coco_data = json.load(f)

    images = {img["id"]: img for img in coco_data["images"]}
    annotations = coco_data["annotations"]

    # Group annotations by image_id
    gt_by_image = {}
    pred_by_image = {}

    for ann in annotations:
        img_id = ann["image_id"]
        if ann.get("iscrowd", 0) == 0 and ann.get("category_id", 1) == 1:
            # Assume ground truth if "gt" in annotation or if "score" not present
            if "score" not in ann:
                gt_by_image.setdefault(img_id, []).append(ann)
            else:
                pred_by_image.setdefault(img_id, []).append(ann)

    # Write CSV header
    with open(csv_out_path, "w") as f:
        f.write("file_name,num_gt_instances,num_detected_instances,gt_total_area,pred_total_area,gt_area_normalized,pred_area_normalized\n")
        for img_id, img in images.items():
            file_name = img["file_name"]
            height = img["height"]
            width = img["width"]
            img_area = height * width

            gt_anns = gt_by_image.get(img_id, [])
            pred_anns = pred_by_image.get(img_id, [])

            num_gt_instances = len(gt_anns)
            num_detected_instances = len(pred_anns)

            gt_total_area = 0
            for ann in gt_anns:
                if "segmentation" in ann:
                    rle = mask_utils.frPyObjects(ann["segmentation"], height, width)
                    gt_total_area += mask_utils.area(rle).sum()

            pred_total_area = 0
            for ann in pred_anns:
                if "segmentation" in ann:
                    rle = mask_utils.frPyObjects(ann["segmentation"], height, width)
                    pred_total_area += mask_utils.area(rle).sum()

            gt_area_normalized = gt_total_area / img_area if img_area > 0 else 0
            pred_area_normalized = pred_total_area / img_area if img_area > 0 else 0

            f.write(f"{file_name},{num_gt_instances},{num_detected_instances},{gt_total_area},{pred_total_area},{gt_area_normalized},{pred_area_normalized}\n")
            
def update_csv_with_cellpose_results(coco_pred_path, counts_csv_path, areas_csv_path):
    """
    Update existing CSV files with Cellpose evaluation results.
    This function reads Cellpose COCO predictions, evaluates them,
    and appends the results to existing combined CSV files for counts and areas.
    """
    # Load Cellpose results
    eval_coco_instances_to_csv(coco_pred_path, csv_out_path)
    
    # Paths to existing combined CSVs
    counts_df = pd.read_csv(counts_csv_path)
    areas_df = pd.read_csv(areas_csv_path)
    
    # update DataFrames with Cellpose results
    cellpose_df = pd.read_csv(csv_out_path)
    counts_df["Cellpose"] = cellpose_df["num_gt_instances"]
    areas_df["Cellpose"] = cellpose_df["gt_area_normalized"]
    
    # Save updated CSVs
    counts_df.to_csv(counts_csv_path, index=False)
    areas_df.to_csv(areas_csv_path, index=False)




def main():
    """Main function to aggregate data from Mask R-CNN models and compare with Cellpose and StarDist results."""

    # Cellpose Paths
    root_dir = os.getcwd()
    cellpose_coco_pred_path = os.path.join(root_dir, "model_segm_comparison", "cellpose", "tiles_predictions", "predictions", "instances_default.json")
    cellpose_csv_out_path = os.path.join(root_dir, "model_segm_comparison", "cellpose", "tiles_predictions", "predictions", "cellpose_eval_test.csv")

    # StarDist Paths
    stardist_coco_pred_path = os.path.join(root_dir, "model_segm_comparison", "stardist", "prediction_tiles", "instances_default.json")
    stardist_csv_out_path = os.path.join(root_dir, "model_segm_comparison", "stardist", "prediction_tiles", "stardist_eval_test.csv")
    
    # Mask R-CNN Paths
    base_dir = os.path.join(os.getcwd(), "mask_rcnn_configurations")
    folders = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f)) and "training_output" in f]
    folders = [os.path.join(f, "visualizations" ) for f in folders if os.path.exists(os.path.join(f, "visualizations" ))]
    
    # compare the model predictions of the individual models with the GT (with respect to the GT)
    output_dir = os.path.join(base_dir, "comparison_results")
    os.makedirs(output_dir, exist_ok=True)

    # create SeaPlot instance
    seaplot = SeaPlot(output_dir=output_dir)

    print(f"Found folders: {folders}")
    # sort folder such that first comes R_50 then R_101
    # training_output_2025-04-25-09-06-39_mask_rcnn_R_50_C4
    # training_output_2025-04-25-09-06-39_mask_rcnn_R_50_DC5
    # training_output_2025-04-25-09-06-39_mask_rcnn_R_50_FPN

    # training_output_2025-04-25-09-06-39_mask_rcnn_R_101_C4
    # training_output_2025-04-25-09-06-39_mask_rcnn_R_101_DC5
    # training_output_2025-04-25-09-06-39_mask_rcnn_R_101_FPN

    folders.sort(key=lambda x: (x.split("_")[4], x.split("_")[3]))
    print(f"Sorted folders: {folders}")
    df_counts, df_areas = aggregate_data(folders)
    #df_counts_col_names = [col.replace("_3x", "") for col in df_counts.columns]
    #df_areas_col_names = [col.replace("_3x", "") for col in df_areas.columns]
    df_counts.to_csv("combined_cell_counts_mrcnn.csv", index=False)
    df_areas.to_csv("combined_cell_areas_mrcnn.csv", index=False)


    
    # Paths
    eval_coco_instances_to_csv(cellpose_coco_pred_path, cellpose_csv_out_path)
    eval_coco_instances_to_csv(stardist_coco_pred_path, stardist_csv_out_path)

    # Load Cellpose and StarDist results
    cellpose_df = pd.read_csv(cellpose_csv_out_path)
    stardist_df = pd.read_csv(stardist_csv_out_path)
    cellpose_df["basename"] = cellpose_df["file_name"].apply(os.path.basename)
    stardist_df["basename"] = stardist_df["file_name"].apply(os.path.basename)

    # Load combined CSVs
    counts_df = pd.read_csv("combined_cell_counts_mrcnn.csv")
    areas_df = pd.read_csv("combined_cell_areas_mrcnn.csv")

    # Ensure only basename is used for matching
    counts_df["basename"] = counts_df["Image"].apply(os.path.basename)
    areas_df["basename"] = areas_df["Image"].apply(os.path.basename)

    # Match and fill Cellpose and StarDist columns by basename
    counts_df["Cellpose"] = counts_df["basename"].map(
        cellpose_df.set_index("basename")["num_gt_instances"]
    )
    areas_df["Cellpose"] = areas_df["basename"].map(
        cellpose_df.set_index("basename")["gt_area_normalized"]
    )
    counts_df["StarDist"] = counts_df["basename"].map(
        stardist_df.set_index("basename")["num_gt_instances"]
    )
    areas_df["StarDist"] = areas_df["basename"].map(
        stardist_df.set_index("basename")["gt_area_normalized"]
    )

    # Drop helper column before saving
    counts_df.drop(columns=["basename"], inplace=True)
    areas_df.drop(columns=["basename"], inplace=True)

    # Save updated CSVs
    out_counts=os.path.join(output_dir, "combined_cell_counts_mrcnn_cellpose_stardist.csv")
    out_areas=os.path.join(output_dir, "combined_cell_areas_mrcnn_cellpose_stardist.csv")
    counts_df.to_csv(out_counts, index=False)
    areas_df.to_csv(out_areas, index=False)


    # Drop 'Image' column for plotting
    df_counts_plot = counts_df.drop(columns=["Image"], errors='ignore')
    df_areas_plot = areas_df.drop(columns=["Image"], errors='ignore')

    print(f"Counts: {counts_df.columns}")
    print(f"Areas: {areas_df.columns}")

    # Ensure numeric dtype for plotting
    df_counts_plot = df_counts_plot.apply(pd.to_numeric, errors="coerce")
    df_areas_plot = df_areas_plot.apply(pd.to_numeric, errors="coerce")

    # Extract column names from the plot dataframes (excluding 'Image')
    df_counts_col_names_plot = list(df_counts_plot.columns)
    df_areas_col_names_plot = list(df_areas_plot.columns)
    print(f"Counts columns for plotting: {df_counts_col_names_plot}")
    print(f"Areas columns for plotting: {df_areas_col_names_plot}")

    seaplot.compare_detection_models(data=df_areas_plot, 
                                    xticks_labels=df_areas_col_names_plot,
                                    xlabel="Model", ylabel="Nuclei Relative Area", 
                                    figsize=(8,6.5), step=0.1,
                                    lower_limit=-0.025,                              
                                    n_pos=0.78,
                                    x_bar_pos=0.74,
                                    md_pos=0.7,
                                    max_y_val=0.35,
                                    y_bar_lim=0.39,
                                    secondary_xaxis_location=2,
                                    output_name="cell_area_comparison_new.pdf"
                                    )
    seaplot.compare_detection_models(data=df_counts_plot,
                                    xticks_labels=df_counts_col_names_plot,
                                    figsize=(8,6.5), 
                                    step=5,
                                    lower_limit=-1,
                                    n_pos=78, #125
                                    x_bar_pos=74,
                                    md_pos=70,
                                    max_y_val=35,
                                    y_bar_lim=39,
                                    secondary_xaxis_location=2,
                                    xlabel="Model", ylabel="Nuclei Count", 
                                    output_name="cell_count_comparison_new.pdf")
    
if __name__ == "__main__":
    main()