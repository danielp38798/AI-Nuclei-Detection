import os
from pycocotools.coco import COCO

import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


##########################################################
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_dirs = [r'disk/AI_nuclei_detection/GUI/cmu-serif']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
print(font_files)

for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

# set font
plt.rcParams['font.family'] = 'CMU Serif'
plt.rcParams['font.size'] = 12

def load_coco_data(json_file):
    return COCO(json_file)

def get_annotation_counts_per_category(coco_data):
    """
    Get the number of annotations per category for the dataset
    :param coco_data: COCO object
    :return: dictionary with category names as keys and a list of annotation counts as values
    """
    categories = coco_data.loadCats(coco_data.getCatIds())
    category_names = [category['name'] for category in categories]
    annotations_per_image_per_category = {category_name: [] for category_name in category_names}
    # for every category, get the number of annotations per image for training and validation dataset
    for category_name in category_names:
        for category_id in coco_data.getCatIds(catNms=[category_name]):
            image_ids = coco_data.getImgIds(catIds=[category_id])
            for image_id in image_ids:
                annotation_ids = coco_data.getAnnIds(imgIds=image_id, catIds=[category_id])
                annotations = coco_data.loadAnns(annotation_ids)
                annotations_per_image_per_category[category_name].append(len(annotations))
    return annotations_per_image_per_category


def get_mean_annotations_per_image_per_category(coco_data):
    """
    Get the mean number of annotations per image for every category in the dataset
    :param coco_data: COCO object
    :return: dictionary with category names as keys and the mean number of annotations per image as values
    """
    categories = coco_data.loadCats(coco_data.getCatIds())
    category_names = [category['name'] for category in categories]
    mean_annotations_per_image_per_category = {category_name: [] for category_name in category_names}
    mean_annotations_per_image = 0
    # for every category, get the mean number of annotations per image for training and validation dataset
    for category_name in category_names:
        for category_id in coco_data.getCatIds(catNms=[category_name]):
            image_ids = coco_data.getImgIds(catIds=[category_id])
            for image_id in image_ids:
                annotation_ids = coco_data.getAnnIds(imgIds=image_id, catIds=[category_id])
                annotations = coco_data.loadAnns(annotation_ids)
                mean_annotations_per_image_per_category[category_name].append(len(annotations))
                mean_annotations_per_image += len(annotations)
    mean_annotations_per_image /= len(coco_data.dataset['images'])
    # calculate the mean number of annotations per image for every category
    for category_name in mean_annotations_per_image_per_category.keys():
        # mean = sum of all annotations per image for a category / number of images with annotations for that category
        mean_annotations_per_image_per_category[category_name] = sum(mean_annotations_per_image_per_category[category_name]) / len(mean_annotations_per_image_per_category[category_name])
    return mean_annotations_per_image_per_category, mean_annotations_per_image


    
def analyse_dataset(train_json_path, val_json_path, analysis_path):
    """
    This function creates a histogram of the number of annotations per image for every category in the dataset.
    :param json_path: path to the COCO-Format dataset
    :param stage: stage of the dataset (e.g. training, validation, test)
    :return: None
    """
    colors = {
    "decondensed nucleus": [255,112,31],
    "nucleus": [44,153,168]
    }
   
    if not os.path.exists(analysis_path):
        os.makedirs(analysis_path)

    # Load COCO format dataset
    train_coco = COCO(train_json_path)
    val_coco = COCO(val_json_path)

    # Get all categories
    train_categories = train_coco.loadCats(train_coco.getCatIds())
    val_categories = val_coco.loadCats(val_coco.getCatIds())
    category_names = [category['name'] for category in train_categories]
          
    # For every image in the dataset, get the number of annotations per category
    train_annotations_over_all_images = get_annotation_counts_per_category(train_coco)
    val_annotations_over_all_images = get_annotation_counts_per_category(val_coco)
    print("----------------- AMOUNT OF ANNOTATIONS (TOTAL) -----------------")
    print(f"Train: {len(train_coco.dataset['annotations'])} annotations")
    print(f"Val: {len(val_coco.dataset['annotations'])} annotations\n")

    print("----------------- AMOUNT OF ANNOTATIONS PER CATEGORY -----------------")
    for category_name in category_names:
        train_anno = sum(train_annotations_over_all_images[category_name])
        val_anno = sum(val_annotations_over_all_images[category_name])
        print(f"{category_name}: Train: {train_anno} annotations (percentage: {train_anno/len(train_coco.dataset['annotations'])*100:.2f}%)")
        print(f"{category_name}: Val: {val_anno} annotations (percentage: {val_anno/len(val_coco.dataset['annotations'])*100:.2f}%)\n")

    print("----------------- MEAN ANNOTATIONS PER IMAGE PER CATEGORY -----------------")
    mean_annotations_per_image_per_category_train, mean_annotations_per_image_train = get_mean_annotations_per_image_per_category(train_coco)
    mean_annotations_per_image_per_category_val, mean_annotations_per_image_val = get_mean_annotations_per_image_per_category(val_coco)
    print(f"Mean Annotations per Image (Train): {mean_annotations_per_image_train:.2f}")
    print(f"Mean Annotations per Image (Val): {mean_annotations_per_image_val:.2f}\n")

    print("----------------- MEAN ANNOTATIONS PER IMAGE PER CATEGORY -----------------")
    for category_name in category_names:
        print(f"{category_name}: Train: {mean_annotations_per_image_per_category_train[category_name]:.2f} annotations per image")
        print(f"{category_name}: Val: {mean_annotations_per_image_per_category_val[category_name]:.2f} annotations per image\n")


    # Create a boxplot of the number of annotations per image for every category in the dataset
    df = pd.DataFrame()
    for category_name in train_annotations_over_all_images.keys():
        df = pd.concat([df, pd.DataFrame({'category': [category_name]*len(train_annotations_over_all_images[category_name]),
                                    'dataset': ['train']*len(train_annotations_over_all_images[category_name]),
                                    'annotation_count': train_annotations_over_all_images[category_name]})])
        df = pd.concat([df, pd.DataFrame({'category': [category_name]*len(val_annotations_over_all_images[category_name]),
                                    'dataset': ['val']*len(val_annotations_over_all_images[category_name]),
                                    'annotation_count': val_annotations_over_all_images[category_name]})])

    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x='category', y='annotation_count', hue='dataset')
    plt.xlabel('Category')
    plt.ylabel('Annotation Count')
    plt.title('Annotation Count per Image per Category (Train vs. Val)')
    plt.xticks(rotation=45, ha='left')
    plt.legend(title='Dataset')
    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(analysis_path, f"boxplot_annotations_train_vs_val.png"))
    plt.savefig(os.path.join(analysis_path, f"boxplot_annotations_train_vs_val.pdf"))
    """
    plotly:
    fig_box = go.Figure()
    for category_name, annotations in train_annotations_over_all_images.items():
        fig_box.add_trace(go.Box(y=annotations, name=f"{category_name} (train)"))
    for category_name, annotations in val_annotations_over_all_images.items():
        fig_box.add_trace(go.Box(y=annotations, name=f"{category_name} (val)"))
    
    fig_box.update_layout(
        title=f"Number of Annotations per Image (Train vs. Val)",
        xaxis_title="Category",
        yaxis_title="Number of Annotations"
    )
    fig_box.write_image(os.path.join(analysis_path, f"boxplot_annotations_per_image_train_vs_val.png"))"""

    # show how many annotations per image are present for every category for train and val dataset in one figure
    """
    plotly:
    fig = go.Figure()
    for category_name, annotations in train_annotations_over_all_images.items():
        fig.add_trace(go.Histogram(x=annotations, name=f"{category_name} (train)", opacity=0.7))
    for category_name, annotations in val_annotations_over_all_images.items():
        fig.add_trace(go.Histogram(x=annotations, name=f"{category_name} (val)", opacity=0.7))

    fig.update_layout(
        title=f"Number of Annotations per Image (Train vs. Val)",
        xaxis_title="Number of Annotations",
        yaxis_title="Count"
    )
    fig.write_image(os.path.join(analysis_path, f"histogram_annotations_per_image_train_vs_val.png"))
    """

    plt.figure(figsize=(12, 8))
    for category_name, annotations in train_annotations_over_all_images.items():
        plt.hist(annotations, bins=20, alpha=0.7, label=f"{category_name} (train)")
    for category_name, annotations in val_annotations_over_all_images.items():
        plt.hist(annotations, bins=20, alpha=0.7, label=f"{category_name} (val)")
    plt.xlabel('Number of Annotations')
    plt.ylabel('Count')
    plt.title('Histogram of Annotation Count per Image per Category (Train vs. Val)')
    plt.legend(title='Category')
    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(analysis_path, f"histogram_annotations_train_vs_val.png"))

    # also generate a dist plot for the number of annotations per image for train and val dataset into one figure
    """
    for category_name, annotations in train_annotations_over_all_images.items():
        fig_dist = ff.create_distplot([annotations], [f"{category_name} (train)"], show_hist=True)
    for category_name, annotations in val_annotations_over_all_images.items():
        fig_dist = ff.create_distplot([annotations], [f"{category_name} (val)"], show_hist=True)
    fig_dist.update_layout(
        title=f"Distribution of Annotations per Image (Train vs. Val)",
        xaxis_title="Number of Annotations",
        yaxis_title="Density"
    )
    fig_dist.write_image(os.path.join(analysis_path, f"distplot_annotations_per_image_train_vs_val.png"))
    """
    plt.figure(figsize=(12, 8))
    for category_name, annotations in train_annotations_over_all_images.items():
        sns.kdeplot(annotations, label=f"{category_name} (train)", fill=True)
    for category_name, annotations in val_annotations_over_all_images.items():
        sns.kdeplot(annotations, label=f"{category_name} (val)", fill=True)
    plt.xlabel('Number of Annotations')
    plt.ylabel('Density')
    plt.title('Distribution of Annotations per Image per Category (Train vs. Val)')
    plt.legend(title='Category')
    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(analysis_path, f"distplot_annotations_per_image_train_vs_val.png"))



if __name__ == "__main__":
    #train_json_file = "/home/pod44433/disk/AI_nuclei_detection/train_data/annotations/instances_default_train.json"
    #val_json_file = "/home/pod44433/disk/AI_nuclei_detection/train_data/annotations/instances_default_val.json"

    #train_json_file = "/home/pod44433/disk/AI_nuclei_detection/train_data/new_balanced_data/annotations/instances_train2017.json"
    #val_json_file = "/home/pod44433/disk/AI_nuclei_detection/train_data/new_balanced_data/annotations/instances_val2017.json"
    #analysis_path = "/home/pod44433/disk/AI_nuclei_detection/train_data/new_balanced_data/analysis"

    train_json_file = "/home/pod44433/disk/AI_nuclei_detection/train_data/new_balanced_data/sliced_coco/annotations/instances_train2017.json"
    val_json_file = "/home/pod44433/disk/AI_nuclei_detection/train_data/new_balanced_data/sliced_coco/annotations/instances_val2017.json"
    analysis_path = "/home/pod44433/disk/AI_nuclei_detection/train_data/new_balanced_data/sliced_coco/analysis"

    analyse_dataset(train_json_file, val_json_file, analysis_path)
    #analyse_dataset_old(train_json_file, "training")
    #analyse_dataset_old(val_json_file, "validation")