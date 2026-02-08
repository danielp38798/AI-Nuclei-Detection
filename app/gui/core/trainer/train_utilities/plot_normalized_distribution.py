import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

import matplotlib.font_manager
fonts = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
font_names = [matplotlib.font_manager.FontProperties(fname=font).get_name() for font in fonts]

font_dirs = [r'disk/AI_nuclei_detection/GUI/cmu-serif']
font_files = matplotlib.font_manager.findSystemFonts(fontpaths=font_dirs)
print(font_files)

for font_file in font_files:
    matplotlib.font_manager.fontManager.addfont(font_file)

# set font
plt.rcParams['font.family'] = 'CMU Serif'
plt.rcParams['font.size'] = 15


def load_coco_annotations(annotation_file):
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)
    return coco_data

def calculate_category_distribution(coco_data):
    category_distribution = {}
    categories = coco_data['categories']
    annotations = coco_data['annotations']

    for category in categories:
        category_distribution[category['id']] = {'name': category['name'], 'count': 0}

    for annotation in annotations:
        category_id = annotation['category_id']
        category_distribution[category_id]['count'] += 1

    # Normalize counts
    total_annotations = sum([category['count'] for category in category_distribution.values()])
    for category_id, category_info in category_distribution.items():
        category_distribution[category_id]['normalized_count'] = category_info['count'] / total_annotations

    return [{**{'id': cat_id}, **info} for cat_id, info in category_distribution.items()]

def prepare_comparison_data(annotation_files, dataset_labels):
    all_data = []
    for file, label in zip(annotation_files, dataset_labels):
        coco_data = load_coco_annotations(file)
        category_distribution = calculate_category_distribution(coco_data)
        for category in category_distribution:
            category['dataset'] = label
            all_data.append(category)
    return pd.DataFrame(all_data)

# Plotting the comparison
def plot_comparison(df, output_dir):
    plt.figure(figsize=(12, 7))
    # set font to cmu serif
    ax = sns.barplot(x='dataset', y='normalized_count', hue='name', data=df)
    plt.title('Normalized Distribution of Annotations by Category across Datasets')
    plt.ylabel('Normalized Distribution')
    plt.xlabel('Dataset')
    plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Annotate bars with actual counts
    #for p in ax.patches:
    #    height = p.get_height()
    #    if height > 0:
    #        ax.annotate(f'{height:.3f}', (p.get_x() + p.get_width() / 2., (0.7*height)),
    #                    ha='center', fontsize=15, color='white', xytext=(0, 5),
    #                    textcoords='offset points')
    ax.bar_label(ax.containers[0], fmt='%.3f', fontsize=15, padding=-20, color='white')
    ax.bar_label(ax.containers[1], fmt='%.3f', fontsize=15, padding=-20, color='white')
    out_file = os.path.join(output_dir, 'normalized_distribution_comparison.png')
    plt.tight_layout()
    plt.savefig(out_file)
    out_file = os.path.join(output_dir, 'normalized_distribution_comparison.pdf')
    plt.savefig(out_file)
    plt.close()


#annotation_file1 = r"disk/AI_nuclei_detection/train_data/balanced_data/annotations/instances_default_train.json"
#annotation_file2 = r"disk/AI_nuclei_detection/train_data/balanced_data/annotations/instances_default_val.json"
#annotation_file1 = r"disk/AI_nuclei_detection/train_data/balanced_data/patches/annotations/instances_train2017.json"
#annotation_file2 = r"disk/AI_nuclei_detection/train_data/balanced_data/patches/annotations/instances_val2017.json"
annotation_file1 = r"/home/pod44433/disk/AI_nuclei_detection/train_data/new_balanced_data/annotations/instances_train2017.json"
annotation_file2 = r"/home/pod44433/disk/AI_nuclei_detection/train_data/new_balanced_data/annotations/instances_val2017.json"
annotation_files = [annotation_file1, annotation_file2]
dataset_labels = ['Train', 'Validation']
df = prepare_comparison_data(annotation_files, dataset_labels)
output_dir = r"/home/pod44433/disk/AI_nuclei_detection/train_data/new_balanced_data/analysis"

# print total number of annotations in each dataset
print(f"Total number of annotations in each dataset: {df.groupby('dataset')['count'].sum()}")
# print normalized distribution of annotations in each dataset
print(f"Normalized distribution of annotations in each dataset: {df.groupby('dataset')['normalized_count'].sum()}")

plot_comparison(df, output_dir)
