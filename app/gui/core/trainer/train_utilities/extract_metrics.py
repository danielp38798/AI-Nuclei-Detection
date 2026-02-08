import json
import matplotlib.pyplot as plt
import textwrap
import os

def extract_metrics(json_data, category_id, iou_type, iou_threshold, verbose=True):
    metrics = {'f1_score': [], 'average_precision': [], 'average_recall': [], 
               'AP': [], 'AP_50': [], 'AP_75': [], 'AP_small': [], 'AP_medium': [], 'AP_large': [], 
               'AR_1': [], 'AR_10': [], 'AR_100': [], 'AR_small': [], 'AR_medium': [], 'AR_large': []}
    if verbose == True:
        print(f"Extracting metrics for category {category_id}, iou type {iou_type}, iou threshold {iou_threshold}")
    for epoch, data in json_data.items():
        if iou_type in data and iou_threshold in data[iou_type]:
            if str(category_id) in data[iou_type][iou_threshold]:
                metrics['f1_score'].append(data[iou_type][iou_threshold][str(category_id)]['f1_score'])
                metrics['average_precision'].append(data[iou_type][iou_threshold][str(category_id)]['average_precision'])
                metrics['average_recall'].append(data[iou_type][iou_threshold][str(category_id)]['average_recall'])
                metrics['AP'].append(data[iou_type][iou_threshold][str(category_id)]['AP']) # IoU 0.5:0.95
                metrics['AP_50'].append(data[iou_type][iou_threshold][str(category_id)]['AP_50']) # IoU 0.5
                metrics['AP_75'].append(data[iou_type][iou_threshold][str(category_id)]['AP_75']) # IoU 0.75
                metrics['AP_small'].append(data[iou_type][iou_threshold][str(category_id)]['AP_small']) # Area small
                metrics['AP_medium'].append(data[iou_type][iou_threshold][str(category_id)]['AP_medium']) # Area medium
                metrics['AP_large'].append(data[iou_type][iou_threshold][str(category_id)]['AP_large']) # Area large
                metrics['AR_1'].append(data[iou_type][iou_threshold][str(category_id)]['AR_1']) # IoU 0.5:0.95
                metrics['AR_10'].append(data[iou_type][iou_threshold][str(category_id)]['AR_10']) # IoU 0.5:0.95    
                metrics['AR_100'].append(data[iou_type][iou_threshold][str(category_id)]['AR_100']) # IoU 0.5:0.95
                metrics['AR_small'].append(data[iou_type][iou_threshold][str(category_id)]['AR_small']) # Area small
                metrics['AR_medium'].append(data[iou_type][iou_threshold][str(category_id)]['AR_medium']) # Area medium
                metrics['AR_large'].append(data[iou_type][iou_threshold][str(category_id)]['AR_large']) # Area large

    return metrics

def plot_f1_ap_ar_metrics_plt(metrics, category_id, iou_type, iou_threshold, file_path=None):
    fig = plt.figure(figsize=(15, 8))
    plt.plot(metrics['f1_score'], label='F1 Score')
    plt.plot(metrics['average_precision'], label='Average Precision')
    plt.plot(metrics['average_recall'], label='Average Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    title = f'F1 Score, Average Precision and Average Recall over Epochs for Category {category_id} - IoU type {iou_type} and IOU Threshold {iou_threshold}'
    plt.title("\n".join(textwrap.wrap(title, 60)))
    plt.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1)) # using a size in points
    plot_file_name = f'F1_AP_AR_metrics_category_{category_id}_{iou_type}_iou_{iou_threshold}.png'
    plot_file_name = os.path.join(file_path, plot_file_name) if file_path else plot_file_name
    fig.savefig(plot_file_name, dpi=300)
    #plt.show()
    plt.close(fig)

def plot_ap_ar_metrics_plt(metrics, category_id, iou_type, iou_threshold, file_path=None):
    fig = plt.figure(figsize=(12, 8))
    plt.plot(metrics['AP'], label='AP')
    plt.plot(metrics['AP_50'], label='AP_50')
    plt.plot(metrics['AP_75'], label='AP_75')
    plt.plot(metrics['AP_small'], label='AP_small')
    plt.plot(metrics['AP_medium'], label='AP_medium')
    plt.plot(metrics['AP_large'], label='AP_large')
    plt.plot(metrics['AR_1'], label='AR_1')
    plt.plot(metrics['AR_10'], label='AR_10')
    plt.plot(metrics['AR_100'], label='AR_100')
    plt.plot(metrics['AR_small'], label='AR_small')
    plt.plot(metrics['AR_medium'], label='AR_medium')
    plt.plot(metrics['AR_large'], label='AR_large')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    #plt.title(f'Average Precision and Average Recall over Epochs for Category {category_id} and IOU Threshold {iou_threshold}')
    title = f"Average Precision and Average Recall over Epochs for Category {category_id} - IoU type '{iou_type}'; IOU Threshold {iou_threshold}"
    plt.title("\n".join(textwrap.wrap(title, 60)))
    plt.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1)) # using a size in points
    plot_file_name = f'AP_AR_metrics_category_{category_id}_{iou_type}_iou_{iou_threshold}.png'
    plot_file_name = os.path.join(file_path, plot_file_name) if file_path else plot_file_name
    fig.savefig(plot_file_name, dpi=300)
    #plt.show()
    plt.close(fig)


def plot_pr_curve(json_data, category_id, iou_type, iou_threshold, file_path=None):
    # for the last epoch extract the precision and recall values
    print(f"Plotting Precision-Recall curve for Category {category_id} - IoU type {iou_type} and IOU Threshold {iou_threshold}")
    # json_data is a dictionary with keys as epoch numbers and values as metrics but entries could be missing for some epochs
    # get the number of the last epoch which has metrics
    for epoch in json_data.keys():
        if iou_type in json_data[epoch] and iou_threshold in json_data[epoch][iou_type] and str(category_id) in json_data[epoch][iou_type][iou_threshold]:
            last_epoch = epoch
    print(f"Last epoch with metrics: {last_epoch}")
    precision = json_data[last_epoch][iou_type][iou_threshold][str(category_id)]['precision_values']
    recall = json_data[last_epoch][iou_type][iou_threshold][str(category_id)]['recall_values']
    
    fig = plt.figure(figsize=(12, 8))
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall curve for Category {category_id} - IoU type {iou_type} and IOU Threshold {iou_threshold}')
    plt.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1)) # using a size in points
    plot_file_name = f'PR_curve_category_{category_id}_{iou_type}_iou_{iou_threshold}.png'
    plot_file_name = os.path.join(file_path, plot_file_name) if file_path else plot_file_name
    fig.savefig(plot_file_name, dpi=300)
    #plt.show()
    plt.close(fig)
    


def process_metrics(json_file, category_ids, iou_types, iou_thresholds, file_path):
    # Load JSON data from file
    with open(json_file) as f:
        json_data = json.load(f)

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    for category_id in category_ids:
        for iou_type in iou_types:
            for iou_threshold in iou_thresholds:
                metrics = extract_metrics(json_data, category_id, iou_type, iou_threshold)
                plot_f1_ap_ar_metrics_plt(metrics, category_id, iou_type, iou_threshold, file_path)
                plot_ap_ar_metrics_plt(metrics, category_id, iou_type, iou_threshold, file_path)
                plot_pr_curve(json_data, category_id, iou_type, iou_threshold, file_path)

# Example usage:
json_file = r"metrics/evaluation_metrics_val.json"
category_ids = [0, 1]
iou_types = ['bbox', 'segm']
iou_thresholds = ['0.5', '0.75']
file_path = os.path.join(os.path.dirname(json_file), 'plots')

process_metrics(json_file, category_ids, iou_types, iou_thresholds, file_path)
