import json
import matplotlib.pyplot as plt
import os
from matplotlib import font_manager
font_dirs = [r'disk/AI_nuclei_detection/GUI/cmu-serif']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
# set font
plt.rcParams['font.family'] = 'CMU Serif'
plt.rcParams['font.size'] = 15
import json

def plot_loss_values(json_files, title, out_dir=None):

    fig = plt.figure(figsize=(10, 10))
    for json_file in json_files:
        print(f"Plotting loss values for {json_file}...")
        with open(json_file) as f:
            data = json.load(f)
        loss_values = [v['loss'] for k, v in data.items() if 'loss' in v]
        
        # extract inner list if loss_values is a list of lists
        if len(loss_values) > 0 and isinstance(loss_values[0], list):
            loss_values = [item for sublist in loss_values for item in sublist]
        keys = [k for k, v in data.items() if 'loss' in v]
        # convert keys to integers
        keys = [int(key) for key in keys]
        # convert loss values to floats
        loss_values = [float(value) for value in loss_values]
        plt.plot(keys, loss_values)

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend(['Training Loss', 'Validation Loss'])
    plt_path = os.path.join(out_dir, 'training_and_validation_losses.png')
    plt.savefig(plt_path)
    plt_path = os.path.join(out_dir, 'training_and_validation_losses.pdf')
    plt.savefig(plt_path)
    plt.close(fig)
    #plt.show()

def plot_accuracy_values(json_files, title, out_dir=None):

    fig = plt.figure(figsize=(10, 10))
    for json_file in json_files:
        print(f"Plotting accuracy values for {json_file}...")
        with open(json_file) as f:
            data = json.load(f)

        for category in [0,1]:
            AP_50_bbox = []
            AP_50_segm = []
            f1_score_bbox = []
            f1_score_segm = []
            epochs = []
            for epoch in data:

                if data[epoch] != {}:
                    for iou_type in data[epoch]:
                        if iou_type == 'bbox':
                            AP_50_bbox.append(data[epoch][iou_type]['0.5'][str(category)]['AP_50'])
                            f1_score_bbox.append(data[epoch][iou_type]['0.5'][str(category)]['f1_score'])
                        elif iou_type == 'segm':
                            AP_50_segm.append(data[epoch][iou_type]['0.5'][str(category)]['AP_50'])
                            f1_score_segm.append(data[epoch][iou_type]['0.5'][str(category)]['f1_score'])
                    epochs.append(int(epoch))

            print(f"AP_50_bbox: {AP_50_bbox}")
            print(f"AP_50_segm: {AP_50_segm}")
            print(f"Epochs: {epochs}")
            if category == 0:
                box_color = 'aqua'
                segm_color = 'darkblue'
                f1_score_bbox_color = 'blue'
                f1_score_segm_color = 'royalblue'
            elif category == 1:
                box_color = 'tomato'
                segm_color = 'darkorange'
                f1_score_bbox_color = 'red'
                f1_score_segm_color = 'darkred'
            
            if category == 0:
                pass
                #plt.plot(epochs, AP_50_bbox, color=box_color, linestyle='dashdot')
                #plt.plot(epochs, AP_50_segm, color=segm_color)
                #plt.plot(epochs, f1_score_bbox, color=f1_score_bbox_color, linestyle='dashdot')
                #plt.plot(epochs, f1_score_segm, color=f1_score_segm_color, linestyle='dotted')
            elif category == 1:
                plt.plot(epochs, AP_50_bbox, color=box_color, linestyle='dashed')
                plt.plot(epochs, AP_50_segm, color=segm_color, linestyle='dotted')
                plt.plot(epochs, f1_score_bbox, color=f1_score_bbox_color, linestyle='dashdot')
                plt.plot(epochs, f1_score_segm, color=f1_score_segm_color, linestyle='dotted')


    plt.xlabel('Epochs')
    plt.xlim(0, max(epochs))

    plt.ylabel('Accuracy')
    plt.title(title)
    #plt.legend(['Training Accuracy (bbox)', 'Training Accuracy (segm)', 'Validation Accuracy (bbox)', 'Validation Accuracy (segm)'])
    if len(json_files) == 1:
        """
        plt.legend(['mAP@IoU 0.5 (bbox) - decondensed nucleus', #'mAP@IoU 0.5 (segm) - decondensed nucleus', 
                    'F1 Score (bbox) - decondensed nucleus', #'F1 Score (segm) - decondensed nucleus', 
                    'mAP@IoU 0.5 (bbox) - nucleus', #'mAP@IoU 0.5 (segm) - nucleus', 
                    'F1 Score (bbox) - nucleus', #'F1 Score (segm) - nucleus'
                    ])
        """
        plt.legend(['mAP@IoU 0.5 (bbox)',
                    'mAP@IoU 0.5 (segm)',
                    'F1 Score (bbox)',
                    'F1 Score (segm)',
                    ])
 
                    
    else:
        plt.legend(['Training Accuracy (bbox)', 'Training Accuracy (segm)', 'Validation Accuracy (bbox)', 'Validation Accuracy (segm)'])
    plt_path = os.path.join(out_dir, 'training_and_validation_accuracies.png')
    plt.savefig(plt_path)
    plt_path = os.path.join(out_dir, 'training_and_validation_accuracies.pdf')
    plt.savefig(plt_path)
    plt.close(fig)

def plot_p_r_curves(json_files, title, out_dir=None):
    
    fig = plt.figure(figsize=(10, 10))
    for json_file in json_files:
        print(f"Plotting precision and recall values for {json_file}...")
        with open(json_file) as f:
            data = json.load(f)
        
         # get the last non empty key in the dictionary
        last_epoch = max([int(k) for k in data.keys() if data[k] != {}])
        print(f"Last epoch: {last_epoch}")
        epoch = str(last_epoch)

        for category in [0,1]:
            bbox_precision = []
            bbox_recall = []
            segm_precision = []
            segm_recall = []
                
            for iou_type in data[epoch]:
                if iou_type == 'bbox':
                    bbox_precision.append(data[epoch][iou_type]['0.5'][str(category)]['precision_values'])
                    bbox_recall.append(data[epoch][iou_type]['0.5'][str(category)]['recall_values'])
                    
                elif iou_type == 'segm':
                    segm_precision.append(data[epoch][iou_type]['0.5'][str(category)]['precision_values'])
                    segm_recall.append(data[epoch][iou_type]['0.5'][str(category)]['recall_values'])

            if category == 0:
                box_color = 'aqua'
                segm_color = 'darkblue'
            elif category == 1:
                box_color = 'tomato'
                segm_color = 'darkorange'
            plt.plot(bbox_precision[0], bbox_recall[0], color=box_color)
            plt.plot(segm_precision[0], segm_recall[0], color=segm_color)

        plt.xlabel('Precision')
        plt.ylabel('Recall')
        plt.title(title)
        categories = ['decondensed nucleus', 'nucleus']
        plt.legend(['Precision-Recall Curve (bbox) - ' + categories[0], 'Precision-Recall Curve (segm) - ' + categories[0], 
                    'Precision-Recall Curve (bbox) - ' + categories[1], 'Precision-Recall Curve (segm) - ' + categories[1]])
        plt_path = os.path.join(out_dir, 'precision_recall_curves_over_all_categories.png')
        plt.tight_layout()
        plt.savefig(plt_path)
        plt_path = os.path.join(out_dir, 'precision_recall_curves_over_all_categories.pdf')
        plt.savefig(plt_path)
        plt.close(fig) 
    


def calculate_delta(loss_values):
    # Calculate differences between consecutive loss values
    deltas = [loss_values[i + 1] - loss_values[i] for i in range(len(loss_values) - 1)]
    # Return the maximum difference as delta
    return max(deltas)

def find_best_delta(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
        
    best_delta = float('-inf')
    loss_values = [v['loss'] for k, v in data.items() if 'loss' in v]
    # extract inner list if loss_values is a list of lists
    if len(loss_values) > 0 and isinstance(loss_values[0], list):
        loss_values = [item for sublist in loss_values for item in sublist]
    # convert loss values to floats
    loss_values = [float(value) for value in loss_values]
    print(f"Loss values: {loss_values}")
    best_delta = calculate_delta(loss_values)
    
    return best_delta



if __name__ == "__main__":


    run_name = "run_20240325-163220"
    metric_logging_dir = f"disk/AI_nuclei_detection/trainer/detection/{run_name}/metrics_logging"
    if not os.path.exists(metric_logging_dir):
        os.makedirs(metric_logging_dir)

    #json_files = []
    #json_files.append(os.path.join(metric_logging_dir, "train", "accuracy", "final_accuracy.json"))
    #json_files.append(os.path.join(metric_logging_dir, "test", "accuracy", "final_accuracy.json"))

    #plot_accuracy_values(json_files, "Training and Validation Accuracy over Epochs", metric_logging_dir)

    losses_json_files = []
    val_loss_json_file = f"disk/AI_nuclei_detection/trainer/detection/{run_name}/metrics/training_losses.json"
    train_loss_json_file = f"disk/AI_nuclei_detection/trainer/detection/{run_name}/metrics/validation_losses.json"
    losses_json_files.append(val_loss_json_file)
    losses_json_files.append(train_loss_json_file)
    plot_loss_values(losses_json_files, "Training and Validation Loss over Epochs", metric_logging_dir)



    json_files = []
    val_accuracy_json_file = f"disk/AI_nuclei_detection/trainer/detection/{run_name}/metrics/evaluation_metrics_val.json"
    #train_accuracy_json_file = f"disk/AI_nuclei_detection/trainer/detection/{run_name}/metrics/evaluation_metrics_train.json"
    json_files.append(val_accuracy_json_file)
    #json_files.append(train_accuracy_json_file)
    plot_accuracy_values(json_files, "mAP@IoU 0.5 and F1 Score over Epochs", metric_logging_dir)

    json_files = []
    val_p_r_json_file = f"disk/AI_nuclei_detection/trainer/detection/{run_name}/metrics/evaluation_metrics_val.json"
    json_files.append(val_p_r_json_file)
    plot_p_r_curves(json_files, "Precision-Recall Curves over all Categories", metric_logging_dir)


