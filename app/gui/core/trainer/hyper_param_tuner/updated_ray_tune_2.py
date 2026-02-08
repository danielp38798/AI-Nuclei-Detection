import torch
import torch.nn as nn
import torch.optim as optim
from ray import train, tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import pyarrow.fs as fs

import os
import zipfile
import torch

from gui.core.trainer.train_utils.engine import *
from gui.core.trainer.train_utils.utils import *
from gui.core.trainer.train_utils.transforms import *
from gui.core.trainer.train_utils.dataset import *

# modified version of torchvision's MaskRCNN which also returns the losses during training not only the detections (roi_heads and generalised_rcnn were modified)
from gui.core.trainer.vision.torchvision.models.detection import MaskRCNN # modified
from gui.core.trainer.vision.torchvision.models.detection.rpn import AnchorGenerator
from gui.core.trainer.vision.torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from gui.core.trainer.vision.torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import matplotlib.pyplot as plt
from gui.core.trainer.vision.torchvision.models import ResNet50_Weights, ResNet101_Weights
from torch.optim.lr_scheduler import ReduceLROnPlateau, LinearLR # StepLR, MultiStepLR, CosineAnnealingLR, CyclicLR, OneCycleLR, ExponentialLR, LambdaLR
from gui.core.trainer.vision.torchvision.models.detection.backbone_utils import resnet_fpn_backbone

import numpy as np
import json
import time
from tqdm import tqdm
from pathlib import Path



class MaskRCNNTrainable(tune.Trainable):
    """
    Trainable class for Mask R-CNN model
    inherits from tune.Trainable allowing for hyperparameter tuning with Ray Tune
    """
    def setup(self, config: dict):
        # config (dict): A dict of hyperparameters
        self.config = config
        self.print_freq = 100
        self.device = None
        self.out_json_paths = None
        self.images_folder = None
        self.train_data_loader = None
        self.test_data_loader = None

        self.root_dir = os.path.join(os.getcwd(), "train_data")
        self.annotations_path = os.path.join(self.root_dir, "annotations", "instances_default.json")
        self.ckpt_dir = os.path.join(os.getcwd(), "ray_tune_checkpoints")
        self.metric_logging_dir = os.path.join(os.getcwd(), self.trial_name, "ray_tune_metric_logging")


        self.create_dirs()

        self.lr = config["learning_rate"]
        self.num_epochs = config["num_epochs"]
        self.batch_size = config["batch_size"]
        self.num_classes = config["num_classes"] + 1 # add background class
        self.momentum = config["momentum"]
        self.weight_decay = config["weight_decay"]
        self.optimizer_name = config["optimizer"]
        self.backbone_name = config["backbone"]
        self.pretrained = config["pretrained"]
        self.trainable_layers = config["trainable_layers"]  
        
        self.model = None
        self.optimizer = None
        self.lr_scheduler_list = []
        self.early_stopping = EarlyStopping(patience=10, verbose=True, path=os.path.join(self.ckpt_dir, "early_stopping_checkpoint.pth"))
        self.perform_slicing = True
        self.do_evaluation_during_training = True
        self.do_model_evaluation = False
        self.eval_freq = 1
        self.train_ratio = 0.8
        self.val_ratio = 0.2
        self.start_epoch = 0
        self.hyperparams = None
        self.final_training_loss_dict = None
        self.final_validation_loss_dict = None
        self.final_accuracy_dict = None
        self.precision_recall_dict = None

    def create_dirs(self):
        """
        Create directories for metric logging and checkpoint saving
        """
        if not os.path.exists(self.metric_logging_dir):
            os.makedirs(self.metric_logging_dir)
            os.makedirs(os.path.join(self.metric_logging_dir, "hyperparameters"))
            os.makedirs(os.path.join(self.metric_logging_dir, "losses"))
            os.makedirs(os.path.join(self.metric_logging_dir, "accuracy"))
            os.makedirs(os.path.join(self.metric_logging_dir, "precision_recall"))
            print("Created metric logging directories.")
        else:
            print("Metric logging directories already exist.")
        
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
            print("Created checkpoint directory.")
        else:
            print("Checkpoint directory already exists.")
 
    def unpack_zip(self, folder_path):
        zip_files = [file for file in os.listdir(folder_path) if file.endswith('.zip')]
        
        if zip_files:
            for zip_file in zip_files:
                zip_path = os.path.join(folder_path, zip_file)
                extraction_path = os.path.splitext(zip_path)[0]
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extraction_path)
                
                os.remove(zip_path)
            print("Data unpacked from zip file.")

    def log_hyperparameters(self, file_path="hyperparameters.json"):
        # Log hyperparameters to TensorBoard
        self.hyperparams = {
            "epochs": self.num_epochs,
            "optimizer": self.optimizer.__class__.__name__,
            "batch_size": self.batch_size,
        }
        if "lr" in self.optimizer.param_groups[0]:
            self.hyperparams["lr"] = self.optimizer.param_groups[0]["lr"]
        if "momentum" in self.optimizer.param_groups[0]: 
            self.hyperparams["momentum"] = self.optimizer.param_groups[0]["momentum"]
        if "weight_decay" in self.optimizer.param_groups[0]:
            self.hyperparams["weight_decay"] = self.optimizer.param_groups[0]["weight_decay"]
        if hasattr(self, "backbone_name"):
            self.hyperparams["backbone"] = self.backbone_name
        if hasattr(self, "num_classes"):
            self.hyperparams["num_classes"] = self.num_classes
        print("HYPERPARAMETERS: ", self.hyperparams)
        # log hyperparameters to json
        with open(file_path, 'w') as f:
            json.dump(self.hyperparams, indent=1, fp=f)
        
    def load_data(self):
        
        # unpack zip file if there is one in data dir
        self.unpack_zip(folder_path=self.root_dir)

        # Create data loaders and perform train, val split
        self.annotations_path = os.path.join(self.root_dir, "annotations", "instances_default.json")
        self.out_json_paths  = split_from_file(self.annotations_path, [self.train_ratio, self.val_ratio], 
                                                names=["train", "val"], do_shuffle=True)
        self.images_folder = os.path.join(self.root_dir, "images")


        #train data set
        train_dataset = create_dataset(images_folder=self.images_folder, train=True, 
                                                annotation_file=self.out_json_paths[0], 
                                                perform_slicing=self.perform_slicing)
        # test data set
        test_dataset  = create_dataset(images_folder=self.images_folder, train=False,
                                                annotation_file=self.out_json_paths[1], 
                                                perform_slicing=self.perform_slicing)
        
        self.train_data_loader = create_dataloader(train_dataset, batch_size=int(self.batch_size), 
                                        shuffle=True, num_workers=8)
        self.test_data_loader   = create_dataloader(test_dataset, batch_size=int(self.batch_size), 
                                        shuffle=True, num_workers=8)
        
        
        print("\n")
        print("|------------ DATA SUMMARY ---------------|")
        print("Data directory: {}".format(self.root_dir))
        print("Checkpoint directory: {}".format(self.args.ckpt_dir))
        if hasattr(self, "out_json_paths"):
            print("Train annotation file: {}".format(self.out_json_paths[0]))
            print("Validation annotation file: {}".format(self.out_json_paths[1]))
        if hasattr(self, "num_classes"):
            print("Number of classes: {}".format(self.num_classes))
        if hasattr(self, "images_folder"):
            print("Images folder: {}".format(self.images_folder))

        print("train/val split: {}/{}".format(self.train_ratio, self.val_ratio))
        print("Number of training images: {}".format(len(self.train_data_loader.dataset)))
        print("Number of validation images: {}".format(len(self.test_data_loader.dataset)))
        print("\n")

    def setup_device(self):
        """
        Setup device (GPU if available, else CPU)
        """
        if self.args.use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("CUDA available. Using GPU: {}".format(torch.cuda.get_device_name(0)))
            torch.cuda.empty_cache()
        else:
            self.device = torch.device("cpu")
            print("CUDA unavailable. Using CPU.")

        
    def setup_lr_scheduler(self):
        """
        Setup learning rate scheduler list for training
        """
        #warmup_factor = 1.0 / 1000
        #warmup_iters = min(1000, len(self.train_data_loader) - 1)
        #self.lr_scheduler_list.append(LinearLR(self.optimizer, start_factor=warmup_factor, total_iters=warmup_iters))
                    
        # StepLR scheduler; reduces the learning rate by a factor of 0.1 every 3 epochs
        # self.lr_scheduler_list.append(torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1, verbose=True))

        # ReduceLROnPlateau scheduler; reduces the learning rate by a factor of 0.1 if the validation loss does not improve for 2 epochs
        self.lr_scheduler_list.append(ReduceLROnPlateau(self.optimizer, factor=0.1,  min_lr=0.00001, 
                                                        mode='min', patience=2, verbose=True))

    def get_model_with_backbone(self, num_classes, backbone_name, pretrained=True, trainable_layers=None, **kwargs):
    
        if backbone_name == "resnet50":

            if pretrained == True:
                backbone  = resnet_fpn_backbone(backbone_name="resnet50", weights=ResNet50_Weights.DEFAULT)
                print("Using pretrained ResNet 50 backbone.")
            elif pretrained == True and trainable_layers != None:
                backbone  = resnet_fpn_backbone(backbone_name="resnet50", 
                                                weights=ResNet50_Weights.DEFAULT, trainable_layers=trainable_layers)
                print("Using pretrained ResNet 50 backbone (training layers {}).".format(trainable_layers))
            else:
                backbone  = resnet_fpn_backbone(backbone_name="resnet50", 
                                                weights=ResNet50_Weights.DEFAULT, trainable_layers=5)
                print("Using ResNet 50 backbone (training all 5 layers).")
        elif backbone_name == "resnet101":
            if pretrained == True:
                backbone  = resnet_fpn_backbone(backbone_name="resnet101", weights=ResNet101_Weights.DEFAULT)
                print("Using pretrained ResNet 101 backbone.")
            elif pretrained == True and trainable_layers != None:
                backbone  = resnet_fpn_backbone(backbone_name="resnet101", 
                                                weights=ResNet101_Weights.DEFAULT, trainable_layers=trainable_layers)
                print("Using pretrained ResNet 101 backbone (training layers {}).".format(trainable_layers))
                
            else:
                backbone  = resnet_fpn_backbone(backbone_name="resnet101", 
                                                weights=ResNet101_Weights.DEFAULT, trainable_layers=5)           
                print("Using ResNet 101 backbone (training all 5 layers).")

 
        # put the pieces together inside a MaskRCNN model
        model = MaskRCNN(backbone=backbone, num_classes=num_classes, 
                         image_mean=[0.485, 0.456, 0.406],
                        image_std=[0.229, 0.224, 0.225])
            
        # Get the number of input features for the classifier
        in_features_box = model.roi_heads.box_predictor.cls_score.in_features
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

        # Get the numbner of output channels for the Mask Predictor
        dim_reduced =  model.roi_heads.mask_predictor.conv5_mask.out_channels

        # Replace the box predictor
        model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_features_box, num_classes=num_classes)

        # Replace the mask predictor
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels=in_features_mask, 
                                                           dim_reduced=dim_reduced, num_classes=num_classes)
  
        return model

    def setup_model(self):
        """
        Initialize Mask R-CNN model, configures backbone, and sets up optimizer and lr scheduler
        """
        # Define the Mask R-CNN model
        self.model = self.get_model_with_backbone(num_classes=self.num_classes, 
                                                  backbone_name=self.backbone_name, 
                                                  pretrained=self.pretrained,
                                                  trainable_layers=self.trainable_layers,
                                                  )

        # Define optimizer and learning rate scheduler
        params = [p for p in self.model.parameters() if p.requires_grad]

        # check if object has lr attribute else use default lr of 0.0001
        if hasattr(self, "optimizer_name"):
            if self.optimizer_name == "Adam":

                if hasattr(self, "lr") and hasattr(self, "weight_decay"):
                    self.optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
                    print("Using Adam optimizer with learning rate: {} and weight decay: {}".format(self.lr, self.weight_decay))

                elif hasattr(self, "lr"):
                    self.optimizer = torch.optim.Adam(params, lr=self.lr)
                    print("Using Adam optimizer with learning rate: {}".format(self.lr))

                else:
                    self.optimizer = torch.optim.Adam(params, lr=0.0001, weight_decay=0.0005)
                    print("Using Adam optimizer with default learning rate: 0.0001 and weight decay: 0.0005")

            elif self.optimizer_name == "AdamW":
                    
                    if hasattr(self, "lr") and hasattr(self, "weight_decay"):
                        self.optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
                        print("Using AdamW optimizer with learning rate: {} and weight decay: {}".format(self.lr, self.weight_decay))
    
                    elif hasattr(self, "lr"):
                        self.optimizer = torch.optim.AdamW(params, lr=self.lr)
                        print("Using AdamW optimizer with learning rate: {}".format(self.lr))

            elif self.args.optimizer == "SGD":

                if hasattr(self, "lr") and hasattr(self, "momentum") and hasattr(self, "weight_decay"):
                    self.optimizer = torch.optim.SGD(params, lr=0.0001, momentum=0.9, weight_decay=self.weight_decay)
                    print("Using SGD optimizer with learning rate: {}, momentum: {} and weight decay: {}".format(self.lr, self.momentum, self.weight_decay))

                elif hasattr(self, "lr") and hasattr(self, "momentum"):
                    self.optimizer = torch.optim.SGD(params, lr=self.lr, momentum=self.momentum)
                    print("Using SGD optimizer with learning rate: {} and momentum: {}".format(self.lr, self.momentum))

                elif hasattr(self, "lr"):
                    self.optimizer = torch.optim.SGD(params, lr=self.lr)
                    print("Using SGD optimizer with learning rate: {}".format(self.lr))

                else:
                    self.optimizer = torch.optim.SGD(params, lr=0.0001, momentum=0.9, weight_decay=0.0005)
                    print("Using SGD optimizer with default learning rate: 0.0001, momentum: 0.9 and weight decay: 0.0005")
            
        else:
            # AdamW optimizer; includes weight decay for regularization
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
            print("Using default AdamW optimizer with learning rate: {}".format(self.lr))

        # Learning rate scheduler
        self.setup_lr_scheduler()

        # send model to device
        self.model.to(self.device)
        
        print("Model initialized.")
        print("\n")
        print("|------------ MODEL SUMMARY ---------------|")
        print("Model name: {}".format(self.model.__class__.__name__))
        print("Backbone: {}".format(self.backbone_name))
        print("Number of classes: {}".format(self.num_classes))
        print("Number of trainable parameters: {}".format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))
        print("Number of epochs: {}".format(self.num_epochs))
        print("Learning rate: {}".format(self.lr))
        print("Batch size: {}".format(self.batch_size))
        print("Device: {}".format(self.device))
        print("Optimizer: {}".format(self.optimizer.__class__.__name__))

    def create_final_training_loss_plot(self):
        
        """
        Create final loss plot, showing the average loss over all epochs

        """
        fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(15, 10))
        # Plot the average loss over all epochs
        # keys are the epochs, values are the average losses for each epoch over all batches
        print("final_training_loss_dict: ", self.final_training_loss_dict)
        if self.final_training_loss_dict:
            # Plot the average loss over all epochs
            # self.final_training_loss_dict = {epoch: {"loss": total_loss, "loss_classifier": total_loss_classifier, "loss_box_reg": total_loss_box_reg, "loss_mask": total_loss_mask, "loss_objectness": total_loss_objectness, "loss_rpn_box_reg": total_loss_rpn_box_reg}}
            if "loss" in self.final_training_loss_dict.keys():
                ax[0, 0].plot(self.final_training_loss_dict.keys(), [v['loss'] for v in self.final_training_loss_dict.values()], label="Total Loss")
                ax[0, 0].set_xlabel("Epoch")
                ax[0, 0].set_ylabel("Loss")
                ax[0, 0].set_title(f"Total Loss")

            if "loss_classifier" in self.final_training_loss_dict.keys():
                ax[0, 1].plot(self.final_training_loss_dict.keys(), [v['loss_classifier'] for v in self.final_training_loss_dict.values()], label="Classifier Loss")
                ax[0, 1].set_xlabel("Epoch")
                ax[0, 1].set_ylabel("Loss")
                ax[0, 1].set_title(f"Classifier Loss")

            if "loss_box_reg" in self.final_training_loss_dict.keys():
                ax[0, 2].plot(self.final_training_loss_dict.keys(), [v['loss_box_reg'] for v in self.final_training_loss_dict.values()], label="Box Regression Loss")
                ax[0, 2].set_xlabel("Epoch")
                ax[0, 2].set_ylabel("Loss")
                ax[0, 2].set_title(f"Box Regression Loss")
            if "loss_mask" in self.final_training_loss_dict.keys():
                ax[1, 0].plot(self.final_training_loss_dict.keys(), [v['loss_mask'] for v in self.final_training_loss_dict.values()], label="Mask Loss")
                ax[1, 0].set_xlabel("Epoch")
                ax[1, 0].set_ylabel("Loss")
                ax[1, 0].set_title(f"Mask Loss")

            if "loss_objectness" in self.final_training_loss_dict.keys():
                ax[1, 1].plot(self.final_training_loss_dict.keys(), [v['loss_objectness'] for v in self.final_training_loss_dict.values()], label="Objectness Loss")
                ax[1, 1].set_xlabel("Epoch")
                ax[1, 1].set_ylabel("Loss")
                ax[1, 1].set_title(f"Objectness Loss")

            if "loss_rpn_box_reg" in self.final_training_loss_dict.keys():
                ax[1, 2].plot(self.final_training_loss_dict.keys(), [v['loss_rpn_box_reg'] for v in self.final_training_loss_dict.values()], label="RPN Box Regression Loss")
                ax[1, 2].set_xlabel("Epoch")
                ax[1, 2].set_ylabel("Loss")
                ax[1, 2].set_title(f"RPN Box Regression Loss")

           # set the title of the plot
            fig.suptitle("Final Training Losses", fontsize=16)

            # Save the plot to a temporary file
            plot_filename = 'final_training_losses.png'
            temp_file_path = os.path.join(self.metric_logging_dir, "losses", plot_filename)
            if not os.path.exists(os.path.dirname(temp_file_path)):
                os.makedirs(os.path.dirname(temp_file_path))
            plt.savefig(temp_file_path)
            plt.close(fig)

    def create_final_validation_loss_plot(self):
            """
            Create final validation loss plot, showing the average loss over all epochs
            """

            fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(15, 10))
            if "loss" in self.final_validation_loss_dict.keys():
                ax[0, 0].plot(self.final_validation_loss_dict.keys(), [v['loss'] for v in self.final_validation_loss_dict.values()], label="Total Loss")
                ax[0, 0].set_title("Total Loss")
                ax[0, 0].set_xlabel("Epoch")
                ax[0, 0].set_ylabel("Loss")
            if "loss_classifier" in self.final_validation_loss_dict.keys():
                ax[0, 1].plot(self.final_validation_loss_dict.keys(), [v['loss_classifier'] for v in self.final_validation_loss_dict.values()], label="Classifier Loss")
                ax[0, 1].set_title("Classifier Loss")
                ax[0, 1].set_xlabel("Epoch")
                ax[0, 1].set_ylabel("Loss")

            if "loss_box_reg" in self.final_validation_loss_dict.keys():
                ax[0, 2].plot(self.final_validation_loss_dict.keys(), [v['loss_box_reg'] for v in self.final_validation_loss_dict.values()], label="Box Regression Loss")
                ax[0, 2].set_title("Box Regression Loss")
                ax[0, 2].set_xlabel("Epoch")
                ax[0, 2].set_ylabel("Loss")

            if "loss_mask" in self.final_validation_loss_dict.keys():
                ax[1, 0].plot(self.final_validation_loss_dict.keys(), [v['loss_mask'] for v in self.final_validation_loss_dict.values()], label="Mask Loss")
                ax[1, 0].set_title("Mask Loss")
                ax[1, 0].set_xlabel("Epoch")
                ax[1, 0].set_ylabel("Loss")

            if "loss_objectness" in self.final_validation_loss_dict.keys():
                ax[1, 1].plot(self.final_validation_loss_dict.keys(), [v['loss_objectness'] for v in self.final_validation_loss_dict.values()], label="Objectness Loss")
                ax[1, 1].set_title("Objectness Loss")
                ax[1, 1].set_xlabel("Epoch")
                ax[1, 1].set_ylabel("Loss")

            if "loss_rpn_box_reg" in self.final_validation_loss_dict.keys():
                ax[1, 2].plot(self.final_validation_loss_dict.keys(), [v['loss_rpn_box_reg'] for v in self.final_validation_loss_dict.values()], label="RPN Box Regression Loss")
                ax[1, 2].set_title("RPN Box Regression Loss")
                ax[1, 2].set_xlabel("Epoch")
                ax[1, 2].set_ylabel("Loss")  
            # set title for the entire plot
            fig.suptitle("Final Validation Losses")
            # Save the plot to a temporary file
            plot_filename = 'final_validation_losses.png'
            temp_file_path = os.path.join(self.metric_logging_dir, "losses", plot_filename)
            if not os.path.exists(os.path.dirname(temp_file_path)):
                os.makedirs(os.path.dirname(temp_file_path))
            plt.savefig(temp_file_path)
            plt.close(fig)

    def log_precision_recall(self, iou_threshold, coco_evaluator, epoch):
        """
        Save precision-recall curves at a specific IoU threshold
        """
        for iou_type, coco_eval in coco_evaluator.coco_eval.items():
            cocoEval = coco_evaluator.coco_eval[iou_type] # bbox, segm, keypoints            
            # Get category IDs
            cocoGt = cocoEval.cocoGt
            category_ids = cocoGt.getCatIds()
            category_ids = [c_id-1 for c_id in category_ids]
            # Extract precision values per category for a specific IoU threshold (e.g., IoU=0.5)
            # iou_threshold = 0.5
            iou_index = np.where(np.isclose(cocoEval.params.iouThrs, iou_threshold))[0][0]
            # Loop over each category and plot the precision-recall curve
            for category_id in category_ids:
                # Extract precision values for the specific category
                precision_values = cocoEval.eval['precision'][iou_index, :, category_id, 0, -1] 
                # [TxRxKxAxM]; T=num_thresholds, R=num_recall_values, K=num_categories, A=num_area_ranges, M=num_max_dets
                # here, we are only interested in the precision values for the specific category and IoU threshold,
                recall_values = cocoEval.params.recThrs

                # store the precision and recall values in the self.precision_recall_dict
                if epoch not in self.precision_recall_dict.keys():
                    self.precision_recall_dict[epoch] = {}
                if iou_type not in self.precision_recall_dict[epoch].keys():
                    self.precision_recall_dict[epoch][iou_type] = {}
                if iou_threshold not in self.precision_recall_dict[epoch][iou_type].keys():
                    self.precision_recall_dict[epoch][iou_type][iou_threshold] = {}

                self.precision_recall_dict[epoch][iou_type][iou_threshold][category_id] = {"precision": precision_values.tolist(), 
                                                                                           "recall": recall_values.tolist()}
                
                # Plot the precision-recall curve for each category
                fig = plt.figure(figsize=(7, 7))
                plt.plot(recall_values, precision_values, label="Category ID {}".format(category_id))
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curve for Category ID {} for IoU = {} ({})'.format(category_id, iou_threshold, iou_type))
                plt.legend()
                plt.grid(True)
                # Save the plot to a temporary file
                plot_filename = 'pr_curve_{}_iou_{}_cat_id_{}_epoch_{}.png'.format(iou_type, iou_threshold, category_id, epoch)
                temp_file_path = os.path.join(self.metric_logging_dir, "precision_recall", plot_filename)
                if not os.path.exists(os.path.dirname(temp_file_path)):
                    os.makedirs(os.path.dirname(temp_file_path))
                plt.savefig(temp_file_path)
                plt.close()
                fig_tag = 'precision_recall_curves_{}/category_{}'.format(iou_type, category_id)           
    
    def evaluate_and_log(self, epoch):

        coco_evaluator, average_losses_dict = evaluate(model=self.model, data_loader=self.test_data_loader, 
                                                device=self.device)
        # Log average losses
        if coco_evaluator is not None and coco_evaluator.coco_eval is not None:

            # Log precision-recall curves
            self.log_precision_recall(iou_threshold=0.5, coco_evaluator=coco_evaluator, epoch=epoch) 
        
        return coco_evaluator, average_losses_dict



       
    def step(self):  # This is called iteratively.
        """
        Train the model for one epoch and perform evaluation during training
        """

        print("\n")
        print("Training started...")
        print("\n")
        since = time.time()
        equal_weight_AP_list =  []
        weighted_AP_1_list = []
        weighted_AP_2_list = []

        # keys are the epochs, values are the the dictionary with the average losses for each epoch over all batches
        self.final_training_loss_dict = { i: {} for i in range(self.start_epoch, self.args.epochs)} 

        # keys are the epochs, values are the the dictionary with the average accuracies for each epoch over all batches
        self.final_validation_loss_dict = { i: {} for i in range(self.start_epoch, self.args.epochs)} 

        # keys are the epochs, values are the the dictionary with the average accuracies for each epoch over all batches
        self.final_accuracy_dict = { i: {} for i in range(self.start_epoch, self.args.epochs)} 

        # keys are epochs values are precision and recall
        self.precision_recall_dict = { i: {} for i in range(self.start_epoch, self.args.epochs)}


        self.hyperparams_dict = { i: {} for i in range(self.start_epoch, self.args.epochs)}

        for epoch in tqdm(range(self.start_epoch, self.args.epochs), desc="Training Progress", unit="epoch"):
            print("\n")
            print("Epoch {}/{}".format(epoch + 1, self.args.epochs))
            #returns average_losses_dict a dictionary with the average losses for one epoch over all batches
            metric_logger, average_losses_dict, hyperparameters = train_one_epoch(model=self.model, optimizer=self.optimizer, 
                                                                    lr_scheduler_list=self.lr_scheduler_list, 
                                                                    data_loader=self.train_data_loader, device=self.device, epoch=epoch,
                                                                    print_freq=self.print_freq) 
            #print(f"average_losses_dict: {average_losses_dict}")
            
            if hyperparameters is not None:
                self.hyperparams_dict[epoch] = hyperparameters
                # store the hyperparameters as json to the metric_logging directory
                hp_filename = f'hyperparameters.json'
                with open(os.path.join(self.metric_logging_dir, "hyperparameters", hp_filename), 'w') as f:
                    json.dump(self.hyperparams_dict, indent=1, fp=f)
                
            if average_losses_dict is not None:
                #self.save_checkpoint(epoch, losses_dict=losses_dict)
                self.final_training_loss_dict[epoch] = average_losses_dict

                # Log the training and validation losses - dump the losses to a json file
                training_losses_filename = f'training_losses.json'
                with open(os.path.join(self.metric_logging_dir, "losses", training_losses_filename), 'w') as f:
                    json.dump(self.final_training_loss_dict, indent=1, fp=f)
        
            # perform evaluation during training to get the APs
            if self.do_evaluation_during_training:

                coco_evaluator, average_losses_dict = self.evaluate_and_log(epoch)

                if 'loss' in average_losses_dict.keys():
                    val_loss = average_losses_dict['loss']
                     # check if ReduceLROnPlateau scheduler is used and step the scheduler
                    for scheduler in self.lr_scheduler_list:
                        if isinstance(scheduler, ReduceLROnPlateau):
                            scheduler.step(val_loss)
                    self.early_stopping(val_loss, self.model)
                    if self.early_stopping.early_stop:
                        print("Early stopping")
                        break
                else: 
                    print("Validation loss could not be determined.")
                
                # store the validation losses for each epoch
                self.final_validation_loss_dict[epoch]= average_losses_dict # store the validation losses for each epoch 
                validation_losses_filename = f'validation_losses.json'
                with open(os.path.join(self.metric_logging_dir, "losses", validation_losses_filename), 'w') as f:
                    json.dump(self.final_validation_loss_dict, indent=3, fp=f)

                if 'bbox' in coco_evaluator.coco_eval and len(coco_evaluator.coco_eval['bbox'].stats) > 0:
                    bbox_AP_50 = coco_evaluator.coco_eval['bbox'].stats[1] # average precision for bounding boxes mAP at IoU=0.5

                if 'segm' in coco_evaluator.coco_eval and len(coco_evaluator.coco_eval['segm'].stats) > 0:
                    segm_AP_50 = coco_evaluator.coco_eval['segm'].stats[1] # average precision for masks mAP at IoU=0.5

                # store the APs for each epoch
                self.final_accuracy_dict[epoch] = {"AP_bbox": bbox_AP_50, "AP_segm": segm_AP_50}
                
                # Equal weighting
                equal_weight_AP = (bbox_AP_50 + segm_AP_50) / 2
                equal_weight_AP_list.append(equal_weight_AP)
                # bbox weightings
                weighted_AP_bbox = 0.7 * bbox_AP_50 + 0.3 * segm_AP_50
                weighted_AP_1_list.append(weighted_AP_bbox)
                # segm weightings
                weighted_AP_segm = 0.3 * bbox_AP_50 + 0.7 * segm_AP_50
                weighted_AP_2_list.append(weighted_AP_segm)
                
                # store the weighted APs for each epoch
                self.final_accuracy_dict[epoch]["equal_weight_AP"] = equal_weight_AP
                self.final_accuracy_dict[epoch]["weighted_AP_bbox"] = weighted_AP_bbox
                self.final_accuracy_dict[epoch]["weighted_AP_segm"] = weighted_AP_segm

                final_accuracy_filename = f'final_accuracy.json'
                with open(os.path.join(self.metric_logging_dir, "accuracy", final_accuracy_filename), 'w') as f:
                    json.dump(self.final_accuracy_dict, indent=1, fp=f)

                precision_recall_filename = f'precision_recall.json'
                # avoid TypeError: Object of type ndarray is not JSON serializable
                precision_recall_dict = self.precision_recall_dict
                with open(os.path.join(self.metric_logging_dir, "precision_recall", precision_recall_filename), 'w') as f:
                    json.dump(precision_recall_dict, indent=3, fp=f)
      
            
        # calculate the mean of the APs over all epochs
        equal_weight_AP_mean = np.mean(equal_weight_AP_list)
        weighted_AP_1_mean = np.mean(weighted_AP_1_list)
        weighted_AP_2_mean = np.mean(weighted_AP_2_list)


        # Return the validation accuracy as the metric to optimize
        return {"val_loss": val_loss, "val_accuracy": equal_weight_AP_mean}




            

def run_hyperparam_tune(num_samples=40):
    # Define the configuration for Ray Tune
    hp_param_space = {
        "learning_rate": tune.grid_search([0.001, 0.01, 0.1]),
        "momentum": tune.grid_search([0.9, 0.95, 0.99]),
        "weight_decay": tune.grid_search([0.0001, 0.0005, 0.001]),
        "batch_size": tune.choice([32]), #tune.choice([2, 4, 8, 16]),
        "epochs": tune.choice([10]),
        "backbone": tune.choice(['resnet50']), #, 'resnet101']),
        "optimizer": tune.choice(['SGD']), #, 'Adam', 'AdamW']) 
        "pretrained": tune.choice([True]),
        "trainable_layers": tune.choice([5, 3, 1]),
        "num_classes": tune.choice([2]), # actual class count; one class is automatically added for the background class
    }
    #experiment_storage_path = Path("./hp_param_tune").resolve()
    storage_path = Path("./hp_param_tune").resolve()
    print(f"storage_path: {storage_path}")

    best_checkpoint_max_acc_storage_path = os.path.join(storage_path, "best_checkpoint_max_accuracy")
    print(f"best_checkpoint_max_acc_storage_path: {best_checkpoint_max_acc_storage_path}")

    best_checkpoint_min_loss_storage_path = os.path.join(storage_path, "best_checkpoint_min_loss")
    print(f"best_checkpoint_min_loss_storage_path: {best_checkpoint_min_loss_storage_path}")
    
    # instantiate the tuner and apply the configuration
    tuner = tune.Tuner(
                        trainable=MaskRCNNTrainable,
                        run_config=train.RunConfig(
                                # Train for 20 steps
                                stop={"training_iteration": 20},
                                storage_path=storage_path,
                                name="train_mask_r_cnn",
                                verbose=1,
                                storage_filesystem=fs.FileSystem.from_uri(storage_path)[0],
                                checkpoint_config=train.CheckpointConfig(checkpoint_at_end=True,
                                                                        num_to_keep=2,
                                                                        # *Best* checkpoints are determined by these params:
                                                                        checkpoint_score_attribute="val_loss",
                                                                        checkpoint_score_order="min",
                                                                        ),
                                ),
                        tune_config=tune.TuneConfig(metric="val_loss",
                                                    mode="min",
                                                    #metric="accuracy",
                                                    #mode="max",
                                                    #scheduler=scheduler,
                                                    num_samples=num_samples),
                        param_space=hp_param_space,
                    )
    result_grid = tuner.fit()

    # Get the best checkpoint and restore the model
    best_checkpoint = result_grid.get_best_checkpoint()

     # Get the best result based on a particular metric.
    best_result_min_loss = result_grid.get_best_result(metric="loss", mode="min")
    # Get the best checkpoint corresponding to the best result.
    best_checkpoint_min_loss = best_result_min_loss.checkpoint 
    best_checkpoint_dir_min_loss = best_checkpoint_min_loss.to_directory(path=best_checkpoint_min_loss_storage_path)
    print(f"best_checkpoint_dir_min_loss: {best_checkpoint_dir_min_loss}")
    # Get a dataframe for the last reported results of all of the trials
    df = result_grid.get_dataframe() 
    # Get a dataframe for the minimum loss seen for each trial
    df = result_grid.get_dataframe(filter_metric="loss", filter_mode="min") 
    print(f"df: {df}")



if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    run_hyperparam_tune(num_samples=40, max_num_epochs=10, gpus_per_trial=1)

