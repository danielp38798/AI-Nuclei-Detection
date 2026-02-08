import time
import glob
import re
import os
import zipfile
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from gui.core.trainer.train_utils.engine import *
from gui.core.trainer.train_utils.utils import *
from gui.core.trainer.train_utils.transforms import *
from gui.core.trainer.train_utils.dataset import *
import argparse


# modified version of torchvision's MaskRCNN which also returns the losses during training not only the detections (roi_heads and generalised_rcnn were modified)
from gui.core.trainer.vision.torchvision.models.detection import MaskRCNN # modified
from gui.core.trainer.vision.torchvision.models.detection.rpn import AnchorGenerator

from gui.core.trainer.vision.torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from gui.core.trainer.vision.torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from gui.core.trainer.vision.torchvision.models.detection.rpn import AnchorGenerator
from torch.optim.lr_scheduler import ReduceLROnPlateau, LinearLR # StepLR, MultiStepLR, CosineAnnealingLR, CyclicLR, OneCycleLR, ExponentialLR, LambdaLR


import matplotlib.pyplot as plt
from gui.core.trainer.vision.torchvision.models import ResNet50_Weights, ResNet101_Weights
from torch.optim.lr_scheduler import ReduceLROnPlateau, LinearLR # StepLR, MultiStepLR, CosineAnnealingLR, CyclicLR, OneCycleLR, ExponentialLR, LambdaLR
from gui.core.trainer.vision.torchvision.models.detection.backbone_utils import resnet_fpn_backbone


import numpy as np
import matplotlib.pyplot as plt
import json

class Trainer:
    """
    Trainer class for Mask R-CNN model
    operations such as training, evaluation, 
    checkpoint saving, tensorboard logging
    are performed here.
    Training is performed on the GPU if available and can be done on image patches if specified.
    
    """
    def __init__(self, args):
        self.args = args

        self.print_freq = args.print_freq
        self.device = None

        self.annotations_path = None

        self.root_dir = args.data_dir
        self.ckpt_dir = args.ckpt_dir
        self.metric_logging_dir = os.path.join(os.getcwd(), "metric_logging")
        self.out_json_paths = None
        self.images_folder = None
        self.train_data_loader = None
        self.test_data_loader = None

        self.create_dirs()

        self.lr = args.lr
        self.num_epochs = args.epochs
        self.batch_size = args.batch_size
        self.num_classes = args.num_classes + 1 # add background class
        
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        
        self.model = None
        self.optimizer = None
        self.optimizer_name = args.optimizer
        self.backbone_name = args.backbone
        self.pretrained = args.pretrained
        self.trainable_layers = args.trainable_layers
        
        self.lr_scheduler_list = []
        self.early_stopping = EarlyStopping(patience=10, verbose=True, path=os.path.join(self.ckpt_dir, "early_stopping_checkpoint.pth"))

        self.perform_slicing = args.perform_slicing
        self.do_evaluation_during_training = args.do_evaluation_during_training
        self.do_model_evaluation = args.do_model_evaluation
        self.eval_freq = args.eval_freq

        self.train_ratio = 0.8
        self.val_ratio = 0.2
        self.start_epoch = 0

        self.writer = None
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

        self.writer.add_hparams(self.hyperparams, {})
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

    def initialize_tensorboard(self):
        """
        Initialize TensorBoard
        """
        self.writer = SummaryWriter()
        
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
        
        # Print other variables
        print("\n")
        print("|------------ OTHER VARIABLES ---------------|")

        print("Perform evaluation during training: {}".format(self.do_evaluation_during_training))
        print("Evaluation frequency: {}".format(self.eval_freq))
        print("Perform slicing: {}".format(self.perform_slicing))

        if self.do_model_evaluation:
            print("current mode: evaluation only")
        else:
            print("current mode: training")
        


    def load_checkpoint(self):
        """
        Load latest checkpoint if available
        """
        self.ckpt_dir = self.args.ckpt_dir
        ext = ".pth"
        prefix = os.path.join(self.ckpt_dir, "checkpoint")
        ckpts = glob.glob(prefix + "-*" + ext)
        ckpts.sort(key=lambda x: int(re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]).group(1)))
        print(f"Detected {len(ckpts)} checkpoints.")
        if ckpts:
            last_ckpt = ckpts[-1]
            print(f"Found latest checkpoint '{last_ckpt}'.")
            checkpoint = torch.load(last_ckpt, map_location=self.device)
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.start_epoch = checkpoint["epochs"]
            del checkpoint
            torch.cuda.empty_cache()
            print(f"Loaded model '{self.model.__class__.__name__}' from checkpoint '{last_ckpt}' to {self.device}.") 
        
        # if the checkpoint directory contains a final model, load it
        final_model_path = os.path.join(self.ckpt_dir, f'maskrcnn_model_final_{self.num_epochs}_epochs.pth')
        if os.path.exists(final_model_path):
            print(f"Found final model '{final_model_path}'.")
            self.model.load_state_dict(torch.load(final_model_path, map_location=self.device))
            print(f"Loaded model '{self.model.__class__.__name__}' from final model '{final_model_path}' to {self.device}.")

        

    def log_model_summary(self):
        # Log model summary to TensorBoard
        self.writer.add_text("Model Summary", str(self.model))
        
    def get_hyperparameters(self):
        # Get hyperparameters
        return self.hyperparams
    
    def save_checkpoint(self, epochs, **kwargs):
        
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        checkpoint = {}
        checkpoint["model"] = self.model.state_dict()
        checkpoint["optimizer"]  = self.optimizer.state_dict()
        checkpoint["epochs"] = epochs
        for k, v in kwargs.items():
            checkpoint[k] = v
        ext = ".pth"
        file_name = "checkpoint-{}{}".format(epochs,ext)
        ckpt_path = os.path.join(self.ckpt_dir, file_name)
        torch.save(checkpoint, ckpt_path)
        print("Saved checkpoint: {}".format(file_name))

        # check the ckeckpoint directory for the amount of checkpoints and only keep the last 5
        prefix = os.path.join(self.ckpt_dir, "checkpoint")
        ext = ".pth"
        ckpts = glob.glob(prefix + "-*" + ext)
        ckpts.sort(key=lambda x: int(re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]).group(1)))
        if ckpts and len(ckpts) > 5:
            for ckpt in ckpts[:-5]:
                os.remove(ckpt)
                print(f"Removed checkpoint '{ckpt}'.")

    # ---- Helper functions for logging and evaluation ---- #
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
            self.writer.add_figure("Final Validation Losses", fig)
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
                self.writer.add_figure(fig_tag, fig)
    
    def evaluate_and_log(self, epoch):

        coco_evaluator, average_losses_dict = evaluate(model=self.model, data_loader=self.test_data_loader, 
                                                device=self.device)
        # Log average losses
        if coco_evaluator is not None and coco_evaluator.coco_eval is not None:

            # Log precision-recall curves
            self.log_precision_recall(iou_threshold=0.5, coco_evaluator=coco_evaluator, epoch=epoch) 

            if 'bbox' in coco_evaluator.coco_eval and len(coco_evaluator.coco_eval['bbox'].stats) > 0:
                AP = coco_evaluator.coco_eval['bbox'].stats[0]
                AP_50 = coco_evaluator.coco_eval['bbox'].stats[1]
                AP_75 = coco_evaluator.coco_eval['bbox'].stats[2]
                AP_small = coco_evaluator.coco_eval['bbox'].stats[3]
                AP_medium = coco_evaluator.coco_eval['bbox'].stats[4]
                AP_large = coco_evaluator.coco_eval['bbox'].stats[5]
                AR_1 = coco_evaluator.coco_eval['bbox'].stats[6]
                AR_10 = coco_evaluator.coco_eval['bbox'].stats[7]
                AR_100 = coco_evaluator.coco_eval['bbox'].stats[8]
                AR_small = coco_evaluator.coco_eval['bbox'].stats[9]
                AR_medium = coco_evaluator.coco_eval['bbox'].stats[10]
                AR_large = coco_evaluator.coco_eval['bbox'].stats[11]
                self.writer.add_scalar("Evaluation 'bbox'/AP", AP, epoch)
                self.writer.add_scalar("Evaluation 'bbox'/AP_50", AP_50, epoch)
                self.writer.add_scalar("Evaluation 'bbox'/AP_75", AP_75, epoch)
                self.writer.add_scalar("Evaluation 'bbox'/AP_small", AP_small, epoch)
                self.writer.add_scalar("Evaluation 'bbox'/AP_medium", AP_medium, epoch)
                self.writer.add_scalar("Evaluation 'bbox'/AP_large", AP_large, epoch)
                self.writer.add_scalar("Evaluation 'bbox'/AR_1", AR_1, epoch)
                self.writer.add_scalar("Evaluation 'bbox'/AR_10", AR_10, epoch)
                self.writer.add_scalar("Evaluation 'bbox'/AR_100", AR_100, epoch)
                self.writer.add_scalar("Evaluation 'bbox'/AR_small", AR_small, epoch)
                self.writer.add_scalar("Evaluation 'bbox'/AR_medium", AR_medium, epoch)
                self.writer.add_scalar("Evaluation 'bbox'/AR_large", AR_large, epoch)

                
            if 'segm' in coco_evaluator.coco_eval and len(coco_evaluator.coco_eval['segm'].stats) > 0:
                AP = coco_evaluator.coco_eval['segm'].stats[0]
                AP_50 = coco_evaluator.coco_eval['segm'].stats[1]
                print(f"AP_50 (segm): {AP_50}")
                AP_75 = coco_evaluator.coco_eval['segm'].stats[2]
                AP_small = coco_evaluator.coco_eval['segm'].stats[3]
                AP_medium = coco_evaluator.coco_eval['segm'].stats[4]
                AP_large = coco_evaluator.coco_eval['segm'].stats[5]
                AR_1 = coco_evaluator.coco_eval['segm'].stats[6]
                AR_10 = coco_evaluator.coco_eval['segm'].stats[7]
                AR_100 = coco_evaluator.coco_eval['segm'].stats[8]
                AR_small = coco_evaluator.coco_eval['segm'].stats[9]
                AR_medium = coco_evaluator.coco_eval['segm'].stats[10]
                AR_large = coco_evaluator.coco_eval['segm'].stats[11]
                self.writer.add_scalar("Evaluation 'segm'/AP", AP, epoch)   
                self.writer.add_scalar("Evaluation 'segm'/AP_50", AP_50, epoch)
                self.writer.add_scalar("Evaluation 'segm'/AP_75", AP_75, epoch)
                self.writer.add_scalar("Evaluation 'segm'/AP_small", AP_small, epoch)
                self.writer.add_scalar("Evaluation 'segm'/AP_medium", AP_medium, epoch)
                self.writer.add_scalar("Evaluation 'segm'/AP_large", AP_large, epoch)
                self.writer.add_scalar("Evaluation 'segm'/AR_1", AR_1, epoch)
                self.writer.add_scalar("Evaluation 'segm'/AR_10", AR_10, epoch)
                self.writer.add_scalar("Evaluation 'segm'/AR_100", AR_100, epoch)
                self.writer.add_scalar("Evaluation 'segm'/AR_small", AR_small, epoch)
                self.writer.add_scalar("Evaluation 'segm'/AR_medium", AR_medium, epoch)
                self.writer.add_scalar("Evaluation 'segm'/AR_large", AR_large, epoch)
        
        return coco_evaluator.coco_eval, average_losses_dict
    
    def create_final_accuracy_plot(self):
        """
        Create final accuracy plot, showing the average precision over all epochs
        """


        if self.final_accuracy_dict:
            fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(10, 5))
            if "AP_bbox" in self.final_accuracy_dict.keys():
                # Plot the average precision over all epochs
                ax[0, 0].plot(self.final_accuracy_dict.keys(), [v['AP_bbox'] for v in self.final_accuracy_dict.values()], label="AP_bbox")
                ax[0, 0].set_xlabel("Epoch")
                ax[0, 0].set_ylabel("Average Precision (AP) (bbox)")	
                ax[0, 0].set_title(f"Average Precision (AP) (bbox) over {self.num_epochs} epochs")

            if "AP_segm" in self.final_accuracy_dict.keys():
                ax[0, 1].plot(self.final_accuracy_dict.keys(), [v['AP_segm'] for v in self.final_accuracy_dict.values()], label="AP_segm")
                ax[0, 1].set_xlabel("Epoch")
                ax[0, 1].set_ylabel("Average Precision (AP)")
                ax[0, 1].set_title(f"Average Precision (AP) (segm) over {self.num_epochs} epochs")

            if "equal_weight_AP" in self.final_accuracy_dict.keys():
                ax[1, 0].plot(self.final_accuracy_dict.keys(), [v['equal_weight_AP'] for v in self.final_accuracy_dict.values()], label="equal_weight_AP")
                ax[1, 0].set_xlabel("Epoch")
                ax[1, 0].set_ylabel("Average Precision (AP)")
                ax[1, 0].set_title(f"Average Precision (AP) over {self.num_epochs} epochs")

            if "weighted_AP_bbox" in self.final_accuracy_dict.keys():
                ax[1, 1].plot(self.final_accuracy_dict.keys(), [v['weighted_AP_bbox'] for v in self.final_accuracy_dict.values()], label="weighted_AP_bbox")
                ax[1, 1].set_xlabel("Epoch")
                ax[1, 1].set_ylabel("Average Precision (AP)")
                ax[1, 1].set_title(f"Average Precision (AP) (bbox 70% weight) over {self.num_epochs} epochs")
            
            if "weighted_AP_segm" in self.final_accuracy_dict.keys():
                ax[2, 0].plot(self.final_accuracy_dict.keys(), [v['weighted_AP_segm'] for v in self.final_accuracy_dict.values()], label="weighted_AP_segm")
                ax[2, 0].set_xlabel("Epoch")
                ax[2, 0].set_ylabel("Average Precision (AP)")
                ax[2, 0].set_title(f"Average Precision (AP) (segm 70% weight) over {self.num_epochs} epochs")

            # set title for the plot
            fig.suptitle("Final Accuracy")

            self.writer.add_figure("Final Accuracy", fig)

            # Save the plot to a temporary file
            plot_filename = 'final_accuracy.png'
            temp_file_path = os.path.join(self.metric_logging_dir, "accuracy", plot_filename)
            if not os.path.exists(os.path.dirname(temp_file_path)):
                os.makedirs(os.path.dirname(temp_file_path))
            plt.savefig(temp_file_path)
            plt.close(fig)


    def save_final_model(self):
        """
        Save the final model
        """

        model_path = os.path.join(self.ckpt_dir, f'maskrcnn_model_final_{self.num_epochs}_epochs.pth')
        torch.save(self.model.state_dict(), model_path)
        print(f"Saved final model at {model_path}") 

    def train(self):
        """
        Train the model
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

                coco_eval, average_losses_dict = self.evaluate_and_log(epoch)

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

                AP_bbox = coco_eval['bbox'].stats[1] # average precision for bounding boxes mAP at IoU=0.5
                AP_segm = coco_eval['segm'].stats[1] # average precision for masks mAP at IoU=0.5

                # store the APs for each epoch
                self.final_accuracy_dict[epoch] = {"AP_bbox": AP_bbox, "AP_segm": AP_segm}
                
                # Equal weighting
                equal_weight_AP = (AP_bbox + AP_segm) / 2
                equal_weight_AP_list.append(equal_weight_AP)
                # bbox weightings
                weighted_AP_bbox = 0.7 * AP_bbox + 0.3 * AP_segm
                weighted_AP_1_list.append(weighted_AP_bbox)
                # segm weightings
                weighted_AP_segm = 0.3 * AP_bbox + 0.7 * AP_segm
                weighted_AP_2_list.append(weighted_AP_segm)
                
                # store the weighted APs for each epoch
                self.final_accuracy_dict[epoch]["equal_weight_AP"] = equal_weight_AP
                self.final_accuracy_dict[epoch]["weighted_AP_bbox"] = weighted_AP_bbox
                self.final_accuracy_dict[epoch]["weighted_AP_segm"] = weighted_AP_segm

                final_accuracy_filename = f'final_accuracy.json'
                with open(os.path.join(self.metric_logging_dir, "accuracy", final_accuracy_filename), 'w') as f:
                    json.dump(self.final_accuracy_dict, indent=1, fp=f)

                # Save the model with the best validation accuracy
                #if equal_weight_AP > best_AP:
                    #best_AP = equal_weight_AP
                    #self.save_checkpoint(epoch, losses_dict=average_losses_dict, eval_results=coco_eval)
                    #print("Best model saved. \nmAP (bbox, segm) = {:.4f}, {:.4f}, weighted AP = {:.4f}".format(AP_bbox, AP_segm, equal_weight_AP))

                precision_recall_filename = f'precision_recall.json'
                # avoid TypeError: Object of type ndarray is not JSON serializable
                precision_recall_dict = self.precision_recall_dict

                """
                # convert numpy arrays to lists
                for epoch, iou_type_dict in precision_recall_dict.items():
                    for iou_type, iou_threshold_dict in iou_type_dict.items():
                        for iou_threshold, category_dict in iou_threshold_dict.items():
                            for category_id, precision in category_dict.items():
                                precision_recall_dict[epoch][iou_type][iou_threshold][category_id]["precision"] = precision_recall_dict[epoch][iou_type][iou_threshold][category_id]["precision"].tolist()
                                precision_recall_dict[epoch][iou_type][iou_threshold][category_id]["recall"] = precision_recall_dict[epoch][iou_type][iou_threshold][category_id]["recall"].tolist()

                """

                with open(os.path.join(self.metric_logging_dir, "precision_recall", precision_recall_filename), 'w') as f:
                    json.dump(precision_recall_dict, indent=3, fp=f)
      
            
        # calculate the mean of the APs over all epochs
        equal_weight_AP_mean = np.mean(equal_weight_AP_list)
        weighted_AP_1_mean = np.mean(weighted_AP_1_list)
        weighted_AP_2_mean = np.mean(weighted_AP_2_list)
        #print("\n")
        #print("Mean AP over all epochs:")
        #print("Equal weighting: {}".format(equal_weight_AP_mean))
        #print("Weighted AP 1: {}".format(weighted_AP_1_mean))
        #print("Weighted AP 2: {}".format(weighted_AP_2_mean))
        #print("\n")
   

        # Create final loss plot
        if self.final_training_loss_dict:
            self.create_final_training_loss_plot()

        # Create final validation loss plot
        if self.final_validation_loss_dict:
            self.create_final_validation_loss_plot()

        # Create final accuracy plot
        if self.final_accuracy_dict:
            self.create_final_accuracy_plot()

        # Save the final model
        print("Saving final model...")
        self.save_final_model()
        print("\ntotal time of this training: {:.1f} s".format(time.time() - since))
        if self.start_epoch < self.args.epochs:
            print("already trained: {} epochs\n".format(epoch + 1))
        self.writer.close()

        # Return the validation accuracy as the metric to optimize
        return equal_weight_AP_mean # use this metric to optimize the hyperparameters
        
    def evaluate_model(self):   
        print("\n")
        print("Evaluation started...")
        print("\n")
        since = time.time()
        self.precision_recall_dict = { 0: {}}

        coco_evaluator, average_losses_dict = evaluate(model=self.model, data_loader=self.test_data_loader,  device=self.device)
        if coco_evaluator is not None and coco_evaluator.coco_eval is not None:
            # Log precision-recall curves
            self.log_precision_recall(iou_threshold=0.5, coco_evaluator=coco_evaluator, epoch=0) 
            if 'bbox' in coco_evaluator.coco_eval and len(coco_evaluator.coco_eval['bbox'].stats) > 0:
                AP_50 = coco_evaluator.coco_eval['bbox'].stats[1]
                print(f"AP_50 (bbox): {AP_50}")

            if 'segm' in coco_evaluator.coco_eval and len(coco_evaluator.coco_eval['segm'].stats) > 0:
                AP_50 = coco_evaluator.coco_eval['segm'].stats[1]
                print(f"AP_50 (segm): {AP_50}")
        if 'loss' in average_losses_dict.keys():
            val_loss = average_losses_dict['loss']
            print(f"Validation loss: {val_loss}")
        else:
            print("Validation loss could not be determined.")
        print("\ntotal time of this evaluation: {:.1f} s".format(time.time() - since))
        print("\n")

        precision_recall_filename = f'precision_recall.json'
        # avoid TypeError: Object of type ndarray is not JSON serializable
        precision_recall_dict = self.precision_recall_dict
        """
        # convert numpy arrays to lists
        for epoch, iou_type_dict in precision_recall_dict.items():
            for iou_type, iou_threshold_dict in iou_type_dict.items():
                for iou_threshold, category_dict in iou_threshold_dict.items():
                    for category_id, precision in category_dict.items():
                        precision_recall_dict[epoch][iou_type][iou_threshold][category_id]["precision"] = precision_recall_dict[epoch][iou_type][iou_threshold][category_id]["precision"].tolist()
                        precision_recall_dict[epoch][iou_type][iou_threshold][category_id]["recall"] = precision_recall_dict[epoch][iou_type][iou_threshold][category_id]["recall"].tolist()

        """

        with open(os.path.join(self.metric_logging_dir, "precision_recall", precision_recall_filename), 'w') as f:
            json.dump(precision_recall_dict, indent=3, fp=f)
            

    def run(self):
        """
        This method sets up the device, initializes the model, loads the dataset, and starts the training or evaluation process.

        """
        print("\n")
        print("|--------------------- AI NUCLEI DETECTION ---------------------|")
        print("\n")
        
        print("Setting up device...")
        self.setup_device()
        print("\n")
        
        self.initialize_tensorboard()
        print("TensorBoard initialized.")
        print("\n")
        
        print("Loading dataset...")
        self.load_data()
        print("Dataloaders created.")
        
        print("Setting up model...")
        self.setup_model()
        self.log_model_summary()
        print("\n")
        print("Loading checkpoint...")
        self.load_checkpoint()
        self.log_hyperparameters(file_path=os.path.join(self.metric_logging_dir, "hyperparameters", "initial_hyperparameters.json"))
        print("\n")
        print("|------------------- INITIALIZATION COMPLETE --------------------|")

        if self.do_model_evaluation:
            self.evaluate_model()
        else:
            self.train()
            print("\n")

def setup_args():
    parser = argparse.ArgumentParser(description="AI Nuclei Detection")

    parser.add_argument("--use_cuda", type=bool, default=True, help="Use CUDA")
    
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes")
    parser.add_argument("--lr", type=float, default=3.7e-05, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--backbone", type=str, default="resnet50", help="Backbone")

    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer")


    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay")


    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints", help="Checkpoint directory")

    parser.add_argument("--do_evaluation_during_training", type=bool, default=True, help="Perform evaluation")
    parser.add_argument("--perform_slicing", type=bool, default=True, help="Perform slicing")
    parser.add_argument("--print_freq", type=int, default=400, help="Print frequency")

    parser.add_argument("--eval_freq", type=int, default=1, help="Evaluation frequency")
    parser.add_argument("--do_model_evaluation", type=bool, default=False, help="Evaluate model")
    
    return parser.parse_args()

if __name__ == "__main__":

    args = setup_args()
    
    trainer = Trainer(args)
    trainer.run()
