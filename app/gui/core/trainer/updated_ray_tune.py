
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from tqdm import tqdm
#from torch.utils.tensorboard import SummaryWriter
from train_utils.engine import *
from train_utils.utils import *
from train_utils.transforms import *
from train_utils.dataset import *

import tempfile
from ray import train, tune

from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.train import Checkpoint
import pyarrow.fs as fs


def load_data(data_dir="./data", train_ratio=0.8, val_ratio=0.2, perform_slicing=True):
    # Create data loaders and perform train, val split
    annotations_path = os.path.join(data_dir, "annotations", "instances_default.json")
    out_json_paths  = split_from_file(annotations_path, [train_ratio, val_ratio], 
                                            names=["train", "val"], do_shuffle=True)
    images_folder = os.path.join(data_dir, "images")

    #train data set
    train_dataset = create_dataset(images_folder=images_folder, train=True, 
                                            annotation_file=out_json_paths[0], 
                                            perform_slicing=perform_slicing)
    # test data set
    test_dataset  = create_dataset(images_folder=images_folder, train=False,
                                            annotation_file=out_json_paths[1], 
                                            perform_slicing=perform_slicing)
    return train_dataset, test_dataset


def get_model(backbone_name="resnet50", num_classes=3):
    """
    Initialize Mask R-CNN model
    """
    # Define the Mask R-CNN model
    #self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)
    # ----- custom backbone -----#
    if backbone_name == "resnet50":
        backbone  = resnet_fpn_backbone("resnet50", pretrained=True, trainable_layers=4) # return 256 feature maps for ResNet 50
        backbone.out_channels = 256
    elif backbone_name == "resnet101":
        backbone  = resnet_fpn_backbone("resnet101", pretrained=True, trainable_layers=4) # return 256 feature maps for ResNet 50
        backbone.out_channels = 256
        
    model = MaskRCNN(backbone, num_classes=num_classes)
    #self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)
    #model = maskrcnn_resnet50_fpn(weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes )
    
    #------ custom anchor generator ----#
    # set anchor size to 16, 32, 64, 128, 256
    model.rpn.anchor_generator.sizes = ((16, 32, 64, 128, 256),)
    # set anchor aspect ratio to 0.5, 1.0, 2.0
    model.rpn.anchor_generator.aspect_ratios = ((0.5, 1.0, 2.0),)
    # set amount of anchors to 2400 
    model.rpn.anchor_generator.num_anchors = (2400,)
    
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels=in_features_mask, 
                                                            dim_reduced=256, 
                                                            num_classes=num_classes)  # hidden_layer = 256
    return model


def evaluate_and_log(model, epoch, test_data_loader, device):
    coco_evaluator = evaluate(model,test_data_loader, device=device)
    if coco_evaluator is not None and coco_evaluator.coco_eval is not None:
        if 'bbox' in coco_evaluator.coco_eval and len(coco_evaluator.coco_eval['bbox'].stats) > 0:
            AP_bbox = coco_evaluator.coco_eval['bbox'].stats[0]
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
        if 'segm' in coco_evaluator.coco_eval and len(coco_evaluator.coco_eval['segm'].stats) > 0:
            AP_segm = coco_evaluator.coco_eval['segm'].stats[0]
            AP_50 = coco_evaluator.coco_eval['segm'].stats[1]
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
    return coco_evaluator.coco_eval, AP_bbox, AP_segm
    

def train_and_eval(config, model=None, optimizer=None, 
                    train_data_loader=None, test_data_loader=None, device=None, 
                    print_freq=None, args=None):
    print("\n")
    print("Hyperparameter Tuning started...")
    print("\n")
    since = time.time()
    equal_weight_AP_list = []
    weighted_AP_1_list = []
    weighted_AP_2_list = []
    trainable_epochs = config["epochs"]
    for epoch in tqdm(range(0, trainable_epochs), desc="Training Progress", unit="epoch"):
        print("\n")
        print("Epoch {}/{}".format(epoch + 1, trainable_epochs))
        # train for one epoch
        metric_logger, losses_dict = train_one_epoch(model=model, optimizer=optimizer, 
                                    data_loader=train_data_loader, device=device, epoch=epoch,
                                    print_freq=print_freq)

        # evaluate on the test dataset
        coco_eval, AP_bbox, AP_segm = evaluate_and_log(model, epoch, test_data_loader, device, 
                                                      )
        # Equal weighting
        equal_weight_AP = (AP_bbox + AP_segm) / 2
        equal_weight_AP_list.append(equal_weight_AP)

        # Experiment with different weightings
        weighted_AP_1 = 0.7 * AP_bbox + 0.3 * AP_segm
        weighted_AP_1_list.append(weighted_AP_1)
        weighted_AP_2 = 0.3 * AP_bbox + 0.7 * AP_segm
        weighted_AP_2_list.append(weighted_AP_2)
        
        # Save checkpoint
        metrics = {"loss": losses_dict['loss'], "accuracy": weighted_AP_1}
        
    
        """
        #working solution:
        trial_dir = train.get_context().get_trial_dir()
        print(f"trial_dir: {trial_dir}")
        trial_checkpoint = train.get_checkpoint()
        print(f"trial_checkpoint: {trial_checkpoint}")
        
        storage_path = os.path.join(trial_dir, f"checkpoint_{epoch}")
        ckpt_file_path = os.path.join(storage_path, "checkpoint.pt")
        
        print(f"path: {ckpt_file_path}")
        
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)
        
        torch.save(
            (model.state_dict(), optimizer.state_dict()),
            ckpt_file_path
        )
        checkpoint = Checkpoint.from_directory(trial_dir)
        train.report(metrics) #, checkpoint=checkpoint)
        print(f"Checkpoint stored in trial_dir {storage_path}")
        """
            
        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and will potentially be accessed through in ``get_checkpoint()``
        # in future iterations.
        # Note to save a file like checkpoint, you still need to put it under a directory
        # to construct a checkpoint.
  
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save(
                (model.state_dict(), optimizer.state_dict()), path
            )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            checkpoint.filesystem = fs.LocalFileSystem()
            print(f"checkpoint.filesystem: {checkpoint.filesystem}")
            train.report(
                metrics=metrics,
                checkpoint=checkpoint,
            )

    # calculate the mean of the APs over all epochs
    equal_weight_AP_mean = np.mean(equal_weight_AP_list)
    weighted_AP_1_mean = np.mean(weighted_AP_1_list)
    weighted_AP_2_mean = np.mean(weighted_AP_2_list)

    print("\n")
    print("Mean AP over all epochs:")
    print("Equal weighting: {}".format(equal_weight_AP_mean))
    print("Weighted AP 1: {}".format(weighted_AP_1_mean))
    print("Weighted AP 2: {}".format(weighted_AP_2_mean))
    print("\n")

    # print the complete list of APs
    print("List of APs over all epochs:")
    print("Equal weighting:")
    print(equal_weight_AP_list)
    print("Weighted AP 1:")
    print(weighted_AP_1_list)
    print("Weighted AP 2:")
    print(weighted_AP_2_list)

   
def train_model(config, data_dir=None): 

    backbone_name = config["backbone"]
    print(f"Setting up model with backbone {backbone_name}")
    model = get_model(backbone_name=backbone_name)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_name = config["optimizer"]

    if optimizer_name == "SGD":
        if "momentum" in config.keys():
            optimizer = optim.SGD(model.parameters(), lr=config["lr"],
                                   momentum=config["momentum"])
        elif "momentum" in config.keys() and "weight_decay" in config.keys(): 
            optimizer = optim.SGD(model.parameters(), lr=config["lr"], 
                                  momentum=config["momentum"],
                                  weight_decay=config["weight_decay"])
        else:
            optimizer = optim.SGD(model.parameters(), lr=config["lr"])
        print(f"optimizer: {optimizer}")
    elif optimizer_name == "Adam":
        if "weight_decay" in config.keys():
            optimizer = optim.Adam(model.parameters(), lr=config["lr"], 
                                   weight_decay=config["weight_decay"])
        else:
            optimizer = optim.Adam(model.parameters(), lr=config["lr"])
        print(f"optimizer: {optimizer}")
    else:
        raise ValueError("Unknown optimizer {}".format(optimizer_name))

    trainset, testset = load_data(data_dir)

    trainloader = create_dataloader(trainset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=8)
    valloader   = create_dataloader(testset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=8)
    
    # Load existing checkpoint through `get_checkpoint()` API.
    if train.get_checkpoint():
        loaded_checkpoint = train.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
            )
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

    
    train_and_eval(config, model=model, optimizer=optimizer,
                    train_data_loader=trainloader, test_data_loader=valloader, device=device,
                    print_freq=200, args=None)
    
      
    print("Finished Training.")   
    
  
    
def perform_ray_tune(num_samples=2, max_num_epochs=4, gpus_per_trial=1):
    config = {
        "lr": tune.loguniform(1e-5, 1e-2),
        "momentum": tune.uniform(0,1),
        "weight_decay": tune.loguniform(1e-3, 1e-5), 
        "batch_size": tune.choice([2]), #tune.choice([2, 4, 8, 16]),
        "epochs": tune.choice([4]),
        "backbone": tune.choice(['resnet50', 'resnet101']),
        "optimizer": tune.choice(['SGD', 'Adam']) 
    }
    #experiment_storage_path = Path("./hp_param_tune").resolve()
    storage_path = Path("./hp_param_tune").resolve()
    print(f"storage_path: {storage_path}")

    best_checkpoint_max_acc_storage_path = os.path.join(storage_path, "best_checkpoint_max_accuracy")
    print(f"best_checkpoint_max_acc_storage_path: {best_checkpoint_max_acc_storage_path}")

    best_checkpoint_min_loss_storage_path = os.path.join(storage_path, "best_checkpoint_min_loss")
    print(f"best_checkpoint_min_loss_storage_path: {best_checkpoint_min_loss_storage_path}")
    
    # Define the checkpoint configuration
    checkpoint_config = train.CheckpointConfig(
        num_to_keep=2,
        # *Best* checkpoints are determined by these params:
        checkpoint_score_attribute="accuracy",
        checkpoint_score_order="max",
    )
    
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy"]) #, "training_iteration"])
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_model, 
                                 data_dir=Path("./data").resolve()
                                 ),
            resources={"cpu": 2, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            #metric="loss",
            #mode="min",
            metric="accuracy",
            mode="max",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=config,
        run_config=train.RunConfig(storage_path=storage_path,
                                   name="train_mask_r_cnn",
                                   verbose=1,
                                   storage_filesystem=fs.FileSystem.from_uri(storage_path)[0],
                                   checkpoint_config=checkpoint_config,
                                   )
        )
    
    result_grid = tuner.fit()
    
    # for maximized accuracy
    best_result_max_acc = result_grid.get_best_result(metric="accuracy", mode="max")
    print("Best trial config (max acc): {}".format(best_result_max_acc.config))
    print("Best trial final validation loss (max acc): {}".format(
        best_result_max_acc.metrics["loss"]))
    print("Best trial final validation accuracy (max acc): {}".format(
        best_result_max_acc.metrics["accuracy"]))

    #test_best_model(best_result, smoke_test=smoke_test)
    # Gets best checkpoint for trial based on accuracy.
    # best_checkpoint = analysis.get_best_checkpoint(trial=best_trial, metric="accuracy", mode="max")
    best_checkpoint_max_acc = best_result_max_acc.checkpoint
    best_checkpoint_dir_max_acc = best_checkpoint_max_acc.to_directory(path=best_checkpoint_max_acc_storage_path)
    print(f"best_checkpoint_dir_max_acc: {best_checkpoint_dir_max_acc}")


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

    # loading the best checkpoint
    config = best_result_max_acc.config
    model = get_model(config["backbone"])
    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)
    model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir_max_acc, "checkpoint.pt"))
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
    print(f"loaded the best checkpoint (accuracy {best_result_max_acc.metrics['accuracy']}) from dir {best_checkpoint_dir_max_acc}")

    
if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    perform_ray_tune(num_samples=40, max_num_epochs=10, gpus_per_trial=1)