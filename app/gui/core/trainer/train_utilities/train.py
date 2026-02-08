
import datetime
import os
import time

import presets
import torch
import torch.utils.data
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn
import utils
from coco_utils import get_coco
from engine import evaluate, train_one_epoch, validate, EarlyStopping
from group_by_aspect_ratio import create_aspect_ratio_groups, GroupedBatchSampler
from torchvision.transforms import InterpolationMode
from transforms import SimpleCopyPaste

import numpy as np
import json
import os
import matplotlib.pyplot as plt

# Set the random seed for reproducibility
torch.manual_seed(0)



def copypaste_collate_fn(batch):
    copypaste = SimpleCopyPaste(blending=True, resize_interpolation=InterpolationMode.BILINEAR)
    return copypaste(*utils.collate_fn(batch))


def get_dataset(is_train, args):
    image_set = "train" if is_train else "val"
    #num_classes, mode = {"coco": (91, "instances"), "coco_kp": (2, "person_keypoints")}[args.dataset]
    num_classes, mode = {"coco": (3, "instances"), "coco_kp": (2, "person_keypoints")}[args.dataset]
    with_masks = "mask" in args.model
    #print(f"with masks: {with_masks}")
    ds = get_coco(
        root=args.data_path,
        image_set=image_set,
        transforms=get_transform(is_train, args),
        mode=mode,
        use_v2=args.use_v2,
        with_masks=with_masks,
    )
    return ds, num_classes


def get_transform(is_train, args):
    if is_train:
        return presets.DetectionPresetTrain(
            data_augmentation=args.data_augmentation, backend=args.backend, use_v2=args.use_v2
        )
    elif args.weights and args.test_only:
        weights = torchvision.models.get_weight(args.weights)
        trans = weights.transforms()
        return lambda img, target: (trans(img), target)
    else:
        return presets.DetectionPresetEval(backend=args.backend, use_v2=args.use_v2)


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="AI Nuclei Detection Training", add_help=add_help)

    #parser.add_argument("--data-path", default="/datasets01/COCO/022719/", type=str, help="dataset path")
    parser.add_argument(
        "--data-path", 
        default=r"/home/pod44433/disk/AI_nuclei_detection/train_data/new_balanced_data/sliced_coco", 
        type=str, 
        help="dataset path")
    parser.add_argument(
        "--dataset",
        default="coco",
        type=str,
        help="dataset name. Use coco for object detection and instance segmentation and coco_kp for Keypoint detection",
    )
    parser.add_argument(
        "--model", 
        default="maskrcnn_resnet50_fpn",
        #default="maskrcnn_resnet50_fpn_v2", 
        type=str, help="model name"
    )
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", 
        #default=2, 
        default=8, # for Faster/Mask R-CNN
        type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", 
                        #default=26,
                        default=100, # for Faster/Mask R-CNN for 100 epochs 
                        type=int, 
                        metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 4)"
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument(
        "--lr",
        default=0.005,
        #default=0.0025, # 0.02/8*1GPU
        type=float,
        help="initial learning rate, 0.02 is the default value for training on 8 gpus and 2 images_per_gpu",
    )
    parser.add_argument("--momentum", 
                        default=0.9, 
                        type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--lr-scheduler", 
        #default="multisteplr", 
        default="cosineannealinglr", 
        type=str, help="name of lr scheduler (default: multisteplr)"
    )
    parser.add_argument(
        "--lr-step-size", default=8, type=int, help="decrease lr every step-size epochs (multisteplr scheduler only)"
    )
    parser.add_argument(
        "--lr-steps",
        default=[16, 22], # default=[16, 22] for Faster/Mask R-CNN for 26 epochs
        # for Faster/Mask R-CNN for 100 epochs
        #default=[62, 85], #for Faster/Mask R-CNN for 100 epochs: (16/26)*100 = 61.5, (22/26)*100 = 84.6 - step size is (6/26)*100 = 23.1
        nargs="+",
        type=int,
        help="decrease lr every step-size epochs (multisteplr scheduler only)",
    )
    parser.add_argument(
        "--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma (multisteplr scheduler only)"
    )
    parser.add_argument("--print-freq", default=100, type=int, help="print frequency")
    output_dir = os.path.join(os.getcwd(), "run_" + time.strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--output-dir", default=output_dir, type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
    parser.add_argument("--aspect-ratio-group-factor", default=3, type=int)
    parser.add_argument("--rpn-score-thresh", default=None, type=float, help="rpn score threshold for faster-rcnn")
    parser.add_argument(
        "--trainable-backbone-layers", 
        #default=None, 
        default=5, # for Faster/Mask R-CNN
        type=int, help="number of trainable layers of backbone"
    )
    parser.add_argument(
        "--data-augmentation", 
        #default="hflip", 
        #default="maskrcnn",
        default="advanced_maskrcnn",
        #default="nuclei",
        #default="ssd",
        type=str, help="data augmentation policy (default: hflip)"
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument("--weights", 
                        default=None, 
                        type=str, help="the weights enum name to load")
    parser.add_argument("--weights-backbone", 
                        default=None, 
                        #default="ResNet50_Weights.IMAGENET1K_V1",
                        type=str, help="the backbone weights enum name to load")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # Use CopyPaste augmentation training parameter
    parser.add_argument(
        "--use-copypaste",
        action="store_true",
        help="Use CopyPaste data augmentation. Works only with data-augmentation='lsj'.",
    )

    parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive")
    parser.add_argument("--use-v2", action="store_true", help="Use V2 transforms")

    return parser




def main(args):
    if args.backend.lower() == "tv_tensor" and not args.use_v2:
        raise ValueError("Use --use-v2 if you want to use the tv_tensor backend.")
    if args.dataset not in ("coco", "coco_kp"):
        raise ValueError(f"Dataset should be coco or coco_kp, got {args.dataset}")
    if "keypoint" in args.model and args.dataset != "coco_kp":
        raise ValueError("Oops, if you want Keypoint detection, set --dataset coco_kp")
    if args.dataset == "coco_kp" and args.use_v2:
        raise ValueError("KeyPoint detection doesn't support V2 transforms yet")

    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    # save the arguments to a json file
    if args.output_dir:
        with open(os.path.join(args.output_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=3)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.use_deterministic_algorithms(True)

    # Data loading code
    print("\n")
    print("------------- Loading data -------------")
    dataset, num_classes = get_dataset(is_train=True, args=args)
    dataset_test, _ = get_dataset(is_train=False, args=args)
    print(f"Number of classes: {num_classes}")
    print(f"Number of training images: {len(dataset)}")
    print(f"Number of validation images: {len(dataset_test)}")
    print("------------- Data loaded -------------")

    print("\n")
    print("------- Creating data loaders ---------")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)

    train_collate_fn = utils.collate_fn
    if args.use_copypaste:
        if args.data_augmentation != "lsj":
            raise RuntimeError("SimpleCopyPaste algorithm currently only supports the 'lsj' data augmentation policies")

        train_collate_fn = copypaste_collate_fn

    data_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_sampler=train_batch_sampler, 
        num_workers=args.workers, 
        collate_fn=train_collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, 
        batch_size=1, 
        sampler=test_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
    )

    
    print("------- Data loaders created ---------")

    print("\n")
    print("---------- Creating model -------------")
    kwargs = {"trainable_backbone_layers": args.trainable_backbone_layers}
    if args.data_augmentation in ["multiscale", "lsj"]:
        kwargs["_skip_resize"] = True
    if "rcnn" in args.model:
        if args.rpn_score_thresh is not None:
            kwargs["rpn_score_thresh"] = args.rpn_score_thresh
    model = torchvision.models.get_model(
        args.model, weights=args.weights, weights_backbone=args.weights_backbone, num_classes=num_classes, **kwargs
    )
    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.norm_weight_decay is None:
        parameters = [p for p in model.parameters() if p.requires_grad]
    else:
        param_groups = torchvision.ops._utils.split_normalization_params(model)
        wd_groups = [args.norm_weight_decay, args.weight_decay]
        parameters = [{"params": p, "weight_decay": w} for p, w in zip(param_groups, wd_groups) if p]

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
        print("Using SGD optimizer with momentum {} and weight decay {}".format(args.momentum, args.weight_decay))
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
        print("Using AdamW optimizer with weight decay {}".format(args.weight_decay))
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD and AdamW are supported.")

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "multisteplr":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
        print(f"Using MultiStepLR scheduler with milestones {args.lr_steps} and gamma {args.lr_gamma}")
    elif args.lr_scheduler == "cosineannealinglr":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        print(f"Using CosineAnnealingLR scheduler with T_max {args.epochs}")
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only MultiStepLR and CosineAnnealingLR are supported."
        )
    print("---------- Model created -------------")

    if args.resume:
        #checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        checkpoint = torch.load(args.resume, map_location="cpu")
        if "model_state_dict" in checkpoint:
            model_without_ddp.load_state_dict(checkpoint["model_state_dict"])
        elif "model" in checkpoint:
            model_without_ddp.load_state_dict(checkpoint["model"])
        #model_without_ddp.load_state_dict(checkpoint["model"])
        #optimizer.load_state_dict(checkpoint["optimizer"])
        #lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        #args.start_epoch = checkpoint["epoch"] + 1
        #if "optimizer_state_dict" in checkpoint:
        #    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        #elif "optimizer" in checkpoint:
        #    optimizer.load_state_dict(checkpoint["optimizer"])
        #if "lr_scheduler_state_dict" in checkpoint:
        #    lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        #elif "lr_scheduler" in checkpoint:
        #    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        if "epoch" in checkpoint:
            args.start_epoch = checkpoint["epoch"] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        torch.backends.cudnn.deterministic = True
        iou_thresholds = [0.5, 0.75, 0.95]
        metrics_dir = os.path.join(args.output_dir, "metrics")
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            print(f"Created directory {args.output_dir}")
        if not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir)
            print(f"Created directory {metrics_dir}")
       
        epoch = 0
        evaluation_metrics_train = {0: {}}
        evaluation_metrics_val = {0: {}}


        print("Evaluating model on training and validation set")
        coco_evaluator = evaluate(model, data_loader, device=device)
        evaluation_metrics_train_epoch = utils.log_metrics(iou_thresholds, coco_evaluator, data_set="train", metric_logging_dir=metrics_dir)
        if epoch not in evaluation_metrics_train:
            evaluation_metrics_train[epoch] = {}
        evaluation_metrics_train[epoch] = evaluation_metrics_train_epoch
        utils.log_metrics_as_json(evaluation_metrics_train, "evaluation_metrics_train.json", metrics_dir) 

        coco_evaluator = evaluate(model, data_loader_test, device=device)
        evaluation_metrics_val_epoch = utils.log_metrics(iou_thresholds, coco_evaluator, data_set="val", metric_logging_dir=metrics_dir)
        if epoch not in evaluation_metrics_val:
            evaluation_metrics_val[epoch] = {}
        evaluation_metrics_val[epoch] = evaluation_metrics_val_epoch
        utils.log_metrics_as_json(evaluation_metrics_val, "evaluation_metrics_val.json", metrics_dir)

        return

    print("\n")
    print("---- Starting training process --------")
    start_time = time.time()

    training_losses = {i: {} for i in range(args.start_epoch, args.epochs)}
    evaluation_metrics_train = {i : {} for i in range(args.start_epoch, args.epochs)}

    validation_losses = {i : {} for i in range(args.start_epoch, args.epochs)}
    evaluation_metrics_val = {i : {} for i in range(args.start_epoch, args.epochs)}

    iou_thresholds = [0.5, 0.75, 0.95]
    metrics_dir = os.path.join(args.output_dir, "metrics")
    print(f"Storing checkpoints in {args.output_dir}")
    print(f"Storing metrics in {metrics_dir}")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)

    best_delta = 0.02848
    delta_for_training = 0.1 * best_delta # this is the delta for early stopping, allowing the training process more time to find better parameter settings before stopping.
    patience = 20
    early_stopping = EarlyStopping(patience=patience, delta=delta_for_training, # delta=0.1 means that the training process will stop if the validation loss does not decrease by 0.1 after 7 epochs
                                   path=args.output_dir)
    print(f"Using early stopping with delta {delta_for_training} and patience {patience}")
                                   

    for epoch in range(args.start_epoch, args.epochs):
    #for idx, epoch in enumerate(range(args.start_epoch, args.epochs)):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        """"""
        metric_logger = train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq, scaler)
        
        for key, value in metric_logger.meters.items():
            if epoch not in training_losses:
                training_losses[epoch] = {}
            if key not in training_losses[epoch]:
                training_losses[epoch][key] = []
            training_losses[epoch][key].append(value.global_avg)
        utils.log_metrics_as_json(training_losses, "training_losses.json", metrics_dir)

        lr_scheduler.step()
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "args": args,
                "epoch": epoch,
            }
            if args.amp:
                checkpoint["scaler"] = scaler.state_dict()
            #utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth")) 
            #early stopping saves the model
     
 

        # evaluate after every epoch on both training and validation set
        # only every 10th epoch
        #if epoch % 10 == 0:
            #print(f"Epoch {epoch} - Evaluating model on training and validation set")
            #coco_evaluator = evaluate(model, data_loader, device=device)
            #evaluation_metrics_train_epoch = utils.log_metrics(iou_thresholds, coco_evaluator, data_set="train", metric_logging_dir=metrics_dir)
            #evaluation_metrics_train[epoch] = evaluation_metrics_train_epoch
            #utils.log_metrics_as_json(evaluation_metrics_train, "evaluation_metrics_train.json", metrics_dir) 
    
            #coco_evaluator = evaluate(model, data_loader_test, device=device)
            #evaluation_metrics_val_epoch = utils.log_metrics(iou_thresholds, coco_evaluator, data_set="val", metric_logging_dir=metrics_dir)
            #evaluation_metrics_val[epoch] = evaluation_metrics_val_epoch
            #utils.log_metrics_as_json(evaluation_metrics_val, "evaluation_metrics_val.json", metrics_dir)

        # validate after every epoch -> sets model to train_mode to get losses but does not perform backpropagation
        metric_logger = validate(model, data_loader_test, device, args.print_freq, scaler)
        for key, value in metric_logger.meters.items():
            if epoch not in validation_losses:
                validation_losses[epoch] = {}
            if key not in validation_losses[epoch]:
                validation_losses[epoch][key] = []
            validation_losses[epoch][key].append(value.global_avg)
            if key == "loss":
                val_loss = value.global_avg
        utils.log_metrics_as_json(validation_losses, "validation_losses.json", metrics_dir)

        # early stopping
        early_stopping(val_loss=val_loss, model=model, optimizer=optimizer, epoch=epoch, lr_scheduler=lr_scheduler)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch}. Stopping training process.")
            print(f"Best validation loss: {early_stopping.val_loss_min}")
            break


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
