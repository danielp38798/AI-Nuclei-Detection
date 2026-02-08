#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    COCOEvaluator,
    DatasetEvaluators,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.data.datasets import register_coco_instances
from detectron2.data import build_detection_train_loader, DatasetMapper, build_detection_test_loader
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from detectron2.data import DatasetMapper
from copy_paste import CopyPaste, CocoDetectionCP
import albumentations as A
from detectron2.data import DatasetCatalog, MetadataCatalog
import numpy as np
import cv2
import torch    
from pycocotools import mask
from skimage import measure
import copy
from detectron2 import model_zoo
from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
import detectron2.utils.comm as comm
import time
import datetime
import matplotlib.pyplot as plt
from detectron2.config import get_cfg
import json
from fvcore.nn.precise_bn import get_bn_modules
from typing import Optional, List, Tuple
from argparse import Namespace

torch.manual_seed(42)

class MyMapper:
    """Mapper which uses `detectron2.data.transforms` augmentations"""

    def __init__(self, cfg, is_train: bool = True):

        self.is_train = is_train

        mode = "training" if is_train else "inference"
        #print(f"[MyDatasetMapper] Augmentations used in {mode}: {self.augmentations}")

        aug_list = [
            CopyPaste(blend=True, sigma=1, pct_objects_paste=0.5, p=0.5), #pct_objects_paste is a guess
            #A.Resize(400,400),#resize all images to fixed shape
                #CopyPaste(blend=True, sigma=1, pct_objects_paste=0.9, p=1.0) #pct_objects_paste is a guess
                
            ]
        

        transform = A.Compose(
                    aug_list, bbox_params=A.BboxParams(format="coco")
                )

        data = CocoDetectionCP(
            './data/coco/train2017',
            './data/coco/annotations/instances_train2017.json',
            transform
        )
        self.data = data

        self.train_metadata = MetadataCatalog.get("my_dataset_train2017")
        self.test_metadata = MetadataCatalog.get("my_dataset_val2017")

        self.dataset_dicts_train = DatasetCatalog.get("my_dataset_train2017")
        self.dataset_dicts_test = DatasetCatalog.get("my_dataset_val2017")

    def __call__(self, dataset_dict):

        #print(f"Processing image {dataset_dict['image_id']}")
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        img_id = dataset_dict['image_id']
        data = self.data

        data_id_to_num = {i:q for q,i in enumerate(data.ids)}
        ALL_IDS = list(data_id_to_num.keys())
        
        dataset_dicts_train = self.dataset_dicts_train

        dataset_dicts_train = [i for i in dataset_dicts_train if i['image_id'] in ALL_IDS]
        BOX_MODE = dataset_dicts_train[0]['annotations'][0]['bbox_mode']
        #print(f"BOX_MODE: {BOX_MODE}")
        aug_sample = data[data_id_to_num[img_id]]
        image = aug_sample['image']
        image =  cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
        bboxes = aug_sample['bboxes']
        box_classes = np.array([b[-2] for b in bboxes])
        boxes = np.stack([b[:4] for b in bboxes], axis=0)
        mask_indices = np.array([b[-1] for b in bboxes])
        masks = aug_sample['masks']
        annos = []
        
        for enum,index in enumerate(mask_indices):
            curr_mask = masks[index]
            
            fortran_ground_truth_binary_mask = np.asfortranarray(curr_mask)
            encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
            ground_truth_area = mask.area(encoded_ground_truth)
            ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
            contours = measure.find_contours(curr_mask, 0.5)
            
            annotation = {
        "segmentation": [],
        "iscrowd": 0,
        "bbox": ground_truth_bounding_box.tolist(), 
        "category_id": self.train_metadata.thing_dataset_id_to_contiguous_id[box_classes[enum]] ,
        "bbox_mode":BOX_MODE
    }
            for contour in contours:
                contour = np.flip(contour, axis=1)
                segmentation = contour.ravel().tolist()
                if len(segmentation) > 4:
                    annotation["segmentation"].append(segmentation)
                
            annos.append(annotation)

        image_shape = image.shape[:2]  # h, w
        instances = utils.annotations_to_instances(annos, image_shape)
        dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict

class LossEvalHook(HookBase):
    def __init__(self, eval_period: int, model: torch.nn.Module, data_loader: torch.utils.data.DataLoader):
        """
        Initialize the LossEvalHook.

        Args:
            eval_period (int): The period (in iterations) to run the evaluation.
            model (torch.nn.Module): The model to evaluate.
            data_loader (torch.utils.data.DataLoader): The data loader for the evaluation dataset.
        """
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
        self._early_stopping_counter = 0
        self.best_loss = np.inf  # set to infinity so that the first validation loss is always better
    
    def _do_loss_eval(self) -> list:
        """
        Run an evaluation when the training loop calls for it.

        Returns:
            list: A list of loss values for each batch in the evaluation dataset.
        """
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)
            
        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):            
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()

        return losses
            
    def _get_loss(self, data: dict) -> float:
        """
        Calculate the loss for a batch of data.

        Args:
            data (dict): A batch of data.

        Returns:
            float: The total loss for the batch.
        """
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced
    
    def after_step(self) -> None:
        """
        Hook to be called after each training step. It performs loss evaluation and checks for early stopping.

        This method is called after each training iteration. It evaluates the loss on the validation set
        and checks if early stopping criteria are met. If the criteria are met, it stops the training.
        """
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
            self.check_if_early_stop()
        self.trainer.storage.put_scalars(timetest=12)

    def check_if_early_stop(self) -> None:
        """
        Check if early stopping should be activated based on validation loss.

        This method checks the history of validation losses and compares the current validation loss
        with the best validation loss observed so far. If the current validation loss is better, it resets
        the early stopping counter. If not, it increments the counter. If the counter exceeds the patience
        threshold, it activates early stopping.

        Returns:
            None
        """
        print("Checking if early stopping should be activated...")
        validation_losses = self.trainer.storage.history('validation_loss').values()

        if len(validation_losses) > 0:
            current_loss = validation_losses[-1][0]
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self._early_stopping_counter = 0
                print(f"Validation loss improved to {self.best_loss}.")
                print(f"Early stopping counter reset to {self._early_stopping_counter}.")
            else:
                print(f"Validation loss did not improve. Current loss: {current_loss}. Best loss: {self.best_loss}.")
                self._early_stopping_counter += 1
                print(f"Early stopping counter incremented to {self._early_stopping_counter}.")
                if self._early_stopping_counter >= self.trainer.cfg.SOLVER.PATIENCE:
                    print(f"Early stopping activated. No improvement in validation loss for {self.trainer.cfg.SOLVER.PATIENCE} epochs.")
                    self.trainer._should_stop = True

class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_train_loader(cls, cfg, sampler=None):
        """
        Build the training data loader.

        Args:
            cfg (CfgNode): Configuration node.
            sampler (Optional): Sampler for the data loader.

        Returns:
            DataLoader: A data loader for the training dataset.
        """
        return build_detection_train_loader(
            cfg, mapper=DatasetMapper(cfg, True), sampler=sampler
        )

    @classmethod
    def build_evaluator(cls, cfg, dataset_name: str, output_folder: str = None) -> DatasetEvaluators:
        """
        Build the evaluator for the given dataset.

        Args:
            cfg (CfgNode): Configuration node.
            dataset_name (str): Name of the dataset.
            output_folder (Optional[str]): Output folder for the evaluation results.

        Returns:
            DatasetEvaluators: An evaluator for the dataset.
        """
        return build_evaluator(cfg, dataset_name, output_folder)
    
    @classmethod
    def test_with_TTA(cls, cfg, model) -> OrderedDict:
        """
        Run inference with test-time augmentation (TTA).

        Args:
            cfg (CfgNode): Configuration node.
            model (nn.Module): The model to evaluate.

        Returns:
            OrderedDict: Evaluation results with TTA.
        """
        logger = logging.getLogger("detectron2.trainer")
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res
    
    @classmethod
    def test_with_trainset(cls, cfg, model) -> OrderedDict:
        """
        Run inference on the training set.

        Args:
            cfg (CfgNode): Configuration node.
            model (nn.Module): The model to evaluate.

        Returns:
            OrderedDict: Evaluation results on the training set.
        """
        logger = logging.getLogger("detectron2.trainer")
        logger.info("Running inference with trainset ...")
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_on_trainset")
            )
            for name in cfg.DATASETS.TRAIN
        ]
        res = cls.test_on_trainset(cfg, model, evaluators)
        res = OrderedDict({k + "_train": v for k, v in res.items()})
        return res

    def build_hooks(self) -> list:
        """
        Build a list of hooks for the training process.

        Returns:
            list: A list of hooks to be used during training.
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(),
            (
                hooks.PreciseBN(
                    # Run at the same freq as (but before) evaluation.
                    cfg.TEST.EVAL_PERIOD,
                    self.model,
                    # Build a new data loader to not affect training
                    self.build_train_loader(cfg),
                    cfg.TEST.PRECISE_BN.NUM_ITER,
                )
                if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
                else None
            ),
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))
        ret.append(LossEvalHook(cfg.TEST.EVAL_PERIOD, self.model, build_detection_test_loader(self.cfg, self.cfg.DATASETS.TEST[0], DatasetMapper(self.cfg, True))))
        ret.append(hooks.EvalHook(eval_period=cfg.TEST.EVAL_PERIOD, eval_function=lambda: Trainer.test_with_trainset(cfg, self.model)))
        
        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        ret.append(hooks.BestCheckpointer(eval_period=cfg.TEST.EVAL_PERIOD, 
                                        checkpointer=self.checkpointer,
                                        val_metric="segm/AP50"))
        
        return ret
    

    def build_evaluator(cfg, dataset_name: str, output_folder: str = None) -> DatasetEvaluators:
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.

        Args:
            cfg (CfgNode): Configuration node.
            dataset_name (str): Name of the dataset.
            output_folder (Optional[str]): Output folder for the evaluation results.

        Returns:
            DatasetEvaluators: An evaluator for the dataset.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))

        return DatasetEvaluators(evaluator_list)


    def plot_experiment_metrics(experiment_folder: str) -> None:
        """
        Plot the experiment metrics from a given experiment folder.

        Args:
            experiment_folder (str): Path to the experiment folder containing the metrics.json file.

        Returns:
            None
        """
        def load_json_arr(json_path: str) -> list:
            """
            Load a JSON array from a file.

            Args:
                json_path (str): Path to the JSON file.

            Returns:
                list: A list of JSON objects.
            """
            lines = []
            with open(json_path, 'r') as f:
                for line in f:
                    lines.append(json.loads(line))
            return lines

        experiment_metrics = load_json_arr(experiment_folder + '/metrics.json')

        iteration = []
        total_loss = []
        validation_loss = []
        for x in experiment_metrics:
            if "total_loss" in x:
                iteration.append(x['iteration'])
                total_loss.append(x['total_loss'])
                if 'validation_loss' in x:
                    validation_loss.append(x['validation_loss'])

        # plot the total loss and validation loss
        plt.figure()
        plt.plot(iteration, total_loss, label='total_loss')
        if len(validation_loss) > 0:
            plt.plot(iteration, validation_loss, label='validation_loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Total Loss and Validation Loss vs Iteration')

        plt.legend(['total_loss', 'validation_loss'], loc='upper left')
        #plt.show()
        plt.savefig(experiment_folder + '/metrics_plot.png')


def setup(args) -> get_cfg:
    """
    Create and configure the Detectron2 configuration object.

    This function sets up the configuration for the Detectron2 model training and evaluation.
    It merges configurations from a file and command-line arguments, registers custom datasets,
    and sets various training parameters.

    Args:
        args (Namespace): Command-line arguments containing configuration file path and options.

    Returns:
        CfgNode: The configured Detectron2 configuration object.
    """
    cfg = get_cfg()

    # Check if config file is given in args
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    else:
        yaml_path = "./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        cfg.merge_from_file(yaml_path)
        print("No config file given in args. Using default config file:", yaml_path)

    # Check if options are given in args
    if args.opts:
        print("Options given in args:", args.opts)
        cfg.merge_from_list(args.opts)

    # Register the custom COCO dataset
    register_coco_instances("my_dataset_train2017", {}, "./data/coco/train2017/instances_train2017_coco.json", "./data/coco/train2017")
    register_coco_instances("my_dataset_val2017", {}, "./data/coco/val2017/instances_val2017_coco.json", "./data/coco/val2017")

    train_metadata = MetadataCatalog.get("my_dataset_train2017")
    print(f"Metadata for train: {train_metadata}")
    test_metadata = MetadataCatalog.get("my_dataset_val2017")
    print(f"Metadata for test: {test_metadata}")

    cfg.DATASETS.TRAIN = ("my_dataset_train2017",)
    cfg.DATASETS.VAL = ("my_dataset_val2017",)
        
    cfg.INPUT.MIN_SIZE_TEST = 400
    cfg.INPUT.MAX_SIZE_TEST = 400
    cfg.INPUT.MIN_SIZE_TRAIN = 400
    cfg.INPUT.MAX_SIZE_TRAIN = 400

    cfg.INPUT.FORMAT = 'BGR'
    cfg.DATASETS.TEST = ("my_dataset_val2017",)
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2 
    cfg.SOLVER.BASE_LR = 0.002
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.STEPS = (4000,)

    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 3
    cfg.SOLVER.WARMUP_ITERS = 500
    cfg.SOLVER.WARMUP_METHOD = "linear"

    # Get the number of training images
    dataset_dicts_train = DatasetCatalog.get("my_dataset_train2017")
    num_train_images = len(dataset_dicts_train)
    print(f"Number of training images: {num_train_images}")
    dataset_dicts_test = DatasetCatalog.get("my_dataset_val2017")
    num_test_images = len(dataset_dicts_test)
    print(f"Number of test images: {num_test_images}")

    batch_size = 2  # Replace with your actual batch size
    num_epochs = 25  # Replace with your desired number of epochs

    # Calculate max_iter
    iter_per_epoch = num_train_images // batch_size
    max_iter = iter_per_epoch * num_epochs

    # Set this in your config
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.IMS_PER_BATCH = batch_size

    print(f"Batch size set to: {batch_size}")
    print(f"Number of epochs set to: {num_epochs}")
    print(f"Max iteration number set to: {max_iter}")
    print(f"Iteration per epoch: {iter_per_epoch}")

    # Set evaluation period to be every epoch
    cfg.TEST.EVAL_PERIOD = iter_per_epoch
    print(f"Evaluation period set to: {cfg.TEST.EVAL_PERIOD}")

    # Dynamically set patience to X num epochs
    cfg.SOLVER.PATIENCE = 8  # Early stopping will occur if there is no improvement in validation loss for 8 eval periods
    print(f"Patience set to: {cfg.SOLVER.PATIENCE} epochs (representing {cfg.SOLVER.PATIENCE * cfg.TEST.EVAL_PERIOD} iterations)")

    cfg.EVAL_TRAINSET = True

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.RETINANET.NUM_CLASSES = 1
    cfg.SOLVER.CHECKPOINT_PERIOD = iter_per_epoch  # Checkpoints every epoch

    cfg.TEST.AUG.ENABLED = False
    cfg.TEST.AUG.MIN_SIZES = (400,)
    cfg.TEST.AUG.MAX_SIZE = 400
    cfg.TEST.AUG.FLIP = False

    cfg.OUTPUT_DIR = f'./output_{time.strftime("%Y-%m-%d-%H-%M-%S")}'
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args: Namespace) -> Optional[OrderedDict]:
    """
    Main function to set up and run the training or evaluation process.

    Args:
        args (Namespace): Command-line arguments containing configuration options.

    Returns:
        Optional[OrderedDict]: Evaluation results if in evaluation mode, otherwise None.
    """
    # Check if CUDA is available
    if torch.cuda.is_available():
        # Get the number of available GPUs
        num_gpus = torch.cuda.device_count()
        print(f"Number of available GPUs: {num_gpus}")

        # List all GPUs
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

        # Select the second GPU (index 1)
        if num_gpus > 1:
            torch.cuda.set_device(1)
            print(f"\nSelected GPU: {torch.cuda.get_device_name(1)}")
        else:
            print("\nThere is no second GPU available.")
    else:
        print("CUDA is not available on this system.")
        
    cfg = setup(args)
    print(f"cfg.EVAL_TRAINSET: {cfg.EVAL_TRAINSET}")
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.EVAL_TRAINSET:
            print("Evaluating on trainset...")
            res.update(Trainer.test_with_trainset(cfg, model))
        if cfg.TEST.AUG.ENABLED:
            print("Running inference with test-time augmentation ...")
            res.update(Trainer.test_with_TTA(cfg, model))

        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        print("Registering hooks for TTA...")
        hooks = trainer.build_hooks_list(model=trainer.model, cfg=cfg, eval_period=cfg.TEST.EVAL_PERIOD, checkpointer=trainer.checkpointer)
        trainer.register_hooks(hooks)
        print("Hooks registered for TTA.")
    else:
        print("Registering hooks for regular evaluation...")
        # store the model with the best validation mAP
        trainer.register_hooks([
            hooks.BestCheckpointer(eval_period=cfg.TEST.EVAL_PERIOD, 
                                checkpointer=trainer.checkpointer,
                                val_metric="segm/AP50")])
        print("Hooks registered for regular evaluation.")
    
    return trainer.train()

def invoke_main() -> None:
    """
    Parse command-line arguments and launch the main training or evaluation process.

    This function parses the command-line arguments using the default argument parser,
    sets the number of GPUs to 1, and then launches the main function with the parsed arguments.
    It uses the `launch` function from Detectron2 to handle distributed training.

    Args:
        None

    Returns:
        None
    """
    args = default_argument_parser().parse_args()
    args.num_gpus = 1
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


if __name__ == "__main__":
    print("Invoking AI training script...")
    #invoke_main()  
    args = default_argument_parser().parse_args()
    main(args)



