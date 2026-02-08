import datetime
import errno
import os
import time
from collections import defaultdict, deque

import torch
import torch.distributed as dist

import numpy as np
import json
import matplotlib.pyplot as plt



class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        # avoid divide by zero
        if self.count > 0:
            return self.total / self.count
        else:
            return 0.0
        #return self.total / self.count

    @property
    def max(self):
        # avoid ValueError: max() arg is an empty sequence
        if len(self.deque) > 0:
            return max(self.deque)
        else:
            return 0.0
        #return max(self.deque)

    @property
    def value(self):
        #return self.deque[-1]
        return self.deque[-1] if len(self.deque) > 0 else 0.0

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.inference_mode():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)")


def collate_fn(batch):
    return tuple(zip(*batch))


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)

def log_metrics_as_json(metrics, output_name, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    out_file = os.path.join(output_dir, output_name)
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=4, cls=NumpyEncoder)

def calculate_pr_f1_tp_fp_fn(coco_eval, iou_type, iou_threshold, verbose=False):
    total_tp = 0
    total_fp = 0
    total_gt = 0
    epsilon = 0.00000000001 # To avoid divide by 0 case

    # Select the index related to IoU = 0.5
    #iou_treshold_index = 0 # iouThrs - [.5:.05:.95] T=10 IoU thresholds for evaluation, so, index 0 is threshold=0.5
    if iou_threshold == 0.5:
        iou_treshold_index = 0
    elif iou_threshold == 0.75:
        iou_treshold_index = 5
    elif iou_threshold == 0.95:
        iou_treshold_index = 9
    

    for image_evaluation_dict in coco_eval.evalImgs:

        if image_evaluation_dict is None:
            continue
        # All the detections from the model, it is a numpy of True/False
        detection_ignore = image_evaluation_dict["dtIgnore"][iou_treshold_index]

        # Here we consider the detection that we can not ignore (we use the not operator on every element of the array)
        mask = ~detection_ignore
  
        n_ignored = detection_ignore.sum()
        #print(f"Ignore count from detected bboxes from [image: {image_evaluation_dict['image_id']}]: {n_ignored}")

        # And finally we calculate tp, fp and the total positives (n_gt)
        tp = (image_evaluation_dict["dtMatches"][iou_treshold_index][mask] > 0).sum()
        fp = (image_evaluation_dict["dtMatches"][iou_treshold_index][mask] == 0).sum()
        n_gt = len(image_evaluation_dict["gtIds"]) - image_evaluation_dict["gtIgnore"].astype(int).sum()

        per_example_precision = tp / max((tp + fp), epsilon)
        per_example_recall = tp / max(n_gt, epsilon)
        per_example_f1 = 2 * per_example_precision * per_example_recall / max((per_example_precision + per_example_recall), epsilon)

        total_tp += tp
        total_fp += fp
        total_gt += n_gt


    precision = total_tp / max((total_tp + total_fp), epsilon)
    recall = total_tp / max(total_gt, epsilon)
    f1 = 2 * precision * recall / max((precision + recall), epsilon)

    average_precision = coco_eval.stats[0]
    average_recall = coco_eval.stats[8]

    if verbose:

        print(f"\n======= FINAL METRICS FOR IoU TYPE: {iou_type} and IoU THRESHOLD: {iou_threshold} =======")
        print(f"PRECISION: {precision}")
        print(f"RECALL: {recall}")
        print(f"F1_SCORE: {f1}")
        print(f"AVERAGE_PRECISION: {average_precision}")
        print(f"AVERAGE_RECALL: {average_recall}")
        print(f"TOTAL_TP: {total_tp}") # True Positives
        print(f"TOTAL_FP: {total_fp}") # False Positives
        print(f"TOTAL FN: {total_gt - total_tp}") # False Negatives
        print(f"TOTAL_GT: {total_gt}")
        print(f"=============================================")


    return {"precision": precision, 
            "recall": recall, 
            "f1_score": f1, 
            "average_precision": average_precision, 
            "average_recall": average_recall, 
            "total_tp": total_tp, 
            "total_fp": total_fp, 
            "total_fn": total_gt - total_tp,
            "total_gt": total_gt}

def log_metrics(iou_thresholds, coco_evaluator, data_set="train", metric_logging_dir="./metrics"):
    """
    logs average precision, recall, f1 score, precision-recall curve and other metrics for each category and each IoU threshold
    """
    metrics_dict = {}
    for iou_type, coco_eval in coco_evaluator.coco_eval.items(): # bbox, segm, keypoints
        cocoEval = coco_evaluator.coco_eval[iou_type] # bbox, segm, keypoints  

        print(f"Logging {data_set} metrics for {iou_type} at IoU thresholds {iou_thresholds}...")
        AP = cocoEval.stats[0]
        AP_50 = cocoEval.stats[1]
        AP_75 = cocoEval.stats[2]
        AP_small = cocoEval.stats[3]
        AP_medium = cocoEval.stats[4]
        AP_large = cocoEval.stats[5]
        AR_1 = cocoEval.stats[6]
        AR_10 = cocoEval.stats[7]
        AR_100 = cocoEval.stats[8]
        AR_small = cocoEval.stats[9]
        AR_medium = cocoEval.stats[10]
        AR_large = cocoEval.stats[11]

        # Get category IDs
        cocoGt = cocoEval.cocoGt
        category_ids = cocoGt.getCatIds()
        category_ids = [c_id-1 for c_id in category_ids]

        for iou_threshold in iou_thresholds:
            # Extract precision values for the specific IoU threshold
            evaluation_dict = calculate_pr_f1_tp_fp_fn(cocoEval, iou_type, iou_threshold, False)

            # store the precision and recall values in the metrics_dict
            if iou_type not in metrics_dict.keys():
                metrics_dict[iou_type] = {}
            if iou_threshold not in metrics_dict [iou_type].keys():
                metrics_dict[iou_type][iou_threshold] = {}
            
            iou_index = np.where(np.isclose(cocoEval.params.iouThrs, iou_threshold))[0][0]
            # Loop over each category and plot the precision-recall curve
            for category_id in category_ids:
                # Extract precision values for the specific category
                precision_values = cocoEval.eval['precision'][iou_index, :, category_id, 0, -1] 
                # [TxRxKxAxM]; T=num_thresholds, R=num_recall_values, K=num_categories, A=num_area_ranges, M=num_max_dets
                # here, we are only interested in the precision values for the specific category and IoU threshold,
                recall_values = cocoEval.params.recThrs

                # store the precision and recall values in the metrics_dict
                if iou_type not in metrics_dict .keys():
                    metrics_dict[iou_type] = {}
                if iou_threshold not in metrics_dict[iou_type].keys():
                    metrics_dict[iou_type][iou_threshold] = {}
                if category_id not in metrics_dict[iou_type][iou_threshold].keys():
                    metrics_dict[iou_type][iou_threshold][category_id] = {}

                if evaluation_dict is not None:
                    metrics_dict[iou_type][iou_threshold][category_id] = {
                                                                    "precision_values": precision_values.tolist(), 
                                                                    "recall_values": recall_values.tolist(),
                                                                    "precision": evaluation_dict["precision"],
                                                                    "recall": evaluation_dict["recall"],
                                                                    "f1_score": evaluation_dict["f1_score"],
                                                                    "average_precision": evaluation_dict["average_precision"],
                                                                    "average_recall": evaluation_dict["average_recall"],
                                                                    "total_tp": evaluation_dict["total_tp"],
                                                                    "total_fp": evaluation_dict["total_fp"],
                                                                    "total_fn": evaluation_dict["total_fn"],
                                                                    "total_gt": evaluation_dict["total_gt"],
                                                                    "AP": AP,
                                                                    "AP_50": AP_50,
                                                                    "AP_75": AP_75,
                                                                    "AP_small": AP_small,
                                                                    "AP_medium": AP_medium,
                                                                    "AP_large": AP_large,
                                                                    "AR_1": AR_1,
                                                                    "AR_10": AR_10,
                                                                    "AR_100": AR_100,
                                                                    "AR_small": AR_small,
                                                                    "AR_medium": AR_medium,
                                                                    "AR_large": AR_large
                                                                    }
                else:
                    metrics_dict[iou_type][iou_threshold][category_id] = {
                                                                    "precision_values": precision_values.tolist(), 
                                                                    "recall_values": recall_values.tolist(),
                                                                    "precision": 0,
                                                                    "recall": 0,
                                                                    "f1_score": 0,
                                                                    "average_precision": AP,
                                                                    "average_recall": 0,
                                                                    "total_tp": 0,
                                                                    "total_fp": 0,
                                                                    "total_fn": 0,
                                                                    "total_gt": 0,
                                                                    "AP": AP,
                                                                    "AP_50": AP_50,
                                                                    "AP_75": AP_75,
                                                                    "AP_small": AP_small,
                                                                    "AP_medium": AP_medium,
                                                                    "AP_large": AP_large,
                                                                    "AR_1": AR_1,
                                                                    "AR_10": AR_10,
                                                                    "AR_100": AR_100,
                                                                    "AR_small": AR_small,
                                                                    "AR_medium": AR_medium,
                                                                    "AR_large": AR_large
                                                                    }

                plt.figure(figsize=(7, 7))
                plt.plot(recall_values, precision_values, label="Category ID {}".format(category_id))
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curve for Category ID {} for IoU = {} ({})'.format(category_id, iou_threshold, iou_type))
                plt.legend()
                plt.grid(True)
                # Save the plot to a temporary file
                #plot_filename = 'pr_curve_{}_iou_{}_cat_id_{}_epoch_{}.png'.format(iou_type, iou_threshold, category_id, epoch)
                plot_filename = 'pr_curve_{}_iou_{}_cat_id_{}.png'.format(iou_type, iou_threshold, category_id)
                
                if data_set == "train":
                    temp_file_path = os.path.join(metric_logging_dir, "train", "precision_recall", plot_filename)
                else:
                    temp_file_path = os.path.join(metric_logging_dir, "val", "precision_recall", plot_filename)

                if not os.path.exists(os.path.dirname(temp_file_path)):
                    os.makedirs(os.path.dirname(temp_file_path))
                plt.savefig(temp_file_path)
                plt.close()         

    return metrics_dict

