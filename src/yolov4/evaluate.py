from __future__ import division

from yolov4.datasets import ListDataset
from yolov4.bbox_ops import box_iou, change_box_order, non_max_suppression

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable

from tqdm import tqdm
import numpy as np


def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    TENSOR = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    model.eval()
    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = change_box_order(targets[:, 2:], 'xywh2xyxy)
        targets[:, 2:] *= img_size
        # Forward prop and nms to lock in prediction
        imgs = Variable(imgs.type(Tensor), requires_grad=False)
        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
        # Evaluate this batch
        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)
    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
    return precision, recall, AP, f1, ap_class


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    # Find unique classes
    unique_classes = np.unique(target_cls)
    # Create Precision-Recall curve and compute AP for each class
    ap, precision, recall = [], [], []
    for cls in tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == cls
        n_gt = (target_cls == cls).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects
        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            recall.append(0)
            precision.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()
            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            recall.append(recall_curve[-1])
            # Precision
            precision_curve = tpc / (tpc + fpc)
            precision.append(precision_curve[-1])
            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))
    # Compute F1 score (harmonic mean of precision and recall)
    precision, recall, ap = np.array(precision), np.array(recall), np.array(ap)
    f1 = 2 * precision * recall / (precision + recall + 1e-16)
    return precision, recall, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]
    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):
        if outputs[sample_i] is None:
            continue
        output = outputs[sample_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]
            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break
                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue
                iou, box_index = box_iou(pred_box.unsqueeze(0), target_boxes, order="xyxy").max(0)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics
