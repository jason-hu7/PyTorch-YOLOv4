import torch


def change_box_order(boxes, order):
    """Change box order between (xmin,ymin,xmax,ymax) and (xcenter,ycenter,width,height).
    Args:
      boxes: (tensor) bounding boxes, sized [N,4].
      order: (str) either 'xyxy2xywh' or 'xywh2xyxy'.
    Returns:
      (tensor) converted bounding boxes, sized [N,4].
    """
    assert order in ["xyxy2xywh", "xywh2xyxy"]
    a = boxes[:, :2]
    b = boxes[:, 2:]
    if order == "xyxy2xywh":
        return torch.cat([(a + b) / 2, b - a + 1], 1)
    return torch.cat([a - b / 2, a + b / 2], 1)


def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.
    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format
    Returns:
        area (Tensor[N]): area for each box
    Reference:
        https://github.com/pytorch/vision/blob/25694e07199c273ca1a980e6013b7f451f5a3cb0/torchvision/ops/boxes.py#L136
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_wh_iou(wh1, wh2):
    """ bbox iou with just width and height assuming they have the same centers"""
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def box_iou(boxes1, boxes2, order="xyxy"):
    """Compute the intersection over union of two set of boxes.
    The default box order is (xmin, ymin, xmax, ymax).
    Args:
      boxes1: (tensor) bounding boxes, sized [N,4].
      boxes2: (tensor) bounding boxes, sized [M,4].
      order: (str) box order, either 'xyxy' or 'xywh'.
    Return:
      (tensor) iou, sized [N,M].
    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if order == "xywh":
        boxes1 = change_box_order(boxes1, "xywh2xyxy")
        boxes2 = change_box_order(boxes2, "xywh2xyxy")

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def box_diou(boxes1, boxes2, beta=0.6):
    """Compute the intersection over union of two set of boxes.
    The default box order is (xmin, ymin, xmax, ymax).
    Args:
      boxes1: (tensor) bounding boxes, sized [N,4].
      boxes2: (tensor) bounding boxes, sized [M,4].
      order: (str) box order, either 'xyxy' or 'xywh'.
    Return:
      (tensor) iou, sized [N,M].
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    # Calculate the diagonal length of the smallest bbox covering the 2 boxes
    clt=torch.min(boxes1[:, None, :2], boxes2[:, :2])
    crb=torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    c=((crb-clt)**2).sum(dim=2)
    # Calculate the euclidean distance between central points of boxes1 and boxes2
    x1=(boxes1[:, None, 0] + boxes1[:, None, 2])/2
    y1=(boxes1[:, None, 1] + boxes1[:, None, 3])/2
    x2=(boxes2[:, None, 0] + boxes2[:, None, 2])/2
    y2=(boxes2[:, None, 1] + boxes2[:, None, 3])/2
    d=(x1-x2.t())**2 + (y1-y2.t())**2
    return inter / (area1[:, None] + area2 - inter) - (d / c) ** beta


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = change_box_order(prediction[..., :4], order="xywh2xyxy")
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat(
            (image_pred[:, :5], class_confs.float(), class_preds.float()), 1
        )
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = (
                box_iou(detections[0, :4].unsqueeze(0), detections[:, :4], order="xyxy")
                > nms_thres
            )
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(
                0
            ) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)
    return output
