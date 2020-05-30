import torch
import torch.nn as nn

from yolov4.utils import build_targets, to_cpu
from yolov4.evaluate import get_metrics

class YOLOLoss(nn.Module):
    def __init__(self, num_classes=80, ignore_thres=0.5, obj_scale=1, noobj_scale=100):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = obj_scale
        self.noobj_scale = noobj_scale
        self.ignore_thres = ignore_thres
        self.num_classes = num_classes

    def localization_loss(self, direct_boxes, target_boxes, obj_mask):
        """ calculate localization (regression) loss, localization loss is the sum of all 4 coordinates.
        Args:
            direct_boxes: bounding boxes in the direct predictions (x, y, w, h)
            target_boxes: bounding boxes from targets (tx, ty, tw, th)
        Returns:
            loc_loss: total localization loss
        """
        x, y, w, h = direct_boxes
        tx, ty, tw, th = target_boxes
        loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
        loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
        loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
        loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
        loc_loss = loss_x + loss_y + loss_w + loss_h
        return loc_loss

    def classification_loss(
        self, pred_conf, tconf, pred_cls, tcls, obj_mask, noobj_mask
    ):
        loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
        loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
        loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
        loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
        return loss_conf, loss_cls

    def forward(self, output, targets, direct_boxes, anchors):
        nB, nA, nG, _ = direct_boxes[0].shape
        pred_boxes = output[..., :4].view(nB, -1, nG, nG, 4)
        pred_conf = output[..., 4].view(nB, -1, nG, nG)
        pred_cls = output[..., 5:].view(nB, -1, nG, nG, self.num_classes)
        (
            iou_scores,
            class_mask,
            obj_mask,
            noobj_mask,
            tx,
            ty,
            tw,
            th,
            tcls,
            tconf,
        ) = build_targets(
            pred_boxes=pred_boxes,
            pred_cls=pred_cls,
            target=targets,
            anchors=anchors,
            ignore_thres=self.ignore_thres,
        )
        loss_loc = self.localization_loss(direct_boxes, (tx, ty, tw, th), obj_mask)
        loss_conf, loss_cls = self.classification_loss(
            pred_conf, tconf, pred_cls, tcls, obj_mask, noobj_mask
        )
        total_loss = loss_loc + loss_conf + loss_cls
        # Get metrics
        cls_acc, recall50, recall75, precision, conf_obj, conf_noobj = get_metrics(pred_conf, tconf, class_mask, obj_mask, noobj_mask, iou_scores)
        self.metrics = {
            "loss": to_cpu(total_loss).item(),
            "loc" : to_cpu(loss_loc).item(),
            "conf": to_cpu(loss_conf).item(),
            "cls": to_cpu(loss_cls).item(),
            "cls_acc": to_cpu(cls_acc).item(),
            "recall50": to_cpu(recall50).item(),
            "recall75": to_cpu(recall75).item(),
            "precision": to_cpu(precision).item(),
            "conf_obj": to_cpu(conf_obj).item(),
            "conf_noobj": to_cpu(conf_noobj).item(),
        }
        return loss_loc, loss_conf, loss_cls
