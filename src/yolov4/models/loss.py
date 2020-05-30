import torch
import torch.nn as nn

from yolov4.utils import build_targets


class YOLOLoss(nn.Module):
    def __init__(self, ignore_thres=0.5, obj_scale=1, noobj_scale=100):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = obj_scale
        self.noobj_scale = noobj_scale
        self.ignore_thres = ignore_thres

    def localization_loss(self, pred_boxes, target_box, obj_mask):
        x, y, w, h = pred_boxes
        tx, ty, tw, th = target_box
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

    def get_metrics(self, pred_conf, tconf, class_mask, obj_mask, noobj_mask, iou_scores):
        # Metrics
        cls_acc = 100 * class_mask[obj_mask].mean()
        conf_obj = pred_conf[obj_mask].mean()
        conf_noobj = pred_conf[noobj_mask].mean()
        conf50 = (pred_conf > 0.5).float()
        iou50 = (iou_scores > 0.5).float()
        iou75 = (iou_scores > 0.75).float()
        detected_mask = conf50 * class_mask * tconf
        precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
        recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
        recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)
        return cls_acc, recall50, recall75, precision, conf_obj, conf_noobj

    def forward(self, output, targets, anchors):
        pred_boxes = output[..., :4]
        pred_conf = output[..., 4]
        pred_cls = output[..., 5:]
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
        loss_loc = self.localization_loss(pred_boxes, targets, obj_mask)
        loss_conf, loss_cls = self.classification_loss(
            self, pred_conf, tconf, pred_cls, tcls, obj_mask, noobj_mask
        )
        total_loss = loss_loc + loss_conf + loss_cls
        # Get metrics
        cls_acc, recall50, recall75, precision, conf_obj, conf_noobj = self.get_metrics(pred_conf, tconf, class_mask, obj_mask, noobj_mask, iou_scores)
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
