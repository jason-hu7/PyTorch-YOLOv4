import torch
import torch.nn as nn


class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100

    def localization_loss(self, pred_box, target_box, obj_mask):
        x, y, w, h = pred_box
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
        cls_loss = loss_conf + loss_cls

    def forward(self):
        loc_loss = self.localization_loss(self, pred_box, target_box, obj_mask)
        cls_loss = self.classification_loss(
            self, pred_conf, t_conf, pred_cls, tcls, obj_mask, noobj_mask
        )
        return loc_loss + cls_loss
