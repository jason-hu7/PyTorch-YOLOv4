import torch
import torch.nn as nn
from yolov4.utils import build_targets, to_cpu


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super().__init__()


class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim=416):
        super().__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0  # grid size

    def compute_grid_offsets(self, grid_size, cuda=True):
        """ Generate grid offsets and scale anchor widths and heights
            grid_x: x coordinates of each grid (1, 1, g, g)
            grid_y: y coordinates of each grid (1, 1, g, g)
            scaled_anchors: anchor widths and heights scaled, same shape as anchors
            anchor_w: scaled anchor widths (1, A, 1, 1)
            anchor_h: scaled anchor heights (1, A, 1, 1)
        """
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        self.grid_size = grid_size
        g = self.grid_size
        self.stride = self.img_dim / self.grid_size  # Stride is basically grid length
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = (
            torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        )
        self.scaled_anchors = FloatTensor(
            [(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors]
        )
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None, img_dim=None):
        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor
        # Get feature map grid info.
        self.img_dim = img_dim  # (N,C,H,W)
        num_samples = x.size(0)
        grid_size = x.size(2)
        # Get predictions and reshape into (N,A,H,W,Cls+5)
        # Which means for each of the N images, at every (hi, wj) in (H, W) there is A achors
        # For each anchor there is C class probs, bbox in (x,y,w,h) and a confidence score
        prediction = (
            x.view(
                num_samples,
                self.num_anchors,
                self.num_classes + 5,
                grid_size,
                grid_size,
            )
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )
        # Direct bounding box prediction
        x = torch.sigmoid(prediction[..., 0])  # Center x (N,A,H,W)
        y = torch.sigmoid(prediction[..., 1])  # Center y (N,A,H,W)
        w = prediction[..., 2]  # Width (N,A,H,W)
        h = prediction[..., 3]  # Height (N,A,H,W)
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.
        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)
        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)  # placeholder (N,A,H,W,4)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h
        # Concat bbox locations, confidence scores and class probs together (N,AxHxW,Cls+5)
        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )
        # No targets means it is in detection mode
        if targets is None:
            return output, 0
        # Get targets and calculate loss if in training mode
        else:
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
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
            )
            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            loss_conf = (
                self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            )
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
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
            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                "x": to_cpu(loss_x).item(),
                "y": to_cpu(loss_y).item(),
                "w": to_cpu(loss_w).item(),
                "h": to_cpu(loss_h).item(),
                "conf": to_cpu(loss_conf).item(),
                "cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),
                "grid_size": grid_size,
            }
            return output, total_loss
