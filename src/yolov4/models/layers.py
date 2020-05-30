import torch
import torch.nn as nn
from yolov4.utils import build_targets, to_cpu
from yolov4.loss import YOLOLoss


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super().__init__()


class Mish(nn.Module):
    """ Mish activation"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x


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
        self.loss = YOLOLoss(num_classes, ignore_thres=0.5, obj_scale=1, noobj_scale=100)

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
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf (N,A,H,W)
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred (N,A,H,W,C)
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
            loss_loc, loss_conf, loss_cls = self.loss(output, targets, (x,y,w,h), self.scaled_anchors)
            total_loss = loss_loc + loss_conf + loss_cls
            self.metrics = self.loss.metrics
            self.metrics["grid_size"] = grid_size
            return output, total_loss
