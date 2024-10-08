import torch

def intersection_over_union(boxes_pred, boxes_label):
    
    box_pred_x1 = boxes_pred[..., 0:1] - boxes_pred[..., 2:3] / 2
    box_pred_y1 = boxes_pred[..., 1:2] - boxes_pred[..., 3:4] / 2
    
    box_pred_x2 = boxes_pred[..., 0:1] + boxes_pred[..., 2:3] / 2
    box_pred_y2 = boxes_pred[..., 1:2] + boxes_pred[..., 3:4] / 2
    
    
    box_label_x1 = boxes_label[..., 0:1] - boxes_label[..., 2:3] / 2
    box_label_y1 = boxes_label[..., 1:2] - boxes_label[..., 3:4] / 2
    
    box_label_x2 = boxes_label[..., 0:1] + boxes_label[..., 2:3] / 2
    box_label_y2 = boxes_label[..., 1:2] + boxes_label[..., 3:4] / 2

    x1 = torch.max(box_pred_x1, box_label_x1)
    y1 = torch.max(box_pred_y1, box_label_y1)
    
    x2 = torch.max(box_pred_x2, box_label_x2)
    y2 = torch.max(box_pred_y2, box_label_y2)
    
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box_pred_area = abs((box_pred_x2 - box_pred_x1) * (box_pred_y2 - box_pred_y1))
    box_label_area = abs((box_label_x2 - box_label_x1) * (box_label_y2 - box_label_y1))
    
    return intersection / (box_pred_area + box_label_area - intersection + 1e-6) # to avoid null division

class Loss(torch.nn.Module):
    
    def __init__(self, S=7, B=2, C=20):
        super(Loss, self).__init__()
        
        self.S = S
        self.B = B
        self.C = C
        
        # params values from the paper
        self.lambda_cord = 0.5
        self.lambda_noobj = 5
        
    def forward(self, preds, target):
        
        # converts from (batch size, S*S*(C+B*5)) to (batch size, S, S,C+B*5)
        preds = preds.reshape(-1, self.S, self.S, self.C + self.B * 5) 
        
        