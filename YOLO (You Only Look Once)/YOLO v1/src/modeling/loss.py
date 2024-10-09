from modeling.metrics import intersection_over_union


class Loss(torch.nn.Module):
    
    def __init__(self, S=7, B=2, C=20):
        super(Loss, self).__init__()
        
        self.mse = torch.nn.MSELoss(reduction="sum")
        
        self.S = S
        self.B = B
        self.C = C
        
        # params values from the paper
        self.lambda_cord = 0.5
        self.lambda_noobj = 5
        
    def forward(self, preds, target):
        
        # converts from (batch size, S*S*(C+B*5)) to (batch size, S, S,C+B*5)
        preds = preds.reshape(-1, self.S, self.S, self.C + self.B * 5) 
        
        iou_box_1 = intersection_over_union(preds[..., 21:25], target[..., 21:25])
        iou_box_2 = intersection_over_union(preds[..., 26:30], target[..., 21:25])
        
        ious = torch.concat([iou_box_1.unsequeeze(0), iou_box_2.unsequeeze(0)], dim=0)
        
        iou_maxes, bestbox = torch.max(ious, dim=0)
        exists_box = target[..., 20].unsequeeze(3)
        
        
        box_predictions = exists_box * ( bestbox * preds[..., 26:30] + (1 - bestbox) * preds[..., 21:25])
        box_targets = exists_box * target[..., 21:25]
        
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(torch.sign(box_predictions[..., 2:4]) + 1e-6)
        box_targets[..., 2:4] = torch.sign(box_targets[..., 2:4])
        
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim==-2),
            torch.flatten(box_targets, end_dim==-2)
        ) 
        
        # object loss
        pred_box_confidences = bestbox * preds[..., 20:21] + (1 - bestbox) * preds[..., 25:26]
        
        obj_loss = self.mse(
            torch.flatten(exists_box * pred_box_confidences), 
            torch.flatten(exists_box * target[..., 20:21])
        )
        
        # no object loss
        noobj_loss = self.mse(
            torch.flatten((1 - exists_box) * preds[..., 20:21], start_dim=1), 
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )
        
        noobj_loss += self.mse(
            torch.flatten((1 - exists_box) * preds[..., 25:26], start_dim=1), 
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        # class loss
        class_loss = self.mse(
            torch.flatten(exists_box * preds[..., :20], end_dim=-2,),
            torch.flatten(exists_box * target[..., :20], end_dim=-2,)
        )
        
        return (self.lambda_cord * box_loss) +  obj_loss + (self.lambda_noobj * noobj_loss) + class_loss