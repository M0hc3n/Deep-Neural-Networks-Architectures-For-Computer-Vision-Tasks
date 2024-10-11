import torch

from collections import Counter

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
    
    x2 = torch.min(box_pred_x2, box_label_x2)
    y2 = torch.min(box_pred_y2, box_label_y2)
    
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box_pred_area = abs((box_pred_x2 - box_pred_x1) * (box_pred_y2 - box_pred_y1))
    box_label_area = abs((box_label_x2 - box_label_x1) * (box_label_y2 - box_label_y1))
    
    return intersection / (box_pred_area + box_label_area - intersection + 1e-6) # to avoid null division

def non_max_suppression(preds, iou_threshold, prob_threshold):
    bboxes = [box for box in preds if box[1] > prob_threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    result = []
    
    while bboxes:
        chosen_box = bboxes.pop(0)
        
        bboxes = [
            box 
            for box in bboxes
            if box[0] != chosen_box[0] # need to be of different class
            or intersection_over_union(
                boxes_pred=torch.tensor(chosen_box[2:]),
                boxes_label=torch.tensor(box[2:])
            ) < iou_threshold
        ]
        
        result.append(chosen_box)

    return result

def mean_average_precision(pred, target, iou_threshold=0.5, num_classes=20):
    avg_precisions = []
    
    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred:
            if detection[1] == c:
                detections.append(detection)

        for true_box in target:
            if true_box[1] == c:
                ground_truths.append(true_box)
                
        # keep dict of trainning set indexes as keys, with their occurences as values
        amount_bboxes = Counter([gt[0] for gt in ground_truths])
        
        # now the values are a tensor of zeros repeated value times
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)
            
        # sort by probabilities (at index 2)
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)
        
        if total_true_bboxes < 1:
            continue
        
        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]
            
            num_gts = len(ground_truth_img)
            best_iou = 0
            
            for idx, gt_img in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt_img[3:]),
                )
                
                if iou > best_iou:
                    best_iou = iou
                    bst_gt_idx = idx
            
            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][bst_gt_idx] == 0:
                    amount_bboxes[detection[0]][bst_gt_idx] = 1
                    TP[detection_idx] = 1
                else:
                    FP[detection_idx] = 1
                    
            else:
                FP[detection_idx] = 1
        
        TP_cumulative_sum = torch.cumsum(TP, dim=0)
        FP_cumulative_sum = torch.cumsum(FP, dim=0)
        
        recalls = TP_cumulative_sum / (total_true_bboxes + 1e-6)
        precisions = TP_cumulative_sum / (TP_cumulative_sum + FP_cumulative_sum + 1e-6)
        
        recalls = torch.cat((torch.tensor([0]), recalls))
        precisions = torch.cat((torch.tensor([1]), precisions))
        
        avg_precisions.append(torch.trapz(precisions, recalls)) 
    
    return sum(avg_precisions) / len(avg_precisions)
        