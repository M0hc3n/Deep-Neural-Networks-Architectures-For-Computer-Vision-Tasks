import torch
from tqdm import tqdm

def cellboxes_to_boxes(predictions, S=7, B=2, C=20):
    """
    Convert YOLO output to bounding boxes
    
    Args:
    predictions (tensor): Model predictions of shape (batch_size, S*S*(C+B*5))
    S (int): Grid size (default 7)
    B (int): Number of bounding boxes per cell (default 2)
    C (int): Number of classes (default 20)
    
    Returns:
    list: List of lists containing bounding boxes [class_pred, prob_score, x, y, w, h]
    """
    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, S, S, C + B*5)
    bboxes1 = predictions[..., C+1:C+5]
    bboxes2 = predictions[..., C+6:C+10]
    scores = torch.cat(
        (predictions[..., C].unsqueeze(0), predictions[..., C+5].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(S).repeat(batch_size, S, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_h = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_h), dim=-1)
    predicted_class = predictions[..., :C].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., C], predictions[..., C+5]).unsqueeze(-1)
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds.reshape(batch_size, S*S, 6).tolist()

def get_bboxes(loader, model, iou_threshold, threshold):
    looper = tqdm(loader, leave=True)
    
    all_pred_boxes = []
    all_target_boxes = []
    model.eval()

    for batch_idx, (x, y) in enumerate(looper):
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        
        pred_bboxes = cellboxes_to_boxes(predictions)
        
        target_bboxes = cellboxes_to_boxes(y)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                preds=pred_bboxes[idx],
                iou_threshold=iou_threshold,
                prob_threshold=threshold,
            )
            
            for nms_box in nms_boxes:
                all_pred_boxes.append([batch_idx] + nms_box)

            for box in target_bboxes[idx]:
                if box[1] > threshold:
                    all_target_boxes.append([batch_idx] + box)

    model.train()
    return all_pred_boxes, all_target_boxes

