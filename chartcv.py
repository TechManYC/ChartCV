import os
import json
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F
import random
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import time
import torch.optim.lr_scheduler as lr_scheduler
import utils

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches

# Data Directories
train_images_dir = ""
coco_annotations_path = ""

# Load COCO Annotations
with open(coco_annotations_path, 'r') as f:
    coco_data = json.load(f)
images_data = coco_data['images']
annotations_data = coco_data['annotations']
categories_data = coco_data['categories']

# Create a mapping of image IDs to filenames
image_id_to_filename = {img['id']: img['file_name'] for img in images_data}

# Create a mapping of category IDs to names
category_id_to_name = {cat['id']: cat['name'] for cat in categories_data}

# Custom Dataset
class GraphDataset(Dataset):
    def __init__(self, images_dir, annotations, image_id_to_filename, transform=None):
        self.images_dir = images_dir
        self.annotations = annotations
        self.image_id_to_filename = image_id_to_filename
        self.transform = transform
        self.image_ids = list(set([ann['image_id'] for ann in annotations]))
        self.resize = T.Resize((224, 224)) 

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_filename = self.image_id_to_filename[img_id]
        img_path = os.path.join(self.images_dir, img_filename)
        img = read_image(img_path) 

        boxes = []
        labels = []
        for ann in self.annotations:
            if ann['image_id'] == img_id:
                boxes.append(ann['bbox'])
                labels.append(ann['category_id'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([img_id])

        original_width, original_height = img.shape[2], img.shape[1]
        img = self.resize(img)  
        new_width, new_height = 224, 224
        width_scale = new_width / original_width
        height_scale = new_height / original_height

        boxes = target["boxes"]
        # Convert to [x_min, y_min, x_max, y_max]
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]  # x_max = x_min + width
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]  # y_max = y_min + height
        boxes[:, 2] = torch.max(boxes[:, 0] + 1, boxes[:, 2]) #Ensure x_max > x_min
        boxes[:, 3] = torch.max(boxes[:, 1] + 1, boxes[:, 3]) #Ensure y_max > y_min

        # Scale the bounding boxes to [x_min, y_min, x_max, y_max]
        boxes[:, 0] *= width_scale  # Scale x_min
        boxes[:, 1] *= height_scale # Scale y_min
        boxes[:, 2] *= width_scale  # Scale x_max
        boxes[:, 3] *= height_scale # Scale y_max
        target["boxes"] = boxes

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

# Create Dataset and DataLoader
train_dataset = GraphDataset(train_images_dir, annotations_data, image_id_to_filename)
val_dataset = GraphDataset(train_images_dir, annotations_data, image_id_to_filename)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, _ = random_split(train_dataset, [train_size, val_size])
_, val_dataset = random_split(val_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=utils.collate_fn)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=utils.collate_fn)

# Load Pre-trained Model
num_classes = len(category_id_to_name) + 1 
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Move model to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Learning Rate Scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)  

# Training Loop
num_epochs = 20
start_epoch = 0 # to be adjusted if there are saved checkpoint

for epoch in range(start_epoch, num_epochs): 
    train_counter=0
    val_counter=0
    model.train()
    epoch_loss = 0
     
    for images, targets in train_loader:        
        images = [image.to(device) for image in images]   
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets] #moves any PyTorch tensor values to a specified device, leaving other non-tensor values (int, float) unchanged

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()

        train_counter += 1 
        if train_counter % 1000 == 0: 
            print(f"Epoch: {epoch+1}, Loss: {losses.item()}")
                
        # Step the schedule
        scheduler.step()
        print(f"Epoch: {epoch+1}, Learning Rate: {optimizer.param_groups[0]['lr']}")


    # Validation loop
    model.eval()
    val_loss = 0
    predictions = [] 
    with torch.no_grad():
        for images, targets in val_loader:
            images = [image.to(device) for image in images] 
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            outputs = model(images) 

            for i, output in enumerate(outputs): #Iterate through each image in the batch.
                image_id = targets[i]['image_id'].item()
                pred_boxes = output['boxes'].cpu().numpy() 
                labels = output['labels'].cpu().numpy() 
                scores = output['scores'].cpu().numpy() 
                for box, label, score in zip(pred_boxes, labels, scores): 
                    x_min, y_min, x_max, y_max = box.astype(float)
                    width_box = x_max - x_min
                    height_box = y_max - y_min
                    prediction = {
                        'image_id': image_id,
                        'category_id': int(label),
                        'bbox': [x_min, y_min, width_box, height_box],
                        'score': float(score)
                    }
                    predictions.append(prediction)
    
            print("predictions",predictions)

            val_counter += 1 
            if val_counter % 1000 == 0: 
                print(f"Epoch: {epoch+1}, {val_counter}") 
    
        # Calculate mAP
        val_coco_annotations_path = ""
        coco_gt = COCO(val_coco_annotations_path)
        coco_dt = coco_gt.loadRes(predictions)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
print("Training Complete!")