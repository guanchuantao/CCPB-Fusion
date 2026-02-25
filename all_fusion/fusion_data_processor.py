import torch
import numpy as np
import os
from tqdm import tqdm
from torchvision import transforms as transforms_tv
import xml.etree.ElementTree as ET
from PIL import Image
from get_two_stage_clusterer import  TwoStageClusterer


def bbox_iou(box1, box2):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)

    inter_area = max(0, inter_rect_x2 - inter_rect_x1) * max(0, inter_rect_y2 - inter_rect_y1)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    return inter_area / (b1_area + b2_area - inter_area + 1e-6)

class FusionDataProcessor:
    def __init__(self, models, device, img_size=512):
        self.models = models
        self.device = device
        self.img_size = img_size
        self.transforms = transforms_tv.Compose([
            transforms_tv.ToTensor(),
        ])

    def infer_single_image(self, img_tensor):
        """
        List of dicts: [{'boxes': [[x1,y1,x2,y2]...], 'scores': [...], 'labels': [...]}, ...]
        """
        preds = {}
        with torch.no_grad():
            # Faster R-CNN
            if 'faster_rcnn' in self.models:
                out = self.models['faster_rcnn']([img_tensor.to(self.device)])[0]
                preds['faster_rcnn'] = out

            # Mask R-CNN
            if 'mask_rcnn' in self.models:
                out = self.models['mask_rcnn']([img_tensor.to(self.device)])[0]
                preds['mask_rcnn'] = out

            # SSD
            if 'ssd' in self.models:
                img_ssd = transforms_tv.Resize((300, 300))(img_tensor.unsqueeze(0).to(self.device))
                out = self.models['ssd'](img_ssd)[0]
                scale = torch.tensor([self.img_size, self.img_size, self.img_size, self.img_size], device=self.device)
                out['boxes'] = out['boxes'] * scale
                preds['ssd'] = out

            # YOLOv5
            if 'yolov5' in self.models:
                out = self.models['yolov5'](img_tensor.unsqueeze(0).to(self.device))[0]
                # [N, 6] (x1, y1, x2, y2, conf, cls)
                preds['yolov5'] = {
                    'boxes': out[:, :4],
                    'scores': out[:, 4],
                    'labels': out[:, 5].long()
                }
        return preds

    def generate_mlp_data(self, dataset_path, save_path='mlp_train_data.pth'):

        print(f"\n[Data Gen] Generating training data for MLP from {dataset_path}...")

        # 1.  (XML Parsing)
        img_dir = os.path.join(dataset_path, "JPEGImages")
        ann_dir = os.path.join(dataset_path, "Annotations")
        xml_files = [f for f in os.listdir(ann_dir) if f.endswith('.xml')]

        mlp_inputs = []
        mlp_targets = []
        
        model_order = ['yolov5', 'ssd', 'faster_rcnn', 'mask_rcnn']

        for xml_file in tqdm(xml_files):
            tree = ET.parse(os.path.join(ann_dir, xml_file))
            root = tree.getroot()

            img_name = root.find('filename').text
            img_path = os.path.join(img_dir, img_name)
            if not os.path.exists(img_path): continue

            img = Image.open(img_path).convert("RGB")
            orig_w, orig_h = img.size
            img = img.resize((self.img_size, self.img_size))
            img_tensor = self.transforms(img)

            gt_boxes = []
            gt_labels = []
            for obj in root.findall('object'):
                bndbox = obj.find('bndbox')

                xmin = float(bndbox.find('xmin').text) / orig_w
                ymin = float(bndbox.find('ymin').text) / orig_h
                xmax = float(bndbox.find('xmax').text) / orig_w
                ymax = float(bndbox.find('ymax').text) / orig_h
                label = 1

                gt_boxes.append([xmin, ymin, xmax, ymax])
                gt_labels.append(label)
                all_gt_wh.append([xmax - xmin, ymax - ymin])

            if len(gt_boxes) == 0: continue

            preds = self.infer_single_image(img_tensor)

            all_pred_boxes = []
            for m_name in model_order:
                if m_name in preds and len(preds[m_name]['boxes']) > 0:
                    p_boxes = preds[m_name]['boxes'] / self.img_size 
                    p_scores = preds[m_name]['scores']
                    p_labels = preds[m_name]['labels']
                    
                    for k in range(len(p_boxes)):
                        all_pred_boxes.append({
                            'box': p_boxes[k].tolist(),
                            'score': float(p_scores[k]),
                            'label': float(p_labels[k]),
                            'model': m_name
                        })
            
            all_pred_boxes.sort(key=lambda x: x['score'], reverse=True)
            grouped_candidates = []
            
            used_flags = [False] * len(all_pred_boxes)
            
            for i in range(len(all_pred_boxes)):
                if used_flags[i]: continue
                
                current_group = [all_pred_boxes[i]]
                used_flags[i] = True
                base_box = all_pred_boxes[i]['box']
                
                for j in range(i+1, len(all_pred_boxes)):
                    if used_flags[j]: continue
                    
                    iou = bbox_iou(base_box, all_pred_boxes[j]['box'])
                    if iou > 0.5: 
                        current_group.append(all_pred_boxes[j])
                        used_flags[j] = True
                
                grouped_candidates.append(current_group)
            
            
            for group in grouped_candidates:
                input_vec = []
                group_by_model = {m: [] for m in model_order}
                for item in group:
                    group_by_model[item['model']].append(item)
                
                for m_name in model_order:
                    items = group_by_model[m_name]
                    if len(items) > 0:
                        best_item = max(items, key=lambda x: x['score'])
                        b = best_item['box']
                        # [cx, cy, w, h, cls, score]
                        feat = [(b[0]+b[2])/2, (b[1]+b[3])/2, b[2]-b[0], b[3]-b[1], best_item['label'], best_item['score']]
                    else:
                        feat = [0.0] * 6
                    input_vec.extend(feat)
                
                rep_item = group[0]
                rep_box = rep_item['box']
                
                max_iou = 0.0
                best_gt_idx = -1
                
                if len(gt_boxes) > 0:
                    ious = [bbox_iou(rep_box, gtb) for gtb in gt_boxes]
                    max_iou = max(ious)
                    best_gt_idx = np.argmax(ious)
                
                if max_iou >= 0.5:
                    gt = gt_boxes[best_gt_idx]
                    gt_label = gt_labels[best_gt_idx]
                    target_vec = [(gt[0]+gt[2])/2, (gt[1]+gt[3])/2, gt[2]-gt[0], gt[3]-gt[1], float(gt_label), 1.0]
                else:
                    target_vec = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                
                mlp_inputs.append(input_vec)
                mlp_targets.append(target_vec)


            for i, gt_box in enumerate(gt_boxes):
                gt_x1, gt_y1, gt_x2, gt_y2 = gt_box
                gt_cx, gt_cy = (gt_x1 + gt_x2) / 2, (gt_y1 + gt_y2) / 2
                gt_w, gt_h = gt_x2 - gt_x1, gt_y2 - gt_y1

                #[tx, ty, tw, th, cls, score=1.0]
                target_vec = [gt_cx, gt_cy, gt_w, gt_h, gt_labels[i], 1.0]

                input_vec = []


                for m_name in model_order:
                    feature = [0.0] * 6

                    if m_name in preds and len(preds[m_name]['boxes']) > 0:
                        p_boxes = preds[m_name]['boxes'] / self.img_size
                        p_scores = preds[m_name]['scores']
                        p_labels = preds[m_name]['labels']


                        ious = [bbox_iou(gt_box, pb.tolist()) for pb in p_boxes]
                        best_idx = np.argmax(ious)
                        best_iou = ious[best_idx]

                        if best_iou > 0.5:
                            pb = p_boxes[best_idx].tolist()
                            px, py = (pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2
                            pw, ph = pb[2] - pb[0], pb[3] - pb[1]

                            # [x, y, w, h, cls, score]
                            feature = [px, py, pw, ph, float(p_labels[best_idx]), float(p_scores[best_idx])]

                    input_vec.extend(feature)

                # input_vec
                mlp_inputs.append(input_vec)
                mlp_targets.append(target_vec)


        mlp_inputs = torch.tensor(mlp_inputs, dtype=torch.float32)
        mlp_targets = torch.tensor(mlp_targets, dtype=torch.float32)

        clusterer = TwoStageClusterer()
        anchors = clusterer.fit(np.array(all_gt_wh))


        torch.save({'inputs': mlp_inputs, 'targets': mlp_targets, 'anchors': anchors}, save_path)
        print(f"[Data Gen] Saved {len(mlp_inputs)} samples to {save_path}")
        return mlp_inputs, mlp_targets, anchors
