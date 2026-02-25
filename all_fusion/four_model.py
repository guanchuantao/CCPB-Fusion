import torch
import torch.nn as nn
from faster_r_cnn.backbone import resnet50_fpn_backbone as resnet50_fpn_backbone_faster_rcnn
from faster_r_cnn.network_files import FasterRCNN, FastRCNNPredictor
from mask_rcnn.network_files import MaskRCNN
from mask_rcnn.backbone import resnet50_fpn_backbone as resnet50_fpn_backbone_mask_rcnn
from ssd.src import SSD300
from ssd.src import Backbone as Backbone_ssd
# from yolov5.model_one import model_return


def load_all_models(device, configs):
    models = {}
    print("\n[Models] Loading 4 base models...")

    # 1. Faster R-CNN
    if 'faster_rcnn' in configs:
        backbone = resnet50_fpn_backbone_faster_rcnn(norm_layer=nn.BatchNorm2d, trainable_layers=5)
        model = FasterRCNN(backbone=backbone, num_classes=91)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, configs['faster_rcnn']['num_classes'])
        state = torch.load(configs['faster_rcnn']['path'], map_location='cpu')
        model.load_state_dict(state['model'] if 'model' in state else state)
        model.to(device).eval()
        models['faster_rcnn'] = model

    # 2. Mask R-CNN
    if 'mask_rcnn' in configs:
        backbone = resnet50_fpn_backbone_mask_rcnn(pretrain_path=None, trainable_layers=3)
        model = MaskRCNN(backbone, num_classes=configs['mask_rcnn']['num_classes'])
        state = torch.load(configs['mask_rcnn']['path'], map_location='cpu')
        model.load_state_dict(state['model'] if 'model' in state else state)
        model.to(device).eval()
        models['mask_rcnn'] = model

    # 3. SSD
    if 'ssd' in configs:
        backbone = Backbone_ssd()
        model = SSD300(backbone=backbone, num_classes=configs['ssd']['num_classes'])
        state = torch.load(configs['ssd']['path'], map_location='cpu')
        model.load_state_dict(state['model'] if 'model' in state else state)
        model.to(device).eval()
        models['ssd'] = model

    # 4. YOLOv5 (使用 torch.hub 或你自己的接口)
    if 'yolov5' in configs:
        # 示例：直接加载
        try:
            # model = torch.hub.load('ultralytics/yolov5', 'custom', path=configs['yolov5']['path'])
            model = torch.load(configs['yolov5']['path'], map_location=device)['model'].float()
            model.to(device).eval()
            models['yolov5'] = model
        except:
            print("  Warning: YOLOv5 load failed, make sure yolov5 repo is available.")

    print(f"[Models] Successfully loaded {len(models)} models.")
    return models