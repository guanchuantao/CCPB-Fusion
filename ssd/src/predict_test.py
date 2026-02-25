import os
import time
import json

import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt

from torchvision import transforms
from network_files import FasterRCNN, FastRCNNPredictor, AnchorsGenerator
from backbone import resnet50_fpn_backbone, MobileNetV2
from draw_box_utils import draw_objs

def create_model(num_classes):
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5)
    return model

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    model = create_model(num_classes=3)
    weights_path = "/Users/guanchuantao/Desktop/PythonProject/Faster R-CNN/all_project/faster_rcnn_fc/multi_ train/model_33.pth"
    assert os.path.exists(weights_path), "{} file does not exist.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)

    label_json_path = '/Users/guanchuantao/Desktop/PythonProject/Faster R-CNN/all_project/faster_rcnn_fc/pascal_voc_classes.json'
    assert os.path.exists(label_json_path), "json file {} does not exist.".format(label_json_path)
    with open(label_json_path, 'r') as f:
        class_dict = json.load(f)
    category_index = {str(v): str(k) for k, v in class_dict.items()}

    # 读取图片名称列表
    image_list_path = '/Users/guanchuantao/Desktop/PythonProject/Faster R-CNN/all_project/faster_rcnn_fc/VOCdevkit/VOC2012/ImageSets/Main/test_fc_well_two.txt'
    with open(image_list_path, 'r') as f:
        image_names = f.read().splitlines()

    # 图片文件夹路径
    image_folder = '/Users/guanchuantao/Desktop/计算机视觉图片/image_test/two_fc'
    # 输出文件夹路径
    output_folder = '/Users/guanchuantao/Desktop/计算机视觉图片/image_test/two_success_fc'
    os.makedirs(output_folder, exist_ok=True)

    data_transform = transforms.Compose([transforms.ToTensor()])
    # 初始化一个空字典来存储所有图片的预测结果
    results = {}
    model.eval()
    with torch.no_grad():
        for image_name in image_names:
            t_start = time_synchronized()
            image_path = os.path.join(image_folder, image_name)
            if not os.path.exists(image_path):
                print(f"Image {image_path} does not exist.")
                continue

            original_img = Image.open(image_path)
            img = data_transform(original_img)
            img = torch.unsqueeze(img, dim=0)

            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)
            predictions = model(img.to(device))[0]

            predict_boxes = predictions[0].to("cpu").numpy()
            predict_boxes[:, [0, 2]] = predict_boxes[:, [0, 2]] * original_img.size[0]
            predict_boxes[:, [1, 3]] = predict_boxes[:, [1, 3]] * original_img.size[1]
            predict_classes = predictions[1].to("cpu").numpy()
            predict_scores = predictions[2].to("cpu").numpy()

            # 将每张图片的预测结果存储到字典中
            results[image_name] = {
                "boxes": predict_boxes.tolist(),
                "classes": predict_classes.tolist(),
                "scores": predict_scores.tolist()
            }

            if len(predict_boxes) == 0:
                print("没有检测到任何目标!")

            plot_img = draw_objs(original_img,
                                 predict_boxes,
                                 predict_classes,
                                 predict_scores,
                                 category_index=category_index,
                                 box_thresh=0.5,
                                 line_thickness=3,
                                 font='/Users/guanchuantao/Desktop/PythonProject/Faster R-CNN/all_project/faster_rcnn_fc/simkai.ttf',
                                 font_size=20)
            output_path = os.path.join(output_folder, image_name)
            plot_img.save(output_path)
            t_end = time_synchronized()
            print(f"所用时间 for {image_name}: {t_end - t_start}")

        # 将结果字典保存为 JSON 文件
        with open('/Users/guanchuantao/Desktop/PythonProject/Faster R-CNN/all_project/faster_rcnn_fc/VOCdevkit/VOC2012/ImageSets/Main/two_fc_well.json',
                'w', encoding='utf-8') as json_file:
            json.dump(results, json_file, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    main()