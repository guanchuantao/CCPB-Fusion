


import torch
import torchvision
import torch.utils.data
from pycocotools.coco import COCO


def convert_to_coco_api(ds):
    """
    将自定义数据集转换为COCO API格式
    :param ds: 自定义数据集对象
    :return: COCO格式的数据集对象
    """
    # 创建一个空的COCO对象
    coco_ds = COCO()
    # 标注ID从1开始，而不是0
    ann_id = 1
    # 初始化COCO格式的数据集字典
    dataset = {
        'images': [],       # 存储图像信息
        'categories': [],   # 存储类别信息
        'annotations': []   # 存储标注信息
    }
    # 使用集合存储所有类别ID
    categories = set()
    # 遍历数据集中的每一张图像
    for img_idx in range(len(ds)):
        # 获取图像的高度、宽度和目标信息
        # 假设数据集实现了coco_index方法返回图像尺寸和目标
        hw, targets = ds.coco_index(img_idx)
        if len(targets) > 3:
            # 获取图像ID并转换为Python标量
            image_id = targets["image_id"].item()
            # 构建图像信息字典
            img_dict = {}
            # 图像ID
            img_dict['id'] = image_id
            # 图像高度
            img_dict['height'] = hw[0]
            # 图像宽度
            img_dict['width'] = hw[1]
            # 添加到图像列表
            dataset['images'].append(img_dict)
            # 处理边界框：将(x1,y1,x2,y2)格式转换为(x,y,width,height)格式
            bboxes = targets["boxes"]
            # 计算宽度和高度
            bboxes[:, 2:] -= bboxes[:, :2]
            # 转换为Python列表
            bboxes = bboxes.tolist()
            # 获取其他目标信息
            # 类别标签
            labels = targets['labels'].tolist()
            # 区域面积
            areas = targets['area'].tolist()
            # 是否拥挤标志
            iscrowd = targets['iscrowd'].tolist()
            # 获取当前图像中的目标数量
            num_objs = len(bboxes)
            # 为每个目标创建标注信息
            for i in range(num_objs):
                ann = {}
                ann['image_id'] = image_id          # 所属图像ID
                ann['bbox'] = bboxes[i]             # 边界框坐标
                ann['category_id'] = labels[i]      # 类别ID
                categories.add(labels[i])           # 添加到类别集合
                ann['area'] = areas[i]              # 区域面积
                ann['iscrowd'] = iscrowd[i]         # 是否拥挤
                ann['id'] = ann_id                  # 标注ID
                dataset['annotations'].append(ann)  # 添加到标注列表
                ann_id += 1                         # 标注ID递增
    # 构建类别信息列表
    dataset['categories'] = [{'id': i} for i in sorted(categories)]
    # 将数据集字典赋值给COCO对象
    coco_ds.dataset = dataset
    # 创建索引以便快速查找
    coco_ds.createIndex()
    return coco_ds


def get_coco_api_from_dataset(dataset):
    """
    从数据集对象获取COCO API对象
    :param dataset: 数据集对象
    :return: COCO API对象
    """
    # 最多尝试10次获取底层数据集
    for _ in range(10):
        # 如果是CocoDetection数据集直接跳出循环
        if isinstance(dataset, torchvision.datasets.CocoDetection):
            break
        # 如果是子集数据集，获取其底层数据集
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    # 如果是CocoDetection数据集，直接返回其coco属性
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
    # 否则将数据集转换为COCO API格式
    return convert_to_coco_api(dataset)
