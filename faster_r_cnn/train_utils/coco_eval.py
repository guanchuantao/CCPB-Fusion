


import json
from collections import defaultdict
import numpy as np
import copy
import torch
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import pycocotools.mask as mask_util
from train_utils.distributed_utils import all_gather


class CocoEvaluator(object):
    """
    COCO评估器，用于计算目标检测和分割的评估指标
    """
    def __init__(self, coco_gt, iou_types):
        """
        初始化参数
        :param coco_gt: COCO标注对象
        :param iou_types: 评估类型列表 (如['bbox', 'segm'])
        """
        assert isinstance(iou_types, (list, tuple))
        # 深拷贝避免修改原始数据
        coco_gt = copy.deepcopy(coco_gt)
        # 存储COCO标注对象
        self.coco_gt = coco_gt
        # 评估类型列表
        self.iou_types = iou_types
        # 存储每种评估类型的评估器
        self.coco_eval = {}
        for iou_type in iou_types:
            # 为每种评估类型创建COCO评估器
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)
        # 存储待评估图像ID
        self.img_ids = []
        # 存储每种评估类型的评估结果
        self.eval_imgs = {k: [] for k in iou_types}

    def update(self, predictions):
        """
        使用新的预测结果更新评估器
        :param predictions: 预测结果字典 {image_id: prediction_dict}
        :return:
        """
        # 获取当前批次的图像ID
        img_ids = list(np.unique(list(predictions.keys())))
        # 添加到总图像ID列表
        self.img_ids.extend(img_ids)
        # 对每种评估类型处理预测结果
        for iou_type in self.iou_types:
            # 准备COCO格式的预测结果
            results = self.prepare(predictions, iou_type)
            # 加载预测结果到COCO格式
            coco_dt = loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]

            # 设置评估参数
            # 预测结果
            coco_eval.cocoDt = coco_dt
            # 当前批次的图像ID
            coco_eval.params.imgIds = list(img_ids)
            # 执行评估
            img_ids, eval_imgs = evaluate(coco_eval)
            # 存储评估结果
            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self):
        """
        在分布式训练中同步所有进程的评估结果
        """
        for iou_type in self.iou_types:
            # 拼接所有进程的评估结果
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            # 创建统一的COCO评估对象
            create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])

    def accumulate(self):
        """
        累积评估结果
        """
        for coco_eval in self.coco_eval.values():
            # 调用COCOeval的累积方法
            coco_eval.accumulate()

    def summarize(self):
        """
        输出评估结果摘要
        """
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            # 调用COCOeval的摘要方法
            coco_eval.summarize()

    def prepare(self, predictions, iou_type):
        """
        根据评估类型准备预测结果
        :param predictions: 预测结果
        :param iou_type: 评估类型 ('bbox', 'segm', 'keypoints')
        :return: COCO格式的预测结果列表
        """
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        """
        为目标检测准备COCO格式的预测结果
        :param predictions:
        :return:
        """
        coco_results = []
        for original_id, prediction in predictions.items():
            # 跳过空预测
            if len(prediction) == 0:
                continue
            # 提取预测框、分数和标签
            boxes = prediction["boxes"]
            # 转换为[x,y,w,h]格式
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            # 构建COCO格式的检测结果
            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        """
        为实例分割准备COCO格式的预测结果
        :param predictions:
        :return:
        """
        coco_results = []
        for original_id, prediction in predictions.items():
            # 跳过空预测
            if len(prediction) == 0:
                continue
            # 提取预测分数、标签和掩码
            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            # 应用阈值二值化掩码
            masks = masks > 0.5
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            # 将掩码编码为RLE格式
            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            # 构建COCO格式的分割结果
            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        """
        为关键点检测准备COCO格式的预测结果
        :param predictions:
        :return:
        """
        coco_results = []
        for original_id, prediction in predictions.items():
            # 跳过空预测
            if len(prediction) == 0:
                continue
            # 提取预测框、分数、标签和关键点
            boxes = prediction["boxes"]
            # 转换为[x,y,w,h]格式
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            # 展平关键点
            keypoints = keypoints.flatten(start_dim=1).tolist()

            # 构建COCO格式的关键点结果
            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        'keypoints': keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    """
    将边界框从[xmin, ymin, xmax, ymax]格式转换为[x, y, width, height]格式
    :param boxes:
    :return:
    """
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def merge(img_ids, eval_imgs):
    """
    合并分布式训练中各进程的评估结果
    :param img_ids:
    :param eval_imgs:
    :return:
    """
    # 收集所有进程的图像ID和评估结果
    all_img_ids = all_gather(img_ids)
    all_eval_imgs = all_gather(eval_imgs)
    # 合并图像ID
    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)
    # 合并评估结果
    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)
    # 转换为NumPy数组
    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # 保留唯一图像ID（排序后）
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    """
    创建统一的COCO评估对象
    :param coco_eval:
    :param img_ids:
    :param eval_imgs:
    :return:
    """
    # 合并分布式结果
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())
    # 更新COCO评估对象
    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


#################################################################
# 以下是从pycocotools复制并修改的代码
# 主要修改：移除了print语句并修复了Python3的Unicode问题
#################################################################


def createIndex(self):
    """
    创建COCO数据集的索引结构（修改自pycocotools）
    :param self:
    :return:
    """
    # 初始化数据结构
    anns, cats, imgs = {}, {}, {}
    imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
    # 处理标注数据
    if 'annotations' in self.dataset:
        for ann in self.dataset['annotations']:
            # 图像到标注的映射
            imgToAnns[ann['image_id']].append(ann)
            # 标注ID到标注的映射
            anns[ann['id']] = ann
    # 处理图像数据
    if 'images' in self.dataset:
        for img in self.dataset['images']:
            # 图像ID到图像的映射
            imgs[img['id']] = img
    # 处理类别数据
    if 'categories' in self.dataset:
        for cat in self.dataset['categories']:
            # 类别ID到类别的映射
            cats[cat['id']] = cat
    # 创建类别到图像的映射
    if 'annotations' in self.dataset and 'categories' in self.dataset:
        for ann in self.dataset['annotations']:
            catToImgs[ann['category_id']].append(ann['image_id'])

    # 设置类成员
    self.anns = anns
    self.imgToAnns = imgToAnns
    self.catToImgs = catToImgs
    self.imgs = imgs
    self.cats = cats

# 别名
maskUtils = mask_util


def loadRes(self, resFile):
    """
    加载结果文件并返回结果API对象（修改自pycocotools）
    :param self:
    :param resFile: 结果文件或结果数据
    :return: COCO结果对象
    """
    res = COCO()
    # 复制图像数据
    res.dataset['images'] = [img for img in self.dataset['images']]

    # 加载并准备结果
    if isinstance(resFile, str):    # 从文件加载
        anns = json.load(open(resFile))
    elif type(resFile) == np.ndarray:   # 从NumPy数组加载
        anns = self.loadNumpyAnnotations(resFile)
    else:   # 直接使用结果数据
        anns = resFile

    # 验证结果格式
    assert type(anns) == list, 'results in not an array of objects'
    annsImgIds = [ann['image_id'] for ann in anns]
    assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
        'Results do not correspond to current coco set'

    # 根据结果类型处理数据
    if 'caption' in anns[0]:    # 图像描述任务
        imgIds = set([img['id'] for img in res.dataset['images']]) & set([ann['image_id'] for ann in anns])
        res.dataset['images'] = [img for img in res.dataset['images'] if img['id'] in imgIds]
        for id, ann in enumerate(anns):
            ann['id'] = id + 1

    elif 'bbox' in anns[0] and not anns[0]['bbox'] == []:   # 目标检测任务
        res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])       # 复制类别数据
        for id, ann in enumerate(anns):
            bb = ann['bbox']
            x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
            if 'segmentation' not in ann:       # 创建默认分割多边形
                ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
            ann['area'] = bb[2] * bb[3]         # 计算区域面积
            ann['id'] = id + 1      # 设置标注ID
            ann['iscrowd'] = 0      # 设置为单个对象

    elif 'segmentation' in anns[0]:     # 实例分割任务
        # 复制类别数据
        res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
        for id, ann in enumerate(anns):
            # 计算分割区域面积
            ann['area'] = maskUtils.area(ann['segmentation'])
            if 'bbox' not in ann:       # 从分割掩码计算边界框
                ann['bbox'] = maskUtils.toBbox(ann['segmentation'])
            ann['id'] = id + 1          # 设置标注ID
            ann['iscrowd'] = 0          # 设置为单个对象

    elif 'keypoints' in anns[0]:        # 关键点检测任务
        # 复制类别数据
        res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
        for id, ann in enumerate(anns):
            s = ann['keypoints']
            x = s[0::3]     # 提取x坐标
            y = s[1::3]     # 提取y坐标
            x1, x2, y1, y2 = np.min(x), np.max(x), np.min(y), np.max(y)
            ann['area'] = (x2 - x1) * (y2 - y1)     # 计算区域面积
            ann['id'] = id + 1      # 设置标注ID
            ann['bbox'] = [x1, y1, x2 - x1, y2 - y1]        # 计算边界框

    # 设置结果数据集
    res.dataset['annotations'] = anns
    # 创建索引
    createIndex(res)
    return res


def evaluate(self):
    """
    对给定图像运行评估并存储结果（修改自pycocotools）
    :param self:
    :return:
        imgIds: 图像ID列表
        evalImgs: 评估结果数组
    """
    p = self.params
    # 处理向后兼容性
    if p.useSegm is not None:
        p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
        print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
    # 准备评估参数
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    self.params = p
    # 准备评估
    self._prepare()
    # 确定评估类型
    catIds = p.catIds if p.useCats else [-1]

    if p.iouType == 'segm' or p.iouType == 'bbox':
        # 使用IoU计算
        computeIoU = self.computeIoU
    elif p.iouType == 'keypoints':
        # 使用OKS计算
        computeIoU = self.computeOks

    # 计算所有图像和类别的IoU/OKS
    self.ious = {
        (imgId, catId): computeIoU(imgId, catId)
        for imgId in p.imgIds
        for catId in catIds}

    # 评估每张图像
    evaluateImg = self.evaluateImg
    maxDet = p.maxDets[-1]
    evalImgs = [
        evaluateImg(imgId, catId, areaRng, maxDet)
        for catId in catIds
        for areaRng in p.areaRng
        for imgId in p.imgIds
    ]

    # 重塑评估结果为3D数组 [类别数 × 面积范围数 × 图像数]
    evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))
    self._paramsEval = copy.deepcopy(self.params)
    return p.imgIds, evalImgs

#################################################################
# 复制的pycocotools代码结束
#################################################################
