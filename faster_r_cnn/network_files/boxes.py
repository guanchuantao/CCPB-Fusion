
import torch
from typing import Tuple
from torch import Tensor
import torchvision


def nms(boxes, scores, iou_threshold):
    # type: (Tensor, Tensor, float) -> Tensor
    """
    按分数排序：将所有边界框按置信度分数从高到低排序
    选择最高分框：选择分数最高的边界框，将其加入保留列表
    计算IoU并抑制：计算该框与所有剩余框的IoU
    移除IoU超过阈值的框（即抑制与当前框高度重叠的框）
    :param boxes: 边界框坐标，Tensor[N, 4]，格式为 (x1, y1, x2, y2)
    :param scores: 每个框的置信度分数，Tensor[N]
    :param iou_threshold: IoU阈值（如0.5），float
    :return: 保留框的索引（按分数降序排列）
    """
    # 直接调用torchvision内置的NMS操作
    return torch.ops.torchvision.nms(boxes, scores, iou_threshold)


def batched_nms(boxes, scores, idxs, iou_threshold):
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    """
    通过给不同类别的boxes添加不同偏移量，使得它们位于不同的"虚拟空间"，从而避免不同类别的boxes相互抑制
    :param boxes: 形状为[N, 4]的张量，边界框坐标（6960，4）
    :param scores: 形状为[N]的张量，每个边界框的得分（6960，）
    :param idxs: 形状为[N]的张量，每个边界框的类别索引（6960，）
    :param iou_threshold: IoU阈值0.7
    :return:
    """
    # 首先检查boxes是否为空，PyTorch 中用来检查张量是否为空的一种常见方法
    if boxes.numel() == 0:
        # 当没有检测到任何目标时boxes为空，提前返回
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    # 获取所有boxes中最大的坐标值（xmin, ymin, xmax, ymax）
    max_coordinate = boxes.max()
    # 为每一个类别/每一层生成一个很大的偏移量
    # 这里的to只是让生成tensor的dytpe和device与boxes保持一致
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    # boxes加上对应层的偏移量后，保证不同类别/层之间boxes不会有重合的现象
    # offsets[:, None]将形状从[N]变为[N,1]，通过与boxes的广播机制实现每个框的4个坐标加相同偏移
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    # keep = torchvision.ops.nms(boxes=boxes_for_nms, scores=scores, iou_threshold=iou_threshold)
    return keep


def remove_small_boxes(boxes, min_size):
    # type: (Tensor, float) -> Tensor
    """
    移除宽或高小于指定阈值边界框的索引
    :param boxes: 形状为[N, 4]的张量，边界框坐标
    :param min_size: 最小尺寸阈值
    :return:
    """
    # 计算每个边界框的宽度和高度
    ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    # 找出宽度和高度都大于阈值的边界框
    keep = torch.logical_and(torch.ge(ws, min_size), torch.ge(hs, min_size))
    # 返回这些边界框的索引
    keep = torch.where(keep)[0]
    return keep


def clip_boxes_to_image(boxes, size):
    # type: (Tensor, Tuple[int, int]) -> Tensor
    """
    将边界框裁剪到图像范围内
    :param boxes: 形状为[N, 4]的张量，边界框坐标
    :param size: 元组(height, width)，图像尺寸
    :return:
    """
    # 获取输入张量的维度
    dim = boxes.dim()
    # 切片操作，tensor[start:stop:step]，... 通配符确保适用于任意维度的输入
    # 0::2 获取所有第0、2列，坐标x_min, x_max
    boxes_x = boxes[..., 0::2]  # x1, x2
    # 1::2 获取所有第1、3列，y坐标y_min, y_max
    boxes_y = boxes[..., 1::2]  # y1, y2
    # 获取图像尺寸
    height, width = size
    # torchvision._is_tracing() 检查是否在模型导出/跟踪模式下 TODO：什么是模型导出/跟踪模式
    if torchvision._is_tracing():
        # 坐标裁剪（跟踪模式）
        # 显式创建边界值张量，保持类型和设备一致
        boxes_x = torch.max(boxes_x, torch.tensor(0, dtype=boxes.dtype, device=boxes.device))
        boxes_x = torch.min(boxes_x, torch.tensor(width, dtype=boxes.dtype, device=boxes.device))
        boxes_y = torch.max(boxes_y, torch.tensor(0, dtype=boxes.dtype, device=boxes.device))
        boxes_y = torch.min(boxes_y, torch.tensor(height, dtype=boxes.dtype, device=boxes.device))
    else:
        # 坐标裁剪（正常模式）
        # 限制x坐标范围在[0,width]之间
        boxes_x = boxes_x.clamp(min=0, max=width)
        # 限制y坐标范围在[0,height]之间
        boxes_y = boxes_y.clamp(min=0, max=height)
    # 使用torch.stack将x/y坐标重新交错组合，dim=dim 确保在原始维度上拼接，clipped_boxes结果形状与输入相同（如[N,4]）
    clipped_boxes = torch.stack((boxes_x, boxes_y), dim=dim)
    return clipped_boxes.reshape(boxes.shape)


def box_area(boxes):
    """
    计算边界框的面积
    :param boxes:
    :return:
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1, boxes2):
    """
    计算两组边界框之间的IOU（利用广播）
    :param boxes1: 形状为[N, 4]的张量代表真实的目标torch.Size([1, 4])
    :param boxes2: 形状为[M, 4]的张量代表anchor尺寸torch.Size([65472, 4])
    :return:
    """
    # 计算每个边界框的面积
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    # 计算相交区域的左上角(lt)和右下角(rb)坐标
    # boxes1的形状为[N,4]的张量，（:选择所有边界框，None在维度1插入新维度（变成[N,1,4]），2: 选择每个框的前两个坐标(x1,y1)）
    # :选择所有边界框，:2选择每个框的前两个坐标(x1,y1)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    # 计算相交区域的宽度和高度，并确保非负
    wh = (rb - lt).clamp(min=0)
    # 计算相交区域面积
    inter = wh[:, :, 0] * wh[:, :, 1]
    # 计算IoU = 相交面积 / (面积1 + 面积2 - 相交面积)
    iou = inter / (area1[:, None] + area2 - inter)
    return iou

