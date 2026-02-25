

from torch import Tensor
from typing import List, Tuple


class ImageList(object):
    """
    将图像列表（可能大小不同）作为单个张量保存的结构。
    这是通过将图像填充到相同的大小，并将每个图像的原始大小存储在一个字段中来实现的
    """

    def __init__(self, tensors, image_sizes):
        # type: (Tensor, List[Tuple[int, int]]) -> None
        """
        全局变量复制
        :param tensor: 填充后的图像数据
        :param image_sizes: 图像填充后的尺寸
        """
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device):
        # type: (Device) -> ImageList
        """
        将类中存储的图像张量数据移动到指定的设备
        :param device: 设备（CPU、GPU）
        :return:
        """
        # 把传入的tensor转换设备
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)
