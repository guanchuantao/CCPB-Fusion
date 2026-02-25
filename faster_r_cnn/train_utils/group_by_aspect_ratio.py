


import bisect
from collections import defaultdict
import copy
from itertools import repeat, chain
import math
import numpy as np
import torch
import torch.utils.data
from torch.utils.data.sampler import BatchSampler, Sampler
from torch.utils.model_zoo import tqdm
import torchvision
from PIL import Image


def _repeat_to_at_least(iterable, n):
    """
    重复迭代器中的元素直到至少包含n个元素
    :param iterable: 可迭代对象
    :param n: 目标最小长度
    :return:
    """
    # 计算需要重复的次数
    repeat_times = math.ceil(n / len(iterable))
    # 重复并连接
    repeated = chain.from_iterable(repeat(iterable, repeat_times))
    # 转换为列表
    return list(repeated)


class GroupedBatchSampler(BatchSampler):
    """
    分组批采样器，确保每个批次只包含来自同一组的元素
    """
    def __init__(self, sampler, group_ids, batch_size):
        """
        初始化参数
        :param sampler: 基础采样器
        :param group_ids: 每个样本的组ID列表
        :param batch_size: 批次大小
        """
        if not isinstance(sampler, Sampler):
            # 验证采样器类型
            raise ValueError(
                "sampler should be an instance of "
                "torch.utils.data.Sampler, but got sampler={}".format(sampler)
            )
        self.sampler = sampler      # 基础采样器
        self.group_ids = group_ids      # 组ID列表
        self.batch_size = batch_size       # 批次大小

    def __iter__(self):
        """迭代器方法，生成分组批次"""
        buffer_per_group = defaultdict(list)        # 每个组的缓冲区
        samples_per_group = defaultdict(list)       # 每个组的样本列表

        num_batches = 0         # 已生成批次计数

        # 遍历基础采样器的索引
        for idx in self.sampler:
            group_id = self.group_ids[idx]          # 获取样本的组ID

            # 将索引添加到组缓冲区
            buffer_per_group[group_id].append(idx)
            samples_per_group[group_id].append(idx)

            # 如果缓冲区达到批次大小，生成一个批次
            if len(buffer_per_group[group_id]) == self.batch_size:
                yield buffer_per_group[group_id]    # 生成批次
                num_batches += 1
                del buffer_per_group[group_id]      # 清空缓冲区

            # 确保缓冲区不会超过批次大小
            assert len(buffer_per_group[group_id]) < self.batch_size

        # 处理剩余样本
        expected_num_batches = len(self)    # 预期批次数量
        num_remaining = expected_num_batches - num_batches  # 剩余批次数量

        if num_remaining > 0:
            # 按缓冲区大小降序排序（优先处理样本多的组）
            for group_id, _ in sorted(buffer_per_group.items(),
                                      key=lambda x: len(x[1]), reverse=True):
                # 计算需要补充的样本数
                remaining = self.batch_size - len(buffer_per_group[group_id])

                # 从该组样本中重复采样以补充
                samples_from_group_id = _repeat_to_at_least(samples_per_group[group_id], remaining)

                # 添加补充样本到缓冲区
                buffer_per_group[group_id].extend(samples_from_group_id[:remaining])
                assert len(buffer_per_group[group_id]) == self.batch_size

                # 生成批次
                yield buffer_per_group[group_id]
                num_remaining -= 1
                if num_remaining == 0:
                    break   # 所有剩余批次已处理

        # 验证所有剩余批次已处理
        assert num_remaining == 0

    def __len__(self):
        """返回批次数量"""
        return len(self.sampler) // self.batch_size


def _compute_aspect_ratios_slow(dataset, indices=None):
    """
    慢速计算宽高比 - 适用于不支持快速计算的数据集
    :param dataset: 数据集对象
    :param indices: 索引列表（可选）
    :return: 宽高比列表
    """
    print("您的数据集不支持快速计算宽高比，将遍历整个数据集并加载每张图像。这可能需要一些时间...")

    # 设置索引（默认为整个数据集）
    if indices is None:
        indices = range(len(dataset))

    # 定义子集采样器
    class SubsetSampler(Sampler):
        def __init__(self, indices):
            self.indices = indices

        def __iter__(self):
            return iter(self.indices)

        def __len__(self):
            return len(self.indices)

    # 创建数据加载器
    sampler = SubsetSampler(indices)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, sampler=sampler,
        num_workers=14,     # 使用14个工作进程加速
        collate_fn=lambda x: x[0]       # 自定义批处理函数
    )

    aspect_ratios = []      # 存储宽高比
    with tqdm(total=len(dataset)) as pbar:   # 进度条
        for _i, (img, _) in enumerate(data_loader):
            pbar.update(1)          # 更新进度条
            height, width = img.shape[-2:]        # 获取图像的宽高
            aspect_ratio = float(width) / float(height)         # 计算宽高比
            aspect_ratios.append(aspect_ratio)

    return aspect_ratios


def _compute_aspect_ratios_custom_dataset(dataset, indices=None):
    """
    计算自定义数据集的宽高比
    :param dataset: 自定义数据集（需实现get_height_and_width方法）
    :param indices: 索引列表（可选）
    :return: 宽高比列表
    """
    if indices is None:
        indices = range(len(dataset))

    aspect_ratios = []
    for i in indices:
        height, width = dataset.get_height_and_width(i)     # 获取图像尺寸
        aspect_ratio = float(width) / float(height)         # 计算宽高比
        aspect_ratios.append(aspect_ratio)

    return aspect_ratios


def _compute_aspect_ratios_coco_dataset(dataset, indices=None):
    """
    计算COCO数据集的宽高比
    :param dataset: COCO数据集
    :param indices: 索引列表（可选）
    :return: 宽高比列表
    """
    if indices is None:
        indices = range(len(dataset))

    aspect_ratios = []
    for i in indices:
        img_info = dataset.coco.imgs[dataset.ids[i]]    # 获取图像信息
        aspect_ratio = float(img_info["width"]) / float(img_info["height"])     # 计算宽高比
        aspect_ratios.append(aspect_ratio)

    return aspect_ratios


def _compute_aspect_ratios_voc_dataset(dataset, indices=None):
    """
    计算VOC数据集的宽高比
    :param dataset: VOC数据集
    :param indices: 索引列表（可选）
    :return: 宽高比列表
    """
    if indices is None:
        indices = range(len(dataset))
    aspect_ratios = []
    for i in indices:
        # 使用PIL打开图像（不加载到内存）
        width, height = Image.open(dataset.images[i]).size      # 获取图像尺寸
        aspect_ratio = float(width) / float(height)         # 计算宽高比
        aspect_ratios.append(aspect_ratio)

    return aspect_ratios


def _compute_aspect_ratios_subset_dataset(dataset, indices=None):
    """
    计算子集数据集的宽高比
    :param dataset: 子集数据集
    :param indices: 索引列表（可选）
    :return: 宽高比列表
    """
    if indices is None:
        indices = range(len(dataset))

    # 获取原始数据集的索引
    ds_indices = [dataset.indices[i] for i in indices]
    # 递归计算原始数据集的宽高比
    return compute_aspect_ratios(dataset.dataset, ds_indices)


def compute_aspect_ratios(dataset, indices=None):
    """
    计算数据集的宽高比（自动选择最佳方法）
    :param dataset: 数据集对象
    :param indices: 索引列表（可选）
    :return: 宽高比列表
    """
    # 检查数据集是否支持快速方法
    if hasattr(dataset, "get_height_and_width"):
        return _compute_aspect_ratios_custom_dataset(dataset, indices)

    # 处理特定数据集类型
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return _compute_aspect_ratios_coco_dataset(dataset, indices)

    if isinstance(dataset, torchvision.datasets.VOCDetection):
        return _compute_aspect_ratios_voc_dataset(dataset, indices)

    if isinstance(dataset, torch.utils.data.Subset):
        return _compute_aspect_ratios_subset_dataset(dataset, indices)

    # 默认使用慢速方法
    return _compute_aspect_ratios_slow(dataset, indices)


def _quantize(x, bins):
    """
    将连续值量化到离散区间
    :param x: 连续值列表
    :param bins: 区间边界列表
    :return: 量化后的索引列表
    """
    # 深拷贝避免修改原列表
    bins = copy.deepcopy(bins)
    # 确保区间有序
    bins = sorted(bins)
    # 使用二分查找确定每个值所属的区间
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def create_aspect_ratio_groups(dataset, k=0):
    """
    创建宽高比分组
    :param dataset: 数据集对象
    :param k: 分组参数（0表示不分组）
    :return: 每个样本的组ID列表
    """
    # 计算所有图像的宽高比
    aspect_ratios = compute_aspect_ratios(dataset)
    # 将[0.5, 2]区间划分成2*k等份(2k+1个点，2k个区间)
    bins = (2 ** np.linspace(-1, 1, 2 * k + 1)).tolist() if k > 0 else [1.0]

    # 将宽高比量化到区间
    groups = _quantize(aspect_ratios, bins)
    # 统计每个分组的样本数
    counts = np.unique(groups, return_counts=True)[1]
    # 打印分组信息
    fbins = [0] + bins + [np.inf]       # 完整区间边界
    print("使用以下区间进行宽高比量化: {}".format(fbins))
    print("每个区间的样本数: {}".format(counts))

    return groups
