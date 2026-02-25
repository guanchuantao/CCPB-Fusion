

from typing import List, Optional, Dict, Tuple
import torch
from torch import nn, Tensor
from torch.nn import functional as F
import torchvision
from . import det_utils
from . import boxes as box_ops
from .image_list import ImageList


@torch.jit.unused
def _onnx_get_num_anchors_and_pre_nms_top_n(ob, orig_pre_nms_top_n):
    # type: (Tensor, int) -> Tuple[int, int]
    from torch.onnx import operators
    num_anchors = operators.shape_as_tensor(ob)[1].unsqueeze(0)
    pre_nms_top_n = torch.min(torch.cat(
        (torch.tensor([orig_pre_nms_top_n], dtype=num_anchors.dtype),
         num_anchors), 0))

    return num_anchors, pre_nms_top_n


class AnchorsGenerator(nn.Module):
    __annotations__ = {
        "cell_anchors": Optional[List[torch.Tensor]],
        "_cache": Dict[str, List[torch.Tensor]]
    }

    """
    根据给定的anchor尺寸和比例，生成anchors模版，并且映射到原图上
    """

    def __init__(self, sizes=(128, 256, 512), aspect_ratios=(0.5, 1.0, 2.0)):
        """
        多尺度锚点生成器初始化参数
        :param sizes: 锚点的基本尺寸，对应特征图感受野大小
        :param aspect_ratios:锚点的宽高比
        """
        super(AnchorsGenerator, self).__init__()
        # 检查输入参数 sizes 的格式是否合法，如果 sizes 的第一个元素 不是 list 或 tuple
        if not isinstance(sizes[0], (list, tuple)):
            # 使用列表推导式将其转换为多层格式sizes=((128,), (256,), (512,))
            sizes = tuple((s,) for s in sizes)
        # 检查输入参数 aspect_ratios 的格式是否合法，如果 aspect_ratios 的第一个元素 不是 list 或 tuple
        if not isinstance(aspect_ratios[0], (list, tuple)):
            # 广播比例配置aspect_ratios = ((0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0))
            aspect_ratios = (aspect_ratios,) * len(sizes)
        # 尺寸与比例配置数量必须一致
        assert len(sizes) == len(aspect_ratios)
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        # 类型: List[Tensor], 每个特征层的锚点模板
        self.cell_anchors = None
        # 缓存生成的锚点坐标
        self._cache = {}

    def generate_anchors(self, scales, aspect_ratios, dtype=torch.float32, device=torch.device("cpu")):
        # type: (List[int], List[float], torch.dtype, torch.device) -> Tensor
        """
        按照比例和尺寸生成anchors
        :param scales: 锚点基础尺寸列表((32,), (64,), (128,), (256,), (512,)) 表示面积的平方根 scale = sqrt(area)
        :param aspect_ratios: (0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0)宽高比列表定义为 w/h
        :param dtype: 输出张量的数据类型
        :param device: 输出张量的计算设备（默认CPU）
        :return: Tensor [num_anchors, 4]，每个锚点的坐标 (xmin, ymin, xmax, ymax)，相对于中心点 (0, 0)
        """
        # 将Python列表转换为PyTorch张量，并指定设备和数据类型。
        scales = torch.as_tensor(scales, dtype=dtype, device=device)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
        # 给定宽高比 ar = w/h
        # 则：高度比例因子 h_ratio = sqrt(ar)（因为 h = sqrt(area / ar)），宽度比例因子 w_ratio = 1 / h_ratio = sqrt(1/ar)
        # 若 aspect_ratios=[0.5, 1.0, 2.0]：
        # h_ratios = [√0.5, √1.0, √2.0] ≈ [0.707, 1.0, 1.414]，w_ratios = [1/0.707, 1/1.0, 1/1.414] ≈ [1.414, 1.0, 0.707]
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1.0 / h_ratios
        # w_ratios[:, None]：将 w_ratios 从 [1.4142, 1.0000, 0.7071]变为[[1.4142],[1.0000],[0.7071]]尺寸从（3）变成（3，1）
        # scales[None, :]：将 scales 从 [128] 变为 [[128]]尺寸从（1）变成（1，1）
        # 通过广播机制计算外积，得到所有 (w_ratio, scale) 和 (h_ratio, scale) 的组合
        # w_ratios[:, None] * scales[None, :]得到[[[181.0193]], [[128.0000]], [[ 90.5097]]]尺寸变成（3，1，1）
        # 最后通过view(-1)展平变成[181.0193, 128.0000,  90.5097]尺寸为（3）
        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        # hs的值：[ 90.5097, 128.0000, 181.0193]
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)
        # 锚点中心为 (0, 0), xmin = -w/2, ymin = -h/2, xmax = w/2, ymax = h/2
        # torch.stack 将四个坐标分量合并为尺寸(3, 4) 的张量
        # [[-90.5097, -45.2548,  90.5097,  45.2548],
        #  [-64.0000, -64.0000,  64.0000,  64.0000],
        #  [-45.2548, -90.5097,  45.2548,  90.5097]]
        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        # round 四舍五入
        # [[-91., -45.,  91.,  45.],
        #  [-64., -64.,  64.,  64.],
        #  [-45., -91.,  45.,  91.]]   shape：（3，4）
        return base_anchors.round()

    def set_cell_anchors(self, dtype, device):
        # type: (torch.dtype, torch.device) -> None
        """
        根据初始化时配置的 sizes 和 aspect_ratios，生成所有特征层的锚点模板
        :param dtype: 输出张量的数据类型
        :param device: 输出张量的计算设备
        :return: 生成所有特征层的锚点模板cell_anchors
        """
        # 检查全局的cell_anchors是否存在，检查其设备是否与当前 device 一致
        if self.cell_anchors is not None:
            # 如果存在，就赋值
            cell_anchors = self.cell_anchors
            # 检查cell_anchors是否为空
            assert cell_anchors is not None
            # 如果设备一致，直接返回，避免重复生成，如果设备不一致，继续执行后续代码
            if cell_anchors[0].device == device:
                return
        # 根据提供的sizes和aspect_ratios生成anchors模板
        # anchors模板都是以(0, 0)为中心的anchor
        # sizes=((32,), (64,), (128,), (256,), (512,))
        # aspect_ratios = ((0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0))
        cell_anchors = [
            # 通过列表推导式生成不同尺寸和比例的anchor
            self.generate_anchors(sizes, aspect_ratios, dtype, device)
            for sizes, aspect_ratios in zip(self.sizes, self.aspect_ratios)
        ]
        # 将生成的锚点模板保存在 self.cell_anchors 中，后续直接使用，shape：（5，3，4）
        # 0 = {Tensor: (3, 4)} tensor([[-23., -11.,  23.,  11.],
        #                              [-16., -16.,  16.,  16.],
        #                              [-11., -23.,  11.,  23.]])
        # 1 = {Tensor: (3, 4)} tensor([[-45., -23.,  45.,  23.],
        #                              [-32., -32.,  32.,  32.],
        #                              [-23., -45.,  23.,  45.]])
        # 2 = {Tensor: (3, 4)} tensor([[-91., -45.,  91.,  45.],
        #                              [-64., -64.,  64.,  64.],
        #                              [-45., -91.,  45.,  91.]])
        # 3 = {Tensor: (3, 4)} tensor([[-181.,  -91.,  181.,   91.],
        #                              [-128., -128.,  128.,  128.],
        #                              [ -91., -181.,   91.,  181.]])
        # 4 = {Tensor: (3, 4)} tensor([[-362., -181.,  362.,  181.],
        #                              [-256., -256.,  256.,  256.],
        #                              [-181., -362.,  181.,  362.]])
        self.cell_anchors = cell_anchors

    def num_anchors_per_location(self):
        # 计算每个预测特征层上每个滑动窗口的预测目标数
        return [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]

    def grid_anchors(self, grid_sizes, strides):
        # type: (List[List[int]], List[List[Tensor]]) -> List[Tensor]
        """
        根据anchor模版和特征尺寸、下采样率，把anchor映射回原图上
        :param grid_sizes: 每个预测特征层的尺寸(height, width)正常值：[torch.Size([128, 128]), torch.Size([64, 64]), torch.Size([32, 32]), torch.Size([16, 16]), torch.Size([8, 8])]
        :param strides: 每个特征图在原图上的步长（即下采样率）[[tensor(4), tensor(4)], [tensor(8), tensor(8)], [tensor(16), tensor(16)], [tensor(32), tensor(32)], [tensor(64), tensor(64)]]
        :return: {list:5} -> 0:torch.Size([49152, 4]),1:torch.Size([12288, 4]),2:torch.Size([3072, 4]),3:torch.Size([768, 4]),4:torch.Size([192, 4])
        """
        # 初始化输出列表，存储各特征层的锚点坐标结果
        anchors = []
        cell_anchors = self.cell_anchors
        assert cell_anchors is not None

        # 遍历每个预测特征层的grid_size，strides和cell_anchors
        # size：torch.Size([128, 128]), stride：[tensor(4), tensor(4)],
        # base_anchors：tensor([[-23., -11.,  23.,  11.], [-16., -16.,  16.,  16.], [-11., -23.,  11.,  23.]])
        for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
            # 当前特征的高和宽
            grid_height, grid_width = size
            # 当前特征的高和宽映射回原图上的步长
            stride_height, stride_width = stride
            # 设备
            device = base_anchors.device
            # torch.arange(0, grid_width)：生成 [0, 1, ..., width-1] 的序列。将网格宽的索引映射到原图坐标
            shifts_x = torch.arange(0, grid_width, dtype=torch.float32, device=device) * stride_width
            # 将网格高的索引映射到原图坐标
            shifts_y = torch.arange(0, grid_height, dtype=torch.float32, device=device) * stride_height
            # 计算预测特征矩阵上每个点对应原图上的坐标(anchors模板的坐标偏移量)
            # torch.meshgrid函数分别传入行坐标和列坐标，生成网格行坐标矩阵和网格列坐标矩阵
            # shape: [grid_height, grid_width]
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            # shift_x = tensor([[  0.,   4.,   8.,  ..., 500., 504., 508.],
            #                   [  0.,   4.,   8.,  ..., 500., 504., 508.],
            #                   [  0.,   4.,   8.,  ..., 500., 504., 508.],
            #                   ...,
            #                   [  0.,   4.,   8.,  ..., 500., 504., 508.],
            #                   [  0.,   4.,   8.,  ..., 500., 504., 508.],
            #                   [  0.,   4.,   8.,  ..., 500., 504., 508.]])
            # shift_y = tensor([[  0.,   0.,   0.,  ...,   0.,   0.,   0.],
            #                   [  4.,   4.,   4.,  ...,   4.,   4.,   4.],
            #                   [  8.,   8.,   8.,  ...,   8.,   8.,   8.],
            #                   ...,
            #                   [500., 500., 500.,  ..., 500., 500., 500.],
            #                   [504., 504., 504.,  ..., 504., 504., 504.],
            #                   [508., 508., 508.,  ..., 508., 508., 508.]])
            # 将网格坐标展平为一维tensor([  0.,   4.,   8.,  ..., 500., 504., 508.])
            shift_x = shift_x.reshape(-1)
            # tensor([  0.,   0.,   0.,  ..., 508., 508., 508.])
            shift_y = shift_y.reshape(-1)
            # 重复 x, y 是为了与锚点模板 (xmin, ymin, xmax, ymax) 对齐
            # tensor([[  0.,   0.,   0.,   0.],
            #         [  4.,   0.,   4.,   0.],
            #         [  8.,   0.,   8.,   0.],
            #         ...,
            #         [500., 508., 500., 508.],
            #         [504., 508., 504., 508.],
            #         [508., 508., 508., 508.]])
            shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1)
            # 将anchors模板与原图上的坐标偏移量相加得到原图上所有anchors的坐标信息(shape不同时会使用广播机制)
            # shifts.view(-1, 1, 4) -> (4, 1, 4) tensor([[[0., 0., 0., 0.]],
            #                                            [[2., 0., 2., 0.]],
            #                                            [[0., 2., 0., 2.]],
            #                                            [[2., 2., 2., 2.]]])
            # base_anchors.view(1, -1, 4) -> (1, 3, 4) tensor([[[-90.5097, -45.2548,  90.5097,  45.2548],
            #                                                   [-64.0000, -64.0000,  64.0000,  64.0000],
            #                                                   [-45.2548, -90.5097,  45.2548,  90.5097]]])
            # shifts_anchor: tensor([[[-90.5097, -45.2548,  90.5097,  45.2548],
            #                         [-64.0000, -64.0000,  64.0000,  64.0000],
            #                         [-45.2548, -90.5097,  45.2548,  90.5097]],
            #                        [[-88.5097, -45.2548,  92.5097,  45.2548],
            #                         [-62.0000, -64.0000,  66.0000,  64.0000],
            #                         [-43.2548, -90.5097,  47.2548,  90.5097]],
            #                        [[-90.5097, -43.2548,  90.5097,  47.2548],
            #                         [-64.0000, -62.0000,  64.0000,  66.0000],
            #                         [-45.2548, -88.5097,  45.2548,  92.5097]],
            #                        [[-88.5097, -43.2548,  92.5097,  47.2548],
            #                         [-62.0000, -62.0000,  66.0000,  66.0000],
            #                         [-43.2548, -88.5097,  47.2548,  92.5097]]])
            shifts_anchor = shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)
            # tensor([[-90.5097, -45.2548,  90.5097,  45.2548],
            #         [-64.0000, -64.0000,  64.0000,  64.0000],
            #         [-45.2548, -90.5097,  45.2548,  90.5097],
            #         [-88.5097, -45.2548,  92.5097,  45.2548],
            #         [-62.0000, -64.0000,  66.0000,  64.0000],
            #         [-43.2548, -90.5097,  47.2548,  90.5097],
            #         [-90.5097, -43.2548,  90.5097,  47.2548],
            #         [-64.0000, -62.0000,  64.0000,  66.0000],
            #         [-45.2548, -88.5097,  45.2548,  92.5097],
            #         [-88.5097, -43.2548,  92.5097,  47.2548],
            #         [-62.0000, -62.0000,  66.0000,  66.0000],
            #         [-43.2548, -88.5097,  47.2548,  92.5097]])
            anchors.append(shifts_anchor.reshape(-1, 4))

        return anchors

    def cached_grid_anchors(self, grid_sizes, strides):
        # type: (List[List[int]], List[List[Tensor]]) -> List[Tensor]
        """

        :param grid_sizes: 每个预测特征层的尺寸(height, width)[torch.Size([128, 128]), torch.Size([64, 64]), torch.Size([32, 32]), torch.Size([16, 16]), torch.Size([8, 8])]
        :param strides: 每个特征图在原图上的步长（即下采样率）[[tensor(4), tensor(4)], [tensor(8), tensor(8)], [tensor(16), tensor(16)], [tensor(32), tensor(32)], [tensor(64), tensor(64)]]
        :return:
        """
        # 将 grid_sizes 和 strides 转换为字符串拼接作为唯一缓存键,str() 的稳定性强
        key = str(grid_sizes) + str(strides)
        # 如果缓存字典 self._cache 中存在当前 key，直接返回缓存的锚点坐标
        if key in self._cache:
            return self._cache[key]
        # 根据预测特征层尺寸、步长比生成每一个特征层映射到原图的anchor
        anchors = self.grid_anchors(grid_sizes, strides)
        # 把生成的边界框按照键值对存储在字典里面
        self._cache[key] = anchors
        return anchors

    def forward(self, image_list, feature_maps):
        # type: (ImageList, List[Tensor]) -> List[Tensor]
        """
        把每一个特征层的anchor都映射到原图上，并且都放在同一个列表里面
        :param image_list: 使用ImageList包装后的图像张量，里面含有两个元素image_sizes：[(512, 512), (512, 512)]，tensors：torch.Size([2, 3, 512, 512])
        :param feature_maps: 多层特征图FPN的P2-P6，每个特征图[N, C, Hi, Wi]
        :return:
        """
        # 获取每个预测特征层的尺寸(height, width)
        # 0 = {Size: 2} torch.Size([128, 128])
        # 1 = {Size: 2} torch.Size([64, 64])
        # 2 = {Size: 2} torch.Size([32, 32])
        # 3 = {Size: 2} torch.Size([16, 16])
        # 4 = {Size: 2} torch.Size([8, 8])
        grid_sizes = list([feature_map.shape[-2:] for feature_map in feature_maps])
        # 获取输入图像的height和width：torch.Size([512, 512])
        image_size = image_list.tensors.shape[-2:]
        # 获取变量类型和设备类型
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        # 计算每个特征图在原图上的步长（即下采样率）
        # 0 = {list: 2} [tensor(4), tensor(4)]
        # 1 = {list: 2} [tensor(8), tensor(8)]
        # 2 = {list: 2} [tensor(16), tensor(16)]
        # 3 = {list: 2} [tensor(32), tensor(32)]
        # 4 = {list: 2} [tensor(64), tensor(64)]
        strides = [[torch.tensor(image_size[0] // g[0], dtype=torch.int64, device=device),
                    torch.tensor(image_size[1] // g[1], dtype=torch.int64, device=device)] for g in grid_sizes]
        # 根据提供的sizes和aspect_ratios生成anchors模板，保存在 self.cell_anchors 中
        self.set_cell_anchors(dtype, device)
        # 计算/读取所有anchors的坐标信息（这里的anchors信息是映射到原图上的所有anchors信息，不是anchors模板）
        # 得到的是一个list列表，对应每张预测特征图映射回原图的anchors坐标信息
        anchors_over_all_feature_maps = self.cached_grid_anchors(grid_sizes, strides)
        # List[List[torch.Tensor]]对应着：图批次2、特征5、anchor坐标torch.Size([?, 4])
        anchors = torch.jit.annotate(List[List[torch.Tensor]], [])
        # 遍历一个batch中的每张图像，enumerate提供索引给i
        for i, (image_height, image_width) in enumerate(image_list.image_sizes):
            anchors_in_image = []
            # 遍历每张预测特征图映射回原图的anchors坐标信息
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                anchors_in_image.append(anchors_per_feature_map)
            anchors.append(anchors_in_image)
        # 将每一张图像的所有预测特征层的anchors坐标信息拼接在一起
        # anchors是个list，每个元素为一张图像的所有anchors信息
        anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
        # Clear the cache in case that memory leaks.
        self._cache.clear()
        return anchors


class RPNHead(nn.Module):
    """
    头部预测
    """

    def __init__(self, in_channels, num_anchors):
        """
        头部预测的参数初始化
        :param in_channels: 输入特征图的通道数（如FPN输出的256维）
        :param num_anchors: 每个特征层位置预测的锚点数量 3
        """
        super(RPNHead, self).__init__()
        # 3x3 滑动窗口
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # 计算预测的目标分数预测，每个锚点是前景（目标）或背景的概率
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        # 预测每个锚点的边界框偏移量（dx, dy, dw, dh）
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)
        # 遍历模型体
        for layer in self.children():
            # 对卷积权重进行初始化
            if isinstance(layer, nn.Conv2d):
                # 权重，从均值为0、标准差为0.01的正态分布初始化（小数值避免初始梯度爆炸）
                torch.nn.init.normal_(layer.weight, std=0.01)
                # 偏置，初始化为0（平衡正负样本影响）
                torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        # type: (List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        """
        根据FPN的P2-P6得到分类得分和回归参数
        :param x: 多层特征图，FPN的P2-P6
        {list:5} -> 0:torch.Size([2, 256, 128, 128]),
                    1:torch.Size([2, 256, 64, 64]),
                    2:torch.Size([2, 256, 32, 32]),
                    3:torch.Size([2, 256, 16, 16]),
                    4:torch.Size([2, 256, 8, 8])
        :return:
        logits:
        {list:5} -> 0:torch.Size([2, 3, 128, 128]),
                    1:torch.Size([2, 3, 64, 64]),
                    2:torch.Size([2, 3, 32, 32]),
                    3:torch.Size([2, 3, 16, 16]),
                    4:torch.Size([2, 3, 8, 8])
        bbox_reg:
        {list:5} -> 0:torch.Size([2, 12, 128, 128]),
                    1:torch.Size([2, 12, 64, 64]),
                    2:torch.Size([2, 12, 32, 32]),
                    3:torch.Size([2, 12, 16, 16]),
                    4:torch.Size([2, 12, 8, 8])
        """
        logits = []
        bbox_reg = []
        # 遍历特征图[2, 256, 128, 128]、[2, 256, 64, 64]、[2, 256, 32, 32]、[2, 256, 16, 16]、[2, 256, 8, 8]
        for i, feature in enumerate(x):
            # 3x3滑动卷积[2, 256, 128, 128]、[2, 256, 64, 64]、[2, 256, 32, 32]、[2, 256, 16, 16]、[2, 256, 8, 8]
            # 这里不改变通道数和尺寸
            t = F.relu(self.conv(feature))
            # 每层特征图的分类得分[2, 3, 128, 128]、[2, 3, 64, 64]、[2, 3, 32, 32]、[2, 3, 16, 16]、[2, 3, 8, 8]
            # 为每一个特征层（128x128、64x64）的每一个点对应的3个anchor预测一个属于目标的概率分数
            logits.append(self.cls_logits(t))
            # 每层特征图的回归参数[2, 12, 128, 128]、[2, 12, 64, 64]、[2, 12, 32, 32]、[2, 12, 16, 16]、[2, 12, 8, 8]
            # 为每一个特征层（128x128、64x64）的每一个点对应的3个anchor预测四个属于边界框偏移参数
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg


def permute_and_flatten(layer, N, A, C, H, W):
    # type: (Tensor, int, int, int, int, int) -> Tensor
    """
    调整tensor顺序，并进行reshape，把[2, 3, 128, 128]变成[2, 49152, 1]
    :param layer: 预测特征层上预测的目标概率或bbox回归参数 (shape: [N, A*C, H, W])
    :param N: batch_size
    :param A: 每个位置的anchor数量 (anchors_num_per_position)
    :param C: 类别数或4(bbox坐标)
    :param H: 特征图高度
    :param W: 特征图宽度
    :return:
    """
    # 将输入tensor从[N, A*C, H, W]变为[N, A, C, H, W]
    # 把[2, 3, 128, 128]变成[2, 3, 1, 128, 128]
    # 把[2, 12, 128, 128]变成[2, 3, 4, 128, 128]
    layer = layer.view(N, -1, C, H, W)
    # 调整维度顺序为[N, H, W, A, C]
    # 这样做的目的是将同一位置的不同anchor排在一起
    # 把[2, 3, 1, 128, 128]变成[2, 128, 128, 3, 1]
    # 把[2, 3, 4, 128, 128]变成[2, 128, 128, 3, 4]
    layer = layer.permute(0, 3, 4, 1, 2)
    # 合并前三个维度[N, H, W, A]为一个维度[N*H*W*A]
    # 最终shape为[N, -1, C], 其中-1 = H*W*A
    # 把[2, 128, 128, 3, 1]变成[2, 49152, 1]
    # 把[2, 128, 128, 3, 4]变成[2, 49152, 4]
    layer = layer.reshape(N, -1, C)
    return layer


def concat_box_prediction_layers(box_cls, box_regression):
    # type: (List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    合并所有预测特征层的box分类和回归结果分类变成[130944, 1]维度，回归变成[130944, 4]维度
    :param box_cls: 每个预测特征层上的目标概率列表[2, 3, 128, 128]、[2, 3, 64, 64]、[2, 3, 32, 32]、[2, 3, 16, 16]、[2, 3, 8, 8]
    :param box_regression: 每个预测特征层上的bbox回归参数列表[2, 12, 128, 128]、[2, 12, 64, 64]、[2, 12, 32, 32]、[2, 12, 16, 16]、[2, 12, 8, 8]
    :return:
    """
    # 存储各层分类结果
    box_cls_flattened = []
    # 存储各层回归结果
    box_regression_flattened = []

    # 遍历每个预测特征层
    for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
        # 注意，当计算RPN中的proposal时，classes_num=1,只区分目标和背景
        # 获取当前层的shape信息
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        # 计算每个位置的anchor数量，回归参数通道数是4*A
        A = Ax4 // 4
        # 计算类别数 (RPN中C=1，只有前景/背景两类)
        C = AxC // A

        # 调整分类结果的shape，box_cls_per_level为[2, 3, 128, 128]，通道数N：2，
        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C, H, W)
        box_cls_flattened.append(box_cls_per_level)

        # 调整回归结果的shape
        box_regression_per_level = permute_and_flatten(box_regression_per_level, N, A, 4, H, W)
        box_regression_flattened.append(box_regression_per_level)
    # 将所有层的分类结果沿第1维拼接 (合并不同特征层的预测结果)
    # box_cls_flattened为：[2, 49152, 1]、[2, 12288, 1]、[2, 3072, 1]、[2, 768, 1]、[2, 192, 1]变成[130944, 1]
    box_cls = torch.cat(box_cls_flattened, dim=1).flatten(0, -2)
    # 将所有层的回归结果沿第1维拼接并reshape为[-1,4]
    # box_regression_flattened为：[2, 49152, 4]、[2, 12288, 4]、[2, 3072, 4]、[2, 768, 4]、[2, 192, 4]变成[130944, 4]
    box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, 4)
    return box_cls, box_regression


class RegionProposalNetwork(torch.nn.Module):
    """
    RPN
    """
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
        'proposal_matcher': det_utils.Matcher,
        'fg_bg_sampler': det_utils.BalancedPositiveNegativeSampler,
        'pre_nms_top_n': Dict[str, int],
        'post_nms_top_n': Dict[str, int],
    }

    def __init__(self, anchor_generator, head,
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 pre_nms_top_n, post_nms_top_n, nms_thresh, score_thresh=0.0):
        """
        参数初始化
        :param anchor_generator: AnchorGenerator实例，生成多尺度和多宽高比的anchors
        :param head: RPNHead实例，计算anchors的前景概率和边界框回归参数的头部预测
        :param fg_iou_thresh: float，锚点与真实框的最小IOU，用于标记正样本
        :param bg_iou_thresh: float，锚点与真实框的最大IOU，用于标记负样本
        :param batch_size_per_image: int，每张图像用于计算损失的采样锚点数
        :param positive_fraction: float，正样本在采样中的比例
        :param pre_nms_top_n: Dict[str, int]，training: 训练时每张图像在NMS前保留的候选框数，testing: 测试时每张图像在NMS前保留的候选框数
        :param post_nms_top_n: Dict[str, int]，training: 训练时每张图像在NMS后保留的最终候选框数，testing: 测试时每张图像在NMS后保留的最终候选框数
        :param nms_thresh: float，非极大值抑制的IOU阈值
        :param score_thresh: float，候选框的置信度过滤阈值
        """
        super(RegionProposalNetwork, self).__init__()
        # 锚点生成器，用于生成不同尺度和比例的anchors
        self.anchor_generator = anchor_generator
        # RPN头部网络，用于预测目标概率和bbox回归参数
        self.head = head
        # 边界框编解码器，用于将预测的偏移量转换为实际坐标
        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        # 计算anchors与gt boxes的IOU
        self.box_similarity = box_ops.box_iou
        # 匹配器，用于将anchors与gt boxes匹配
        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,  # 前景IoU阈值(0.7)
            bg_iou_thresh,  # 背景IoU阈值(0.3)
            allow_low_quality_matches=True  # 允许低质量匹配
        )
        # 正负样本采样器，平衡训练样本
        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image,  # 每张图像的采样数(256)
            positive_fraction  # 正样本比例(0.5)
        )

        # NMS相关参数
        # NMS前保留的proposal数量{'testing': 1000, 'training': 2000}
        self._pre_nms_top_n = pre_nms_top_n
        # NMS后保留的proposal数量{'testing': 1000, 'training': 2000}
        self._post_nms_top_n = post_nms_top_n
        # NMS阈值(通常0.7)
        self.nms_thresh = nms_thresh
        # 分数阈值 0.0
        self.score_thresh = score_thresh
        # 最小proposal尺寸
        self.min_size = 1.

    def pre_nms_top_n(self):
        if self.training:
            return self._pre_nms_top_n['training']
        return self._pre_nms_top_n['testing']

    def post_nms_top_n(self):
        if self.training:
            return self._post_nms_top_n['training']
        return self._post_nms_top_n['testing']

    def assign_targets_to_anchors(self, anchors, targets):
        # type: (List[Tensor], List[Dict[str, Tensor]]) -> Tuple[List[Tensor], List[Tensor]]
        """

        :param anchors: 所有映射回原图的anchor {list:2}torch.Size([65472, 4])torch.Size([65472, 4])
        :param targets: 图像对应的目标，{list:2}:{dict:5}{dict:5}
        :return: 把anchor与真实目标进行匹配，正样本处标记为1，负样本处标记为0，丢弃样本处标记为-1
        """
        labels = []
        matched_gt_boxes = []
        # 遍历每张图像的anchors和targets
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            # 取出目标边界框
            gt_boxes = targets_per_image["boxes"]
            # 如果边界框没有，那么生成为0的空目标 TODO：这里可做无目标学习
            if gt_boxes.numel() == 0:
                device = anchors_per_image.device
                matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
                labels_per_image = torch.zeros((anchors_per_image.shape[0],), dtype=torch.float32, device=device)
            else:
                # 计算anchors与真实bbox的iou信息
                match_quality_matrix = box_ops.box_iou(gt_boxes, anchors_per_image)
                # 计算每个anchors与gt匹配iou最大的索引
                # 第一种情况：如果iou<0.3索引为-1，0.3<iou<0.7索引为-2，0.7<iou索引为0
                # 第二种情况：使用低质量匹配，即没有IOU大于0.7，那么就把与真实边界框匹配最大的anchor设置为0
                matched_idxs = self.proposal_matcher(match_quality_matrix)
                # 这里使用clamp设置下限0是为了方便取每个anchors对应的gt_boxes信息
                # 负样本和舍弃的样本都是负值，所以为了防止越界直接置为0
                # 因为后面是通过labels_per_image变量来记录正样本位置的，
                # 所以负样本和舍弃的样本对应的gt_boxes信息并没有什么意义，
                # 反正计算目标边界框回归损失时只会用到正样本。
                matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]

                # 正样本处标记为1，负样本处标记为0，丢弃样本处标记为-1
                # 初始化labels (正样本=1，其他=0)
                labels_per_image = matched_idxs >= 0
                labels_per_image = labels_per_image.to(dtype=torch.float32)

                # 背景，即负样本
                bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD  # -1
                labels_per_image[bg_indices] = 0.0

                # 标记忽略样本，介于正负样本之间
                inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS  # -2
                labels_per_image[inds_to_discard] = -1.0
            # 标签：正样本处标记为1，负样本处标记为0，丢弃样本处标记为-1
            labels.append(labels_per_image)
            # 匹配的gt boxes坐标
            matched_gt_boxes.append(matched_gt_boxes_per_image)
        return labels, matched_gt_boxes

    def _get_top_n_idx(self, objectness, num_anchors_per_level):
        # type: (Tensor, List[int]) -> Tensor
        """
        获取每张预测特征图上预测概率排前pre_nms_top_n的anchors索引值
        Args:
            objectness: Tensor(每张图像的预测目标概率信息 )torch.Size([2, 65472])
            num_anchors_per_level: List（每个预测特征层上的预测的anchors个数）[49152, 12288, 3072, 768, 192]
        Returns:

        """
        r = []  # 记录每个预测特征层上预测目标概率前pre_nms_top_n的索引信息
        offset = 0
        # 遍历每个预测特征层上的预测目标概率信息，把objectness按照num_anchors_per_level长度进行分割
        for ob in objectness.split(num_anchors_per_level, 1):
            if torchvision._is_tracing():
                num_anchors, pre_nms_top_n = _onnx_get_num_anchors_and_pre_nms_top_n(ob, self.pre_nms_top_n())
            else:
                num_anchors = ob.shape[1]  # 预测特征层上的预测的anchors个数
                pre_nms_top_n = min(self.pre_nms_top_n(), num_anchors)  # 获取最少anchor数2000

            # 取 ob 每行的前 pre_nms_top_n 个最大值的索引和值
            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
            # offset记录每一批次的anchors数量，这样把索引加上去就属于全局的索引
            r.append(top_n_idx + offset)
            offset += num_anchors
        return torch.cat(r, dim=1)

    def filter_proposals(self, proposals, objectness, image_shapes, num_anchors_per_level):
        # type: (Tensor, Tensor, List[Tuple[int, int]], List[int]) -> Tuple[List[Tensor], List[Tensor]]
        """
        筛选proposals: 移除小框、应用NMS、保留topN
        :param proposals: 预测的proposals坐标torch.Size([2, 65472, 4])
        :param objectness: 预测的目标概率torch.Size([130944, 1])
        :param image_shapes: 图像原始尺寸[(512, 512), (512, 512)]
        :param num_anchors_per_level: 每个特征层的anchor数量[49152, 12288, 3072, 768, 192]
        :return:
        """
        num_images = proposals.shape[0]
        device = proposals.device
        # 分离objectness以便不计算梯度，objectness 是中间结果，但后续计算不需要它的梯度，用 detach() 可减少计算量
        objectness = objectness.detach()
        # 把objectness维度[130944, 1]按照图片批次变成torch.Size([2, 65472])
        objectness = objectness.reshape(num_images, -1)
        # 记录每个proposal来自哪个特征层
        levels = [torch.full((n,), idx, dtype=torch.int64, device=device)
                  for idx, n in enumerate(num_anchors_per_level)]
        # 生成（65472，）维度向量，分别49152个0, 12288个1, 3072个2, 768个3, 192个4
        levels = torch.cat(levels, 0)
        # expand_as(objectness)将 levels 扩展（复制）到与 objectness 相同的形状
        levels = levels.reshape(1, -1).expand_as(objectness)
        # 获取每张图像的前pre_nms_top_n个proposals
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)
        # 根据索引获取对应的proposals、分数和层级
        image_range = torch.arange(num_images, device=device)
        batch_idx = image_range[:, None]  # [batch_size, 1]
        # 根据每个预测特征层预测概率排前pre_nms_top_n的anchors索引值获取相应概率信息
        objectness = objectness[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]
        # 预测概率排前pre_nms_top_n的anchors索引值获取相应bbox坐标信息
        proposals = proposals[batch_idx, top_n_idx]
        # 计算概率分数
        objectness_prob = torch.sigmoid(objectness)
        final_boxes = []
        final_scores = []
        # 遍历每张图像的相关预测信息
        for boxes, scores, lvl, img_shape in zip(proposals, objectness_prob, levels, image_shapes):
            # 调整预测的boxes信息，将越界的坐标调整到图片边界上(限制x坐标范围在[0,width]之间，限制y坐标范围在[0,height]之间)
            boxes = box_ops.clip_boxes_to_image(boxes, img_shape)
            # 移除太小的proposals
            keep = box_ops.remove_small_boxes(boxes, self.min_size)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]
            # 移除低分proposals
            keep = torch.where(torch.ge(scores, self.score_thresh))[0]
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]
            # 按层级分组进行NMS，然后获得索引
            keep = box_ops.batched_nms(boxes, scores, lvl, self.nms_thresh)
            # 保留post_nms_top_n个proposals TODO：这里应该保留分数大的前2000个
            keep = keep[: self.post_nms_top_n()]
            # 取出2000个box和score
            boxes, scores = boxes[keep], scores[keep]
            final_boxes.append(boxes)
            final_scores.append(scores)
        return final_boxes, final_scores

    def compute_loss(self, objectness, pred_bbox_deltas, labels, regression_targets):
        # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
        """
        计算RPN的损失函数
        :param objectness: 预测的前景概率[130944, 1]维度
        :param pred_bbox_deltas: 预测的bbox回归参数[130944, 4]维度
        :param labels: {list:2}:(65472,)(65472,)（anchor的标签）前景（正样本处标记为1），背景（负样本处标记为0），废弃的anchors（丢弃样本处标记为-1）
        :param regression_targets: {tuple:2}:(65472, 4)(65472, 4)真实的bbox回归目标，gt与anchor计算的中心点偏移量、高宽偏移量
        :return:
        """
        # 按照给定的batch_size_per_image, positive_fraction选择正负样本，
        # 这里sampled_pos_inds、sampled_neg_inds分别为尺寸和labels一样的列表，取值全部为0，
        # 选中正样本则在sampled_pos_inds对应位置设置为1，选中负样本则在sampled_neg_inds对应位置设置为1
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        # 合并所有正负样本索引
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        # 准备数据（展平）
        objectness = objectness.flatten()
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        # 计算回归损失(仅正样本)
        box_loss = det_utils.smooth_l1_loss(
            pred_bbox_deltas[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1 / 9,
            size_average=False,
        ) / (sampled_inds.numel())

        # 计算分类损失(二元交叉熵)
        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds], labels[sampled_inds]
        )

        return objectness_loss, box_loss

    def forward(self,
                images,  # type: ImageList
                features,  # type: Dict[str, Tensor]
                targets=None  # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[List[Tensor], Dict[str, Tensor]]
        """
        RPN前向传播
        :param images: ImageList对象，包含batch中的图像
        :param features: 来自backbone的多层特征图即FPN的输出
        :param targets: 真实标注目标，训练时使用
        :return:
        """
        # features是所有预测特征层组成的OrderedDict，这里获取所有特征层的值  0层、1层、2层、3层、pool层
        features = list(features.values())
        # 1、合并所有的预测分数、预测边界框回归参数，然后对映射回原图的anchor进行回归参数偏移，然后再按照图片数量还原批次
        # 通过RPN head计算每个预测特征层上的预测目标概率和bboxes regression参数
        objectness, pred_bbox_deltas = self.head(features)
        # 为batch中的每张图像生成anchors
        anchors = self.anchor_generator(images, features)
        # 获取batch大小
        num_images = len(anchors)

        # 计算每个特征层的anchor数量
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]

        # 合并所有特征层的预测结果objectness为[130944, 1]维度，pred_bbox_deltas为[130944, 4]维度
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness,
                                                                    pred_bbox_deltas)
        # 将预测的bbox regression参数应用到anchors上得到最终预测bbox坐标
        # proposals尺寸torch.Size([130944, 1, 4])
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        # 将proposals尺寸按照图片批次数量还原torch.Size([2, 65472, 4])
        proposals = proposals.view(num_images, -1, 4)

        # 筛除小boxes框，nms处理，根据预测概率获取前post_nms_top_n个目标
        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)
        # 计算损失(仅在训练时)
        losses = {}
        if self.training:
            assert targets is not None
            # 计算每个anchors最匹配的gt，并将anchors进行分类
            # labels：（标签）前景（正样本处标记为1），背景（负样本处标记为0），废弃的anchors（丢弃样本处标记为-1）
            # matched_gt_boxes：匹配的gt boxes坐标
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            # 结合anchors以及对应的gt，计算regression参数
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            # 计算分类和回归损失
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets
            )
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg
            }
        return boxes, losses
