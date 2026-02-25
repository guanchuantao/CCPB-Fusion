

import torch.nn as nn
import torch
from collections import OrderedDict
from torch.jit.annotations import Tuple, List, Dict
import torch.nn.functional as F
from torch import Tensor


class IntermediateLayerGetter(nn.ModuleDict):
    """
    中间层获取器，用于从模型中提取指定层的输出
    继承自 ModuleDict：以字典形式管理子模块，保持顺序和访问效率
    注：模块在模型中的注册顺序必须与其使用顺序一致，不能重复使用同一个nn.Module两次，只能查询直接分配给模型的子模块
    """
    __annotations__ = {
        "return_layers": Dict[str, str],    # 类型注解：返回层字典
    }

    def __init__(self, model, return_layers):
        """
        初始化函数
        :param model: 要提取特征的模型
        :param return_layers: 字典，指定要返回的模块名称及对应的输出名称
        """
        # 检查要返回的层是否都在模型中存在
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("返回层不在模型中")
        # 保存原始返回层字典
        orig_return_layers = return_layers
        # 标准化字典键值（确保为字符串类型）{'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        # 按顺序构建有序字典，仅保留目标层及其前置层
        layers = OrderedDict()
        # 遍历模型子模块按顺序存入有序字典
        # 只保存layer4及其之前的结构，舍去之后不用的结构
        # 即conv1 → bn1 → layer1 → layer2 → layer3 → layer4 → fc保留为conv1 → bn1 → layer1 → layer2 → layer3 → layer4
        for name, module in model.named_children():
            layers[name] = module           # 添加当前模块
            if name in return_layers:       # 若该层是目标层
                del return_layers[name]     # 从待处理列表中移除
            if not return_layers:           # 所有目标层已处理完毕
                break
        # 初始化ModuleDict
        super().__init__(layers)
        # 保存用户指定的原始输出名{'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
        self.return_layers = orig_return_layers

    def forward(self, x):
        """
        获取Backbone四层输出
        :param x: 输入的图像尺寸torch.Size([2, 3, 512, 512])
        :return: {OrderedDict:4} -> {'0':'torch.Size([2, 256, 128, 128])',
                                     '1':'torch.Size([2, 512, 64, 64])',
                                     '2':'torch.Size([2, 1024, 32, 32])',
                                     '3':'torch.Size([2, 2048, 16, 16])'}
        """
        # 存储输出的有序字典
        out = OrderedDict()
        # 依次遍历模型的所有子模块，并进行正向传播，
        # 收集layer1, layer2, layer3, layer4的输出
        # 按添加顺序遍历子模块'conv1'、'bn1'、'relu'、'maxpool'、'layer1'（总共3层）、'layer2'（总共4层）、'layer3'（总共6层）、'layer4'（总共3层）
        for name, module in self.items():
            # 执行当前模块的前向计算
            x = module(x)
            # 若该层是目标输出层'conv1'、'bn1'、'relu'、'maxpool'这四层都不在
            if name in self.return_layers:
                # 获取用户指定的输出名
                out_name = self.return_layers[name]
                # 保存输出到字典
                out[out_name] = x
        return out


class FeaturePyramidNetwork(nn.Module):
    """
    特征金字塔网络实现
    """

    def __init__(self, in_channels_list, out_channels, extra_blocks=None):
        """
        初始化函数
        :param in_channels_list: 各特征图的输入通道数列表（[256,512,1024,2048]）
        :param out_channels: FPN输出的统一通道数（通常256）
        :param extra_blocks: 额外的特征块处理（LastLevelMaxPool）
        """
        super().__init__()
        # 1x1卷积调整输入通道数（横向连接）
        self.inner_blocks = nn.ModuleList()
        # 3x3卷积生成输出特征图
        self.layer_blocks = nn.ModuleList()
        # 为每个输入特征图创建对应的卷积块
        for in_channels in in_channels_list:
            # 跳过无效输入
            if in_channels == 0:
                continue
            # 1x1卷积标准化通道数
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
            # 3x3卷积消除上采样混叠
            layer_block_module = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

        # 参数初始化（避免受后续操作影响）
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                # Kaiming初始化配合ReLU激活函数
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
        # 额外处理块（如P6生成）
        self.extra_blocks = extra_blocks

    def get_result_from_inner_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        安全获取inner_blocks 4层1x1卷积指定层的输出（兼容TorchScript）
        :param x: [2048,1024,512,256]通道数对应的特征
        :param idx: 对应编号
        :return: 通过1x1卷积的特征
        """
        # self.inner_blocks：4层1x1卷积调整输入通道数（横向连接）
        num_blocks = len(self.inner_blocks)
        # 处理负数索引（如 -1 表示最后一个）
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.inner_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def get_result_from_layer_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        依次取出对应卷积
        :param x: 通过1x1卷积的特征，或者完成上采样相加的特征
        :param idx: 对应编号
        :return: [P5,P4,P3,P2]四个中的按序返回
        """
        # self.layer_blocks：4层3x3卷积
        num_blocks = len(self.layer_blocks)
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.layer_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        前向传播
        :param x: {OrderedDict:4} -> {'0':'torch.Size([2, 256, 128, 128])',
                                     '1':'torch.Size([2, 512, 64, 64])',
                                     '2':'torch.Size([2, 1024, 32, 32])',
                                     '3':'torch.Size([2, 2048, 16, 16])'}
        :return: {OrderedDict:5} -> {'0':'torch.Size([2, 256, 128, 128])',
                                     '1':'torch.Size([2, 256, 64, 64])',
                                     '2':'torch.Size([2, 256, 32, 32])',
                                     '3':'torch.Size([2, 256, 16, 16])',
                                     'pool':'torch.Size([2, 256, 8, 8])'}
        """
        # 解包输入（转有序字典为列表）
        names = list(x.keys())
        x = list(x.values())

        # 将resnet layer4的channel调整到指定的out_channels
        # last_inner = self.inner_blocks[-1](x[-1])
        # Step 1: 处理最深层（最后一个特征图通道数为2048的特征，通过卷积变成256通道数）torch.Size([2, 256, 16, 16])
        last_inner = self.get_result_from_inner_blocks(x[-1], -1)
        # result中保存着每个预测特征层
        results = []
        # 将layer4调整channel后的特征矩阵，通过3x3卷积后得到对应的预测特征矩阵
        # results.append(self.layer_blocks[-1](last_inner))
        # 对最深层的特征进行3x3卷积得到第一个输出 P5
        results.append(self.get_result_from_layer_blocks(last_inner, -1))
        # Step 2: 自顶向下融合特征，逆序处理（从深层到浅层）idx:(2, 1, 0)
        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)
            # 获取当前层空间尺寸（H, W）
            feat_shape = inner_lateral.shape[-2:]
            # 上采样深层特征，把深层上采样到浅层尺寸，最近邻插值
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            # 特征融合（逐元素相加）
            last_inner = inner_lateral + inner_top_down
            # 生成当前层输出P4，P3，P2，但是insert在0位置插入([2, 256, 32, 32])([2, 256, 64, 64])([2, 256, 128, 128])
            # 所以results：[P2,P3,P4,P5]
            results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))

        # Step 3: 处理额外特征块（如P6）
        if self.extra_blocks is not None:
            # 加上最大池化层，并且获得输出特征results：[P2, P3, P4, P5, P6]，names：['0', '1', '2', '3', 'pool']
            results, names = self.extra_blocks(results, x, names)

        # 重新打包为有序字典
        out = OrderedDict([(k, v) for k, v in zip(names, results)])

        return out


class LastLevelMaxPool(torch.nn.Module):
    """
    在最后一个特征图上应用最大池化，用于生成FPN的P6特征层（通常用于大目标检测）
    输入: [P2, P3, P4, P5]，输出: [P2, P3, P4, P5, P6]，其中 P6 = MaxPool2d(P5)
    """
    def forward(self, x: List[Tensor], y: List[Tensor], names: List[str]) -> Tuple[List[Tensor], List[str]]:
        """
        前向传播
        :param x: FPN处理后的特征图列表[P2,P3,P4,P5]
        :param y: 原始特征图列表（未使用）（256通道数，512通道数，1024通道数，2048通道数）的特征图
        :param names: 特征图名称列表y对应的键
        :return: 返回扩展后的特征图和名称
        """
        # 在名称列表中添加"pool"标识
        names.append("pool")
        # 对最后一个特征图进行最大池化
        x.append(F.max_pool2d(
            x[-1],      # 取最后一个特征图(P5)
            1,      # 1x1的池化窗口
            2,      # 步长为2（实现下采样）
            0))     # 无填充
        # 返回扩展后的特征图和名称列表x:[P2, P3, P4, P5, P6]
        return x, names


class BackboneWithFPN(nn.Module):
    """
    带FPN的骨干网络,内部使用IntermediateLayerGetter提取指定层的特征图
    """

    def __init__(self,
                 backbone: nn.Module,
                 return_layers=None,
                 in_channels_list=None,
                 out_channels=256,
                 extra_blocks=None,
                 re_getter=True):
        """
        初始化参数
        :param backbone: 骨干网络
        :param return_layers: 要返回的层及其名称映射
        :param in_channels_list: 各层输入通道数列表（[256,512,1024,2048]）
        :param out_channels: FPN输出通道数（通常256）
        :param extra_blocks: 额外的特征块
        :param re_getter: 是否重新获取中间层
        """
        super().__init__()
        # 如果没有提供额外的块，使用默认的最大池化层
        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()
        # 如果需要重新获取中间层(默认True)
        if re_getter is True:
            assert return_layers is not None
            # 使用IntermediateLayerGetter提取指定层的输出
            self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        else:
            # 直接使用提供的backbone
            self.body = backbone
        # 初始化FPN
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
        )

        self.out_channels = out_channels

    def forward(self, x):
        """
        前向传播
        :param x: torch.Size([2, 3, 512, 512])
        :return: 先经过Backbone，再经过FPN结构的特征
        """
        """
        x: torch.Size([2, 3, 512, 512])
        """
        x = self.body(x)
        """
        self.body(x): {OrderedDict:4} -> {'0':'torch.Size([2, 256, 128, 128])',
                                          '1':'torch.Size([2, 512, 64, 64])',
                                          '2':'torch.Size([2, 1024, 32, 32])',
                                          '3':'torch.Size([2, 2048, 16, 16])'}
        """
        x = self.fpn(x)
        """
        self.fpn(x): {OrderedDict:5} -> {'0':'torch.Size([2, 256, 128, 128])',
                                         '1':'torch.Size([2, 256, 64, 64])',
                                         '2':'torch.Size([2, 256, 32, 32])',
                                         '3':'torch.Size([2, 256, 16, 16])',
                                         'pool':'torch.Size([2, 256, 8, 8])'}
        """
        return x
