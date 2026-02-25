

import os
import torch
import torch.nn as nn
from torchvision.ops.misc import FrozenBatchNorm2d
from .feature_pyramid_network import BackboneWithFPN, LastLevelMaxPool


class Bottleneck(nn.Module):
    """
    这里是写Resnet50的每一个块结构
    """
    # 这里设置扩展系数为4，是因为Resnet每一个块结构，输出通道数是输入通道数的4倍
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, norm_layer=None):
        """
        :param in_channel: 输入特征图的通道数
        :param out_channel: 中间层的通道数（注意实际输出通道数是out_channel * expansion）
        :param stride: 卷积步长，默认为1。当不为1时实现下采样功能
        :param downsample: 下采样函数，用于调整shortcut连接的维度
        :param norm_layer: 归一化层类型，默认为BatchNorm2d
        """
        super().__init__()
        # 如果归一化层为空，则默认使用BN层
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # 使用1x1的卷积层
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)
        # 归一化
        self.bn1 = norm_layer(out_channel)
        # -----------------------------------------
        # 使用3x3的卷积层
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        # 归一化
        self.bn2 = norm_layer(out_channel)
        # -----------------------------------------
        # 使用1x1的卷积层，并且输出通道数为out_channel * expansion
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)
        # 归一化
        self.bn3 = norm_layer(out_channel * self.expansion)
        # 激活函数，inplace=True表示原地操作节省内存
        self.relu = nn.ReLU(inplace=True)
        # 下采样函数，当输入输出维度不匹配时使用
        self.downsample = downsample

    def forward(self, x):
        """
        前向传播torch.Size([？, ？, ？, ？])
        :param x: 输入的特征,'layer1'（总共3层）、'layer2'（总共4层）、'layer3'（总共6层）、'layer4'（总共3层）
        [2, 64, 128, 128],[2, 256, 128, 128],[2, 256, 128, 128]
        [2, 256, 128, 128],[2, 512, 64, 64],[2, 512, 64, 64],[2, 512, 64, 64]
        [2, 512, 64, 64],[2, 1024, 32, 32],[2, 1024, 32, 32],[2, 1024, 32, 32],[2, 1024, 32, 32],[2, 1024, 32, 32]
        [2, 1024, 32, 32],[2, 2048, 16, 16],[2, 2048, 16, 16]
        :return: 结构化的特征
        """
        # 保存原始输入作为shortcut连接（恒等映射）
        identity = x
        # 如果有下采样函数，应用于shortcut连接
        # 当stride!=1或输入输出通道数不匹配时需要下采样
        if self.downsample is not None:
            # 调整identity的维度和x匹配
            identity = self.downsample(x)

        # 第一层：1x1卷积 + BN + ReLU
        out = self.conv1(x)  # 降维
        out = self.bn1(out)  # 归一化
        out = self.relu(out)  # 激活

        # 第二层：3x3卷积 + BN + ReLU
        out = self.conv2(out)  # 主要特征提取
        out = self.bn2(out)  # 归一化
        out = self.relu(out)  # 激活

        # 第三层：1x1卷积 + BN（不接ReLU）
        out = self.conv3(out)  # 升维
        out = self.bn3(out)  # 归一化

        # 残差连接：将处理后的特征与原始输入相加
        out += identity
        # 最终激活
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """
    这里是写Resnet50网络结构的
    """
    def __init__(self, block, blocks_num, num_classes=1000, include_top=True, norm_layer=None):
        """
        ResNet 初始化函数
        :param block: 基础块类型 (BasicBlock 或 Bottleneck)
        :param blocks_num: 每个 stage 的 block 数量列表
        :param num_classes: 分类类别数
        :param include_top: 是否包含最后的全连接层
        :param norm_layer: 归一化层类型
        """
        # 调用父类 nn.Module 的初始化
        super().__init__()
        # 如果没有指定归一化，默认使用 BatchNorm2d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # 将 norm_layer 保存为类属性
        self._norm_layer = norm_layer

        self.include_top = include_top
        # 初始通道数
        self.in_channel = 64
        # -----------------------------------------
        # 第一层：7x7 卷积 + BN + ReLU + MaxPool
        # 7x7 卷积，步距为2，输入通道3，输出通道64，填充3
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        # BN
        self.bn1 = norm_layer(self.in_channel)
        # ReLU
        self.relu = nn.ReLU(inplace=True)
        # MaxPool，池化大小3x3，步距2，填充1
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # -----------------------------------------
        # 构建四个 stage (layer1-layer4)，blocks_num=[3, 4, 6, 3]
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        # -----------------------------------------
        # 分类头部
        if self.include_top:
            # 自适应平均池化，输出尺寸为 (1,1)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            # 全连接分类层
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        """
        构建一个包含多个 block 的 stage
        :param block: block 类型 (BasicBlock/Bottleneck)
        :param channel: 该 stage 的基础通道数
        :param block_num: 该 stage 包含的 block 数量
        :param stride: 第一个 block 的步长
        :return:
        """
        # 获取归一化层类型
        norm_layer = self._norm_layer
        # 下采样函数初始化
        downsample = None
        # 判断是否需要下采样：
        # 1. stride 不等于 1 (需要空间下采样)
        # 2. 输入输出通道数不匹配 (需要通道数调整)
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                # 1x1 卷积调整通道数和空间尺寸
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                # 批归一化
                norm_layer(channel * block.expansion))
        # 存储该 stage 的所有 block
        layers = []
        # 添加第一个 block (可能包含下采样)
        layers.append(block(self.in_channel, channel, downsample=downsample,
                            stride=stride, norm_layer=norm_layer))
        # 更新输入通道数为当前 stage 的输出通道数
        self.in_channel = channel * block.expansion
        # 添加剩余的 block (stride=1，无下采样)
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel, norm_layer=norm_layer))
        # 将列表转换为 Sequential 模块
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播
        :param x: 输入图片
        :return: 输出特征（1000）
        """
        # 第一层卷积
        x = self.conv1(x)       # [3,H,W] -> [64,H/2,W/2]
        x = self.bn1(x)         # 批归一化
        x = self.relu(x)        # ReLU激活
        x = self.maxpool(x)     # [B,64,H/2,W/2] -> [B,64,H/4,W/4]
        # 四个 stage 的前向传播
        x = self.layer1(x)      # [B,64,H/4,W/4] -> [B,256,H/4,W/4]
        x = self.layer2(x)      # [B,256,H/4,W/4] -> [B,512,H/8,W/8]
        x = self.layer3(x)      # [B,512,H/8,W/8] -> [B,1024,H/16,W/16]
        x = self.layer4(x)      # [B,1024,H/16,W/16] -> [B,2048,H/32,W/32]
        # 分类头部
        if self.include_top:
            x = self.avgpool(x)         # [B,2048,H/32,W/32] -> [B,2048,1,1]
            x = torch.flatten(x, 1)     # [B,2048,1,1] -> [B,2048]
            x = self.fc(x)              # [B,2048] -> [B,num_classes]
        return x


def overwrite_eps(model, eps):
    """
    覆盖模型中所有FrozenBatchNorm2d层的默认eps值
    在BatchNorm中，eps 是一个极小值(通常1e-5)，用于防止除以零，计算公式：x / sqrt(var + eps)，确保数值稳定性，避免方差为零时出现NaN
    PyTorch早期版本的 FrozenBatchNorm2d 默认 eps=1e-5，新版本改为 eps=1e-4 ，为了兼容旧版预训练权重，需要改回 eps=0.0
    :param model: 要修改的模型
    :param eps: 新的eps值
    :return:
    """
    # 遍历模型中的所有模块
    for module in model.modules():
        # 检查当前模块是否是FrozenBatchNorm2d类型
        if isinstance(module, FrozenBatchNorm2d):
            # 修改该模块的eps属性为指定值
            module.eps = eps


def resnet50_fpn_backbone(pretrain_path="",
                          norm_layer=FrozenBatchNorm2d,
                          trainable_layers=3,
                          returned_layers=None,
                          extra_blocks=None):
    """
    构建ResNet50-FPN骨干网络
    :param pretrain_path: 预训练权重路径，空字符串表示不使用预训练权重
    :param norm_layer: 归一化层类型，默认使用不更新参数的FrozenBatchNorm2d(FrozenBatchNorm2d的功能与BatchNorm2d类似，但参数无法更新)
    :param trainable_layers: 指定训练哪些层结构(0-5)
    :param returned_layers: 指定返回哪些层的输出
    :param extra_blocks: 在输出的特征层基础上额外添加的层结构
    :return: 创建带有FPN的骨干网络
    """
    # -----------------------------------------
    # 构建ResNet50骨干网络(不包含顶部分类层)
    resnet_backbone = ResNet(Bottleneck,                    # 使用Bottleneck块
                             [3, 4, 6, 3],       # 各stage的block数量 [3,4,6,3]对应ResNet50
                             include_top=False,             # 不包含顶部分类层
                             norm_layer=norm_layer)         # 使用指定的归一化层

    # -----------------------------------------
    # 如果使用FrozenBatchNorm2d，设置eps=0.0(保持与预训练权重兼容)
    if isinstance(norm_layer, FrozenBatchNorm2d):
        overwrite_eps(resnet_backbone, 0.0)
    # 加载预训练权重
    if pretrain_path != "":
        assert os.path.exists(pretrain_path), "{} Resnet预训练权重找不到".format(pretrain_path)
        # 加载权重(非严格模式,允许模型和权重文件中的键不完全匹配,只加载能匹配的权重，跳过不匹配的部分,不会因为缺少某些键而报错)
        # 预训练模型通常包含最后的分类层（如1000类的ImageNet分类头），目标检测模型不需要这些层，可以跳过
        # 可能修改了原始网络结构（如移除了某些层），需要兼容不同结构的预训练权重
        # 可能只加载部分层的权重，冻结其他层的参数
        print(resnet_backbone.load_state_dict(                                  # 将加载的权重字典应用到模型中
                                                torch.load(pretrain_path),      # 从文件加载预训练权重
                                                strict=False))                  # 设置为非严格模式
    # 设置可训练层，检查trainable_layers范围是否合法
    assert 0 <= trainable_layers <= 5
    # 可训练层列表(从深层到浅层)默认为：['layer4', 'layer3', 'layer2']
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]

    # 如果要训练所有层(包括conv1)，需要额外加入bn1
    if trainable_layers == 5:
        # 因为BN层包含可训练的weight和bias
        layers_to_train.append("bn1")
    # 冻结不需要训练的层
    # 返回生成器，产生(name, parameter)元组 name是参数的完整路径（如layer1.0.conv1.weight）parameter是torch.nn.Parameter对象
    for name, parameter in resnet_backbone.named_parameters():
        # 如果参数名不以layers_to_train中的任何字符串开头，则冻结该参数
        # startswith()确保层级匹配（如layer1匹配layer1.0.conv1不会误匹配layer11等更高层），all()要求所有条件为真（即该参数不属于任何可训练层）
        if all([not name.startswith(layer) for layer in layers_to_train]):
            # 设置requires_grad=False，设置参数在反向传播时不计算梯度，冻结后，该参数在反向传播时的梯度为0
            parameter.requires_grad_(False)
    # 网络层级conv1 → bn1 → layer1 → layer2 → layer3 → layer4 → fc
    # 这里如果设置训练层trainable_layers=0，那么layers_to_train=[]，即只有fc参与训练，适用于特征提取（作为固定特征提取器）
    # 这里如果设置训练层trainable_layers=3，那么layers_to_train=['layer4', 'layer3', 'layer2']，典型微调场景
    # 这里如果设置训练层trainable_layers=5，那么layers_to_train=['layer4', 'layer3', 'layer2', 'conv1', 'bn1']，大数据集或完全训练
    # -----------------------------------------
    # 如果没有提供extra_blocks，使用默认的LastLevelMaxPool，用于在FPN顶部生成额外的特征图(P6)
    if extra_blocks is None:
        # 添加最大池化层
        extra_blocks = LastLevelMaxPool()
    # 设置要从骨干网络中返回的特征层编号(默认返回所有4层)
    if returned_layers is None:
        # 对应layer1到layer4
        returned_layers = [1, 2, 3, 4]
    # 检查返回层索引是否合法(1-4)
    assert min(returned_layers) > 0 and max(returned_layers) < 5
    # 创建返回层字典，格式如{'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
    return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}
    # 计算各层输入通道数
    # ResNet50中，layer1的输出通道数是256(64*4)
    in_channels_stage2 = resnet_backbone.in_channel // 8  # 256
    # 各层输入通道数列表 [256, 512, 1024, 2048]对应layer1-4
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    # FPN输出通道数(统一为256)
    out_channels = 256
    return BackboneWithFPN(resnet_backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks)

