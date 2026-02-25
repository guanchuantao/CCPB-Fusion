


import os
import torch
import torch.nn as nn
from torchvision.ops.misc import FrozenBatchNorm2d
from .feature_pyramid_network import BackboneWithFPN, LastLevelMaxPool


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

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


def resnet34_fpn_backbone(pretrain_path="",
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
    resnet34_backbone = ResNet(BasicBlock,  # 使用Bottleneck块
                               [3, 4, 6, 3],  # 各stage的block数量 [3,4,6,3]对应ResNet50
                               include_top=False  # 不包含顶部分类层
                               )  # 使用指定的归一化层

    # -----------------------------------------
    # 如果使用FrozenBatchNorm2d，设置eps=0.0(保持与预训练权重兼容)
    if isinstance(norm_layer, FrozenBatchNorm2d):
        overwrite_eps(resnet34_backbone, 0.0)
    # 加载预训练权重
    if pretrain_path != "":
        assert os.path.exists(pretrain_path), "{} Resnet预训练权重找不到".format(pretrain_path)
        # 加载权重(非严格模式,允许模型和权重文件中的键不完全匹配,只加载能匹配的权重，跳过不匹配的部分,不会因为缺少某些键而报错)
        # 预训练模型通常包含最后的分类层（如1000类的ImageNet分类头），目标检测模型不需要这些层，可以跳过
        # 可能修改了原始网络结构（如移除了某些层），需要兼容不同结构的预训练权重
        # 可能只加载部分层的权重，冻结其他层的参数
        print(resnet34_backbone.load_state_dict(  # 将加载的权重字典应用到模型中
            torch.load(pretrain_path),  # 从文件加载预训练权重
            strict=False))  # 设置为非严格模式
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
    for name, parameter in resnet34_backbone.named_parameters():
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
    in_channels_stage2 = resnet34_backbone.in_channel // 8  # 256
    # 各层输入通道数列表 [256, 512, 1024, 2048]对应layer1-4
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    # FPN输出通道数(统一为256)
    out_channels = 256
    return BackboneWithFPN(resnet34_backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks)
