
from torch import nn
import torch


class VGG(nn.Module):
    def __init__(self, features, class_num=1000, init_weights=False, weights_path=None):
        """
        :param features: 一系列的网络结构
        :param class_num: 类别数量
        :param init_weights: 是否自行初始化权重
        :param weights_path: 权重的路径
        """
        super(VGG, self).__init__()
        # 将前半部分网络结构赋值给全局变量,后面直接调用即可
        self.features = features
        # 定义顺序容器，用来存储VGG网络结构的后面全连接操作，前面卷积池化操作在features里面
        self.classifier = nn.Sequential(
            # 尾部第一个全连接
            nn.Linear(512 * 7 * 7, 4096),
            # 这里使用ReLU激活函数，设置为True是对数据进行原地操作，不用重新赋值一个变量，直接在原数据上面进行操作
            nn.ReLU(True),
            # 随机将50%的神经元输出设置为0
            nn.Dropout(0.5),
            # 第二个全连接
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            # 第三个全连接，输出预测类别个数个数值
            nn.Linear(4096, class_num)
        )
        # 此处根据是否设置了自行初始化权重和权重路径，如果init_weights=True并且weights_path为空
        # 那么就说明需要自行初始化权重
        if init_weights and weights_path is None:
            # 调用方法来初始化权重
            self._initialize_weights()
        # 如果设置了权重路径，那么就加载权重
        if weights_path is not None:
            # 这里加载外部的预训练权重
            self.load_state_dict(torch.load(weights_path))

    def forward(self, x):
        # 使用构建的网络结构对输入值进行特征提取，N x 3 x 224 x 224
        x = self.features(x)
        # 将特征图形状为N x 512 x 7 x 7 按照通道维度，展平为 N x (512 * 7 * 7)
        x = torch.flatten(x)
        # 将展平后的特征向量输入到全连接结构，最后得到输出结果
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """
        对神经网络的各个参数（权重和偏置）进行初始化
        """
        # 此处的self.modules()指的是模型里面的所有模块（包含模型自身和嵌套子模块），此处遍历就是访问从顶层到底层的每一个部分
        for m in self.modules():
            # 当模型为卷积的时候
            if isinstance(m, nn.Conv2d):
                # 使用Xavier方法来初始化卷积层的权重
                # 这里Xavier均匀分布初始化的基本思想是根据输入和输出神经元的数量来确定权重分布范围，使得在正向传播过程中，每层的输入和输出的方差尽量保持一致
                # 从而缓解梯度消失和梯度爆炸问题，进而让网络能够更加平稳的进行训练
                nn.init.xavier_uniform_(m.weight)
                # 还有一种方法初始化，是使用 kaiming 正态分布初始化（何凯明），或者叫He初始化
                # 在某些情况下，如果网络中激活函数以 ReLU 为主时，Kaiming 初始化可能会更合适
                # 它主要是基于激活函数的性质以及卷积层输入和输出通道数等因素来确定权重的正态分布参数，
                # 同样是为了保证每层输入输出的方差稳定等良好的训练特性
                # TODO：详细了解Xavier初始化和kaiming初始化的机制和用法，以及其他权重初始化的方法
                # nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                # 如果卷积网络设置了偏置，那么就设置偏置参数
                if m.bias is not None:
                    # 如果设置了偏置，那么将偏置初始化为0
                    # 偏置初始化为 0 是一种比较常见且简单的做法，在网络开始训练时，
                    # 不会因为偏置的非零初始值引入额外的、可能不合理的偏移，
                    # 让网络可以从相对 “中性” 的状态开始学习输入和输出之间的关系。
                    nn.init.constant_(m.bias, 0)
            # 如果网络为全连接层时
            elif isinstance(m, nn.Linear):
                # 同样采用Xavier初始化权重
                # 目的也是为了保证线性层输入输出的方差稳定性，使得网络训练更加顺畅，
                # 避免因权重初始值不合理导致的梯度问题影响训练效果。
                nn.init.xavier_uniform_(m.weight)
                # 或者使用简单的正态分布，将权重初始化为均值为 0、标准差为 0.01 的正态分布
                # nn.init.normal_(m.weight)
                # 偏置初始化为0
                nn.init.constant_(m.bias, 0)


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def make_feature(cfg: list):
    """
    将传入的VGG网络，根据设计结构写出网络
    :param cfg: 选择的VGG结构
    :return: 根据选择的结构输出的网络
    """
    layers = []
    in_channels = 3
    # 遍历传入的结构
    for v in cfg:
        # 如果当前层网络是最大池化层，那么就定义一个最大池化层
        if v == "M":
            # 定义一个最大池化层，然后添加到网络列表里面
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # 否则就是卷积网络，VGG前半部分不是卷积就是池化
            # 此处循环遍历的v不是M那么就是代表当前层卷积层的卷积核的个数
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            # 这里使用了卷积网络那么就跟着一层激活函数，这里用relu激活函数
            layers += [conv2d, nn.ReLU(True)]
            # 然后将卷积核个数赋值给通道数，这样就能够知道下一层卷积层的通道数，从而连续起来
            in_channels = v
    # 这里layers已经是一个列表了，直接放入nn.Sequential()里面会被当作一个结构，从而达不到一层层的构造
    # 因此这里在layers前面加一个*，代表解包操作，即将layers每一层都给nn.Sequential()
    return nn.Sequential(*layers)


def vgg(model_name='vgg16', weights_path=None):
    # 验证输入选择的模型在不在cfgs里面，如果不在那么就抛出警告
    assert model_name in cfgs, "当前选择的模型{}不在cfgs模型内，请正确输入选择的模型".format(model_name)
    # 根据选择的模型，赋值对应模型的结构
    cfg = cfgs[model_name]
    # 对VGG类输入模型结构，并且设置权重路径
    model = VGG(make_feature(cfg), weights_path=weights_path)
    return model


















