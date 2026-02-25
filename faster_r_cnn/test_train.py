


import torch
from torch import nn
from torch.nn import functional as F
from backbone import resnet50_fpn_backbone
from network_files import AnchorsGenerator
from network_files.rpn_function import RegionProposalNetwork, RPNHead
from my_dataset import MyVOCDataSet
from torchvision import transforms
import torchvision
from torchvision.models.detection.image_list import ImageList
from network_files.roi_head import RoIHeads


class TwoMLPHead(nn.Module):
    """
    ROIPooling的7 x 7矩阵进行展平操作，然后通过两个全连接操作
    """

    def __init__(self, in_channels, representation_size):
        """
        参数初始化
        :param in_channels: 预测特征层的通道数 x ROIPooling尺寸（7 * 7）
        :param representation_size: 1024
        """
        super(TwoMLPHead, self).__init__()
        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        # 展平操作：[N, 256, 7, 7] → [N, 256*49=12544]
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x


class FastRCNNPredictor(nn.Module):
    """
    对得到的特征向量进行分类与回归预测
    """

    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        # [N, 1024] → [N, num_classes]
        scores = self.cls_score(x)
        # [N, 1024] → [N, num_classes*4]
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas


# 实例化这个类，传入各个参数，以及对图像进行的变换
voc_data_set = MyVOCDataSet(voc_root="VOCdevkit",
                            year="2012",
                            txt_type="train.txt",
                            transforms=transforms.Compose([transforms.ToTensor()]))

"""
数据加载器将数据集和采样器组合在一起，并在给定的数据集上提供一个可迭代对象。
torch.utils.data.DataLoader支持映射风格和可迭代风格的数据集，
支持单进程或多进程加载、自定义加载顺序以及可选的自动批处理（排序）和内存固定。
dataset（数据集）：从中加载数据的数据集。
batch_size（int，可选）：每批要加载多少个样本（默认值：“1”）。
shuffle（bool，可选）：设置为“True”以重新排列数据在每个epoch（默认值：“False”）。
num_workers（int，可选）：用于数据的子进程数量装载，0表示数据将加载到主进程中。（默认值：“0”）
drop_last（bool，可选）：设置为“True”以删除最后一个不完整的批，如果数据集大小不能被批大小整除。如果“False”和数据集的大小不能被批大小整除，那么最后一批将更小。（默认值：“False”）
"""
# 这里使用了
data_loader = torch.utils.data.DataLoader(dataset=voc_data_set,
                                          batch_size=2,
                                          shuffle=False,
                                          num_workers=0,
                                          drop_last=False)

backbone = resnet50_fpn_backbone(pretrain_path="./backbone/resnet50.pth",
                                 norm_layer=torch.nn.BatchNorm2d,
                                 trainable_layers=3)
for data in data_loader:
    imgs, targets = data
    # imgs:torch.Size([2, 3, 512, 512])
    forward_feature = backbone(imgs)
    # 将字典形式的 targets 转换为列表形式
    new_targets = []
    for i in range(len(targets['boxes'])):  # 遍历每个图像（batch_size=2）
        img_target = {}
        for k, v in targets.items():
            if k == 'boxes':
                img_target[k] = v[i]  # 去掉多余的 1 维，形状从 (2,1,4) → (4,)（单框）
            else:
                img_target[k] = v[i]  # 其他键（如 labels/area 等）同理处理
        new_targets.append(img_target)  # 每个图像的标注作为列表的一个元素
    # 将输入的图像张量（形状：[batch_size, C, H, W]）使用ImageList 来包装
    image_list = ImageList(imgs, [(img.shape[-2], img.shape[-1]) for img in imgs])

    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    rpn_anchor_generator = AnchorsGenerator(anchor_sizes, aspect_ratios)
    rpn_anchors = rpn_anchor_generator.forward(image_list=image_list, feature_maps=list(forward_feature.values()))
    # rpn_anchor_generator.num_anchors_per_location()
    # 0 = {int} 3
    # 1 = {int} 3
    # 2 = {int} 3
    # 3 = {int} 3
    # 4 = {int} 3
    rpn_head = RPNHead(256, rpn_anchor_generator.num_anchors_per_location()[0])
    rpn_feature = rpn_head.forward(x=list(forward_feature.values()))

    rpn = RegionProposalNetwork(
        anchor_generator=rpn_anchor_generator,
        head=rpn_head,
        fg_iou_thresh=0.7,
        bg_iou_thresh=0.3,
        batch_size_per_image=256,
        positive_fraction=0.5,
        pre_nms_top_n={'training': 2000, 'testing': 1000},
        post_nms_top_n={'training': 2000, 'testing': 1000},
        nms_thresh=0.7,
        score_thresh=0.0
    )

    boxes, losses = rpn.forward(images=image_list, features=forward_feature, targets=new_targets)

    # '0' 通常代表高分辨率/浅层特征（如 P2，1/4 缩放），适合检测小物体
    # '1' 代表中分辨率特征（如 P3，1/8 缩放）
    # '2' 代表低分辨率/深层特征（如 P4，1/16 缩放），适合检测大物体
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2'],  # 在哪些特征层上进行RoIAlign pooling
                                                    output_size=[7, 7],  # RoIAlign pooling输出特征矩阵尺寸
                                                    sampling_ratio=2)  # 采样率
    # 预测特征层的channels默认为256
    out_channels = backbone.out_channels

    # fast RCNN中roi pooling后的展平处理两个全连接层部分
    resolution = roi_pooler.output_size[0]  # 默认等于7
    representation_size = 1024
    # 两层全连接操作提取特征
    box_head = TwoMLPHead(out_channels * resolution ** 2, representation_size)
    # 两层全连接操作进行头部预测
    box_predictor = FastRCNNPredictor(representation_size, num_classes=3)

    roi_heads = RoIHeads(
        roi_pooler, box_head, box_predictor,
        fg_iou_thresh=0.5, bg_iou_thresh=0.5,
        batch_size_per_image=512, positive_fraction=0.25,
        bbox_reg_weights=None,
        score_thresh=0.05, nms_thresh=0.5, detection_per_img=100)
    # 训练模式才有损失，测试模式才有结果
    head_result, head_losses = roi_heads.forward(features=forward_feature, proposals=boxes, image_shapes=image_list.image_sizes, targets=new_targets)

    print("111")
