


import warnings
from collections import OrderedDict
from typing import Tuple, List, Dict, Optional, Union
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign
from .roi_head import RoIHeads
from .transform import GeneralizedRCNNTransform
from .rpn_function import AnchorsGenerator, RPNHead, RegionProposalNetwork


class FasterRCNNBase(nn.Module):
    """
    Faster R-CNN 基础类
    """

    def __init__(self, backbone, rpn, roi_heads, transform):
        """
        初始化参数
        :param backbone: 骨干网络 (如ResNet-50)
        :param rpn: 区域建议网络
        :param roi_heads: RoI头部，处理RPN的建议区域
        :param transform: 数据预处理转换模块
        """
        super(FasterRCNNBase, self).__init__()
        # 数据预处理转换模块
        self.transform = transform
        # 骨干网络
        self.backbone = backbone
        # 区域建议网络
        self.rpn = rpn
        # RoI头部
        self.roi_heads = roi_heads
        # 用于脚本模式下的警告
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        在eager模式下选择输出训练损失或检测结果
        """
        # 训练模式返回损失字典
        if self.training:
            return losses
        # 测试模式返回检测结果列表
        return detections

    # 这里传入images是List, targets也是List
    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Faster R-CNN 前向传播
        :param images: 输入图像列表，每张图像为[C, H, W]
        :param targets: 真实标注信息 (可选)
        :return:
            训练模式: 损失字典
            测试模式: 检测结果列表
        """

        # ================ 1. 输入验证 ================
        if self.training and targets is None:
            raise ValueError("训练模式下必须提供targets")

        if self.training:
            assert targets is not None
            for target in targets:
                # 进一步判断传入的target的boxes参数是否符合规定
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("目标边界框应为[N, 4]形状")
                else:
                    raise ValueError("目标边界框应为Tensor类型")

        # ================ 2. 记录原始图像尺寸 ================
        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            # 防止输入的是个一维向量
            assert len(val) == 2
            # 保存原始(H, W)
            original_image_sizes.append((val[0], val[1]))

        # ================ 3. 数据预处理 ================
        # 应用预处理: 标准化、缩放、批处理等
        images, targets = self.transform(images, targets)

        # ================ 4. 骨干网络特征提取 ================
        # 输入: 预处理后的图像张量
        # 输出: 特征图 (单层或多层)
        features = self.backbone(images.tensors)  # 将图像输入backbone得到特征图
        if isinstance(features, torch.Tensor):  # 若只在一层特征层上预测，将feature放入有序字典中，并编号为‘0’
            features = OrderedDict([('0', features)])  # 若在多层特征层上预测，传入的就是一个有序字典

        # ================ 5. RPN区域建议生成 ================
        # 输入: 预处理图像、特征图、真实标注
        # 输出: 建议区域列表、RPN损失
        proposals, proposal_losses = self.rpn(images, features, targets)

        # ================ 6. RoI头部处理 ================
        # 输入: 特征图、建议区域、图像尺寸、真实标注
        # 输出: 检测结果、检测损失
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)

        # ================ 7. 检测结果后处理 ================
        # 将边界框还原到原始图像尺寸
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        # ================ 8. 合并损失 ================
        losses = {}
        # RoI头部损失
        losses.update(detector_losses)
        # RPN损失
        losses.update(proposal_losses)

        # ================ 9. 返回结果 ================
        if torch.jit.is_scripting():
            # Torch脚本模式特殊处理
            if not self._has_warned:
                warnings.warn("RCNN在脚本模式下总是返回(Losses, Detections)元组")
                self._has_warned = True
            return losses, detections
        else:
            # 正常PyTorch模式
            return self.eager_outputs(losses, detections)


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


class FasterRCNN(FasterRCNNBase):
    """
    Faster R-CNN 完整实现
    """

    def __init__(self, backbone, num_classes=None,
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 rpn_score_thresh=0.0,
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None):
        """
        初始化参数
        :param backbone: 骨干网络 (如ResNet-50)
        :param num_classes: 类别数量 (含背景)
        :param min_size: 图像缩放尺寸限制
        :param max_size: 图像缩放尺寸限制
        :param image_mean: 图像标准化参数
        :param image_std: 图像标准化参数
        :param rpn_anchor_generator: RPN参数
        :param rpn_head: RPN参数
        :param rpn_pre_nms_top_n_train: rpn中在nms处理前保留的proposal数(根据score)
        :param rpn_pre_nms_top_n_test: rpn中在nms处理前保留的proposal数(根据score)
        :param rpn_post_nms_top_n_train: rpn中在nms处理后保留的proposal数
        :param rpn_post_nms_top_n_test: rpn中在nms处理后保留的proposal数
        :param rpn_nms_thresh: RPN NMS阈值
        :param rpn_fg_iou_thresh: RPN正样本IOU阈值
        :param rpn_bg_iou_thresh: RPN负样本IOU阈值
        :param rpn_batch_size_per_image: RPN每图样本数
        :param rpn_positive_fraction: RPN正样本比例
        :param rpn_score_thresh: RPN分数阈值
        :param box_roi_pool: RoI头部参数
        :param box_head: RoI头部参数
        :param box_predictor: RoI头部参数
        :param box_score_thresh: 检测分数阈值
        :param box_nms_thresh: 检测NMS阈值
        :param box_detections_per_img: 每图最大检测数
        :param box_fg_iou_thresh: RoI正样本IOU阈值
        :param box_bg_iou_thresh: RoI负样本IOU阈值
        :param box_batch_size_per_image: RoI每图样本数
        :param box_positive_fraction: RoI正样本比例
        :param bbox_reg_weights: 边界框回归权重
        """
        # 验证骨干网络是否有out_channels属性
        if not hasattr(backbone, "out_channels"):
            raise ValueError("骨干网络必须有out_channels属性")
        # 验证参数类型
        assert isinstance(rpn_anchor_generator, (AnchorsGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))
        # 验证类别数和预测器是否匹配
        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("当指定box_predictor时，num_classes应为None")
        else:
            if box_predictor is None:
                raise ValueError("当未指定box_predictor时，必须提供num_classes")

        # 获取骨干网络输出通道数
        out_channels = backbone.out_channels

        # ================ 1. 初始化RPN组件 ================

        # 默认anchor生成器 (针对FPN设计)
        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorsGenerator(
                anchor_sizes, aspect_ratios
            )

        # 默认RPN头部
        if rpn_head is None:
            rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])

        # 设置训练/测试模式下的top_n参数
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        # 初始化RPN网络
        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh,
            score_thresh=rpn_score_thresh)

        # ================ 2. 初始化RoI头部组件 ================

        # 默认RoI Align池化层
        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],  # 在哪些特征层进行roi pooling
                output_size=[7, 7],
                sampling_ratio=2)

        # 默认RoI特征提取头部
        if box_head is None:
            resolution = box_roi_pool.output_size[0]  # 默认等于7
            representation_size = 1024
            box_head = TwoMLPHead(out_channels * resolution ** 2, representation_size)

        # 默认预测器
        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(representation_size, num_classes)

        # 初始化RoI头部
        roi_heads = RoIHeads(
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,  # 0.5  0.5
            box_batch_size_per_image, box_positive_fraction,  # 512  0.25
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)  # 0.05  0.5  100

        # ================ 3. 初始化预处理转换 ================

        # 默认图像标准化参数 (ImageNet)
        if image_mean is None:
            # RGB均值
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            # RGB标准差
            image_std = [0.229, 0.224, 0.225]

        # 初始化预处理模块
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
        # ================ 4. 调用父类初始化 ================
        super(FasterRCNN, self).__init__(backbone, rpn, roi_heads, transform)
