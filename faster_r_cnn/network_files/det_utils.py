
import torch
import math
from typing import List, Tuple
from torch import Tensor


class BalancedPositiveNegativeSampler(object):
    """
    这个类用于采样批次数据，确保每个批次包含固定比例的正样本
    """
    def __init__(self, batch_size_per_image, positive_fraction):
        # type: (int, float) -> None
        """
        全局参数初始化
        :param batch_size_per_image: 每张图像要选择的样本数量 256
        :param positive_fraction: 每个批次中正样本的比例 0.5
        """
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def __call__(self, matched_idxs):
        # type: (List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        """

        :param matched_idxs: {list:2}:(65472,)(65472,)（anchor的标签）前景（正样本处标记为1），背景（负样本处标记为0），废弃的anchors（丢弃样本处标记为-1）
        :return:
        """
        # 存储每张图像的正样本掩码
        pos_idx = []
        # 存储每张图像的负样本掩码
        neg_idx = []
        # 遍历每张图像的matched_idxs
        for matched_idxs_per_image in matched_idxs:
            # 获取正样本索引(>=1的为正样本)
            positive = torch.where(torch.ge(matched_idxs_per_image, 1))[0]
            # 获取负样本索引(=0的为负样本)
            negative = torch.where(torch.eq(matched_idxs_per_image, 0))[0]
            # 指定正样本的数量
            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            # 如果正样本数量不够就直接采用所有正样本，取最小值
            num_pos = min(positive.numel(), num_pos)
            # 计算负样本数量(总样本数-正样本数)
            num_neg = self.batch_size_per_image - num_pos
            # 如果负样本数量不够就直接采用所有负样本，取最小值
            num_neg = min(negative.numel(), num_neg)
            # 随机选择指定数量的正负样本，并且生成随机排列的索引
            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]
            # 根据随机索引选择样本
            pos_idx_per_image = positive[perm1]
            neg_idx_per_image = negative[perm2]

            # 创建正样本和负样本的二进制掩码(后面分别对这个正样本和负样本进行0/1表示是否被选中)
            pos_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )
            neg_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )
            # 将选中的位置设置为1
            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1
            # 添加到结果列表
            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)

        return pos_idx, neg_idx


@torch.jit._script_if_tracing
def encode_boxes(reference_boxes, proposals, weights):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
    """
    reference_boxes[x_1, y_1, x_2, y_2] ; proposals[x_min, y_min, x_max, y_max] ; weights[wx, wy, ww, wh]
    1、计算真实边界框的宽、高、中心点x、y
        gt_widths = x_2 - x_1
        gt_heights = y_2 - y_1
        gt_ctr_x = x_1 + (0.5 * gt_widths)
        gt_ctr_y = y_1 + (0.5 * gt_heights)
    2、计算anchor的宽、高、中心点x、y
        proposals_widths = x_max - x_min
        proposals_heights = y_max - y_min
        proposals_ctr_x = x_min + (0.5 * proposals_widths)
        proposals_ctr_y = y_min + (0.5 * proposals_heights)
    3、计算中心点x、y的偏移量，宽度、高度偏移量
        中心x坐标偏移 targets_dx = wx * (gt_ctr_x - proposals_ctr_x) / proposals_widths
        中心y坐标偏移 targets_dy = wy * (gt_ctr_y - proposals_ctr_y) / proposals_heights
        宽度偏移 targets_dw = ww * torch.log(gt_widths / proposals_widths)
        高度偏移 targets_dh = wh * torch.log(gt_heights / proposals_heights)
    :param reference_boxes: 拼接在一起的匹配的gt boxes坐标
    :param proposals: 拼接在一起的batch中的每张图像生成anchors
    :param weights: 放大的权重
    :return:
    """
    # 解包权重（1., 1., 1., 1）
    wx = weights[0]
    wy = weights[1]
    ww = weights[2]
    wh = weights[3]
    # 提取提议框坐标并增加维度
    proposals_x1 = proposals[:, 0].unsqueeze(1)
    proposals_y1 = proposals[:, 1].unsqueeze(1)
    proposals_x2 = proposals[:, 2].unsqueeze(1)
    proposals_y2 = proposals[:, 3].unsqueeze(1)
    # 提取参考框坐标并增加维度
    reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
    reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1)
    reference_boxes_x2 = reference_boxes[:, 2].unsqueeze(1)
    reference_boxes_y2 = reference_boxes[:, 3].unsqueeze(1)
    # 计算提议框的宽度、高度和中心坐标
    ex_widths = proposals_x2 - proposals_x1
    ex_heights = proposals_y2 - proposals_y1
    ex_ctr_x = proposals_x1 + 0.5 * ex_widths
    ex_ctr_y = proposals_y1 + 0.5 * ex_heights
    # 计算参考框的宽度、高度和中心坐标
    gt_widths = reference_boxes_x2 - reference_boxes_x1
    gt_heights = reference_boxes_y2 - reference_boxes_y1
    gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths
    gt_ctr_y = reference_boxes_y1 + 0.5 * gt_heights
    # 计算回归目标
    # 中心x坐标偏移
    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    # 中心y坐标偏移
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    # 宽度偏移
    targets_dw = ww * torch.log(gt_widths / ex_widths)
    # 高度偏移
    targets_dh = wh * torch.log(gt_heights / ex_heights)
    # 拼接所有回归目标
    targets = torch.cat((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
    return targets


class BoxCoder(object):
    """
    这个类将边界框编码为回归训练使用的表示形式，并解码回原始坐标
    """
    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        # type: (Tuple[float, float, float, float], float) -> None
        """

        :param weights: 回归权重 weights=(1.0, 1.0, 1.0, 1.0)
        :param bbox_xform_clip: 限制回归参数的最大值 4.135166556742356
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes, proposals):
        # type: (List[Tensor], List[Tensor]) -> List[Tensor]
        """
        将参考框(reference_boxes)编码为相对于提议框(proposals)的偏移量
        :param reference_boxes: 匹配的gt boxes坐标
        :param proposals: batch中的每张图像生成anchors
        :return:
        """
        # reference_boxes和proposal数据结构相同
        # 统计每张图像的anchors数量
        boxes_per_image = [len(b) for b in reference_boxes]
        # 将所有图像的框拼接在一起处理
        reference_boxes = torch.cat(reference_boxes, dim=0)
        proposals = torch.cat(proposals, dim=0)
        # 计算回归参数
        targets = self.encode_single(reference_boxes, proposals)
        return targets.split(boxes_per_image, 0)

    def encode_single(self, reference_boxes, proposals):
        """
        根据真实边界框与anchor计算偏移量
        :param reference_boxes: 拼接在一起的匹配的gt boxes坐标
        :param proposals: 拼接在一起的batch中的每张图像生成anchors
        :return:
        """
        # 获取类型和设备
        dtype = reference_boxes.dtype
        device = reference_boxes.device
        # 将权重转换为tensor
        weights = torch.as_tensor(self.weights, dtype=dtype, device=device)
        # 调用编码函数
        targets = encode_boxes(reference_boxes, proposals, weights)

        return targets

    def decode(self, rel_codes, boxes):
        # type: (Tensor, List[Tensor]) -> Tensor
        """
        将回归参数解码为边界框坐标
        :param rel_codes: bbox回归参数[130944, 4]
        :param boxes: anchors/proposals列表每一个特征层上的anchor映射到原图上的尺寸{list:2}(65472, 4)(65472, 4)
        :return:
        """
        assert isinstance(boxes, (list, tuple))
        assert isinstance(rel_codes, torch.Tensor)
        # 统计每张图像的框数量[65472, 65472]
        boxes_per_image = [b.size(0) for b in boxes]
        # 拼接所有框[130944, 4]
        concat_boxes = torch.cat(boxes, dim=0)
        box_sum = 0
        # 统计图像框的总数量
        for val in boxes_per_image:
            box_sum += val
        # 解码回归参数
        pred_boxes = self.decode_single(rel_codes, concat_boxes)
        # 防止pred_boxes为空时导致reshape报错
        if box_sum > 0:
            pred_boxes = pred_boxes.reshape(box_sum, -1, 4)

        return pred_boxes

    def decode_single(self, rel_codes, boxes):
        """
        从原始框和编码的相对框偏移量中获取解码后的框
        bboxes[x_min, y_min, x_max, y_max] ; rel_codes[dx, dy, dw, dh]
        1、widths = x_max - x_min ; heights = y_max - y_min
        2、ctr_x = x_min + (0.5 * widths) ; ctr_y = y_min + (0.5 * heights)
        3、pred_ctr_x = ctr_x + (dx * widths) ; pred_ctr_y = ctr_y + (dy * heights)
          pred_w = e^dw * widths ; pred_h = e^dh * heights
        4、x_min = pred_ctr_x - (0.5 * pred_w)
          y_min = pred_ctr_y - (0.5 * pred_h)
          x_max = pred_ctr_x + (0.5 * pred_w)
          y_max = pred_ctr_y + (0.5 * pred_h)
        编码后的结果[x_min, y_min, x_max, y_max]
        :param rel_codes: 编码的框(bbox回归参数)[130944, 4]
        :param boxes: 参考框(anchors/proposals)[130944, 4]
        :return:
        """
        boxes = boxes.to(rel_codes.dtype)
        # 计算参考框的宽度、高度和中心坐标
        # anchor/proposal宽度
        widths = boxes[:, 2] - boxes[:, 0]
        # anchor/proposal高度
        heights = boxes[:, 3] - boxes[:, 1]
        # anchor/proposal中心x坐标
        ctr_x = boxes[:, 0] + 0.5 * widths
        # anchor/proposal中心y坐标
        ctr_y = boxes[:, 1] + 0.5 * heights
        # 获取权重
        # RPN中为[1,1,1,1], fastrcnn中为[10,10,5,5]
        wx, wy, ww, wh = self.weights
        # 预测anchors/proposals的中心坐标x回归参数
        dx = rel_codes[:, 0::4] / wx
        # 预测anchors/proposals的中心坐标y回归参数
        dy = rel_codes[:, 1::4] / wy
        # 预测anchors/proposals的宽度回归参数
        dw = rel_codes[:, 2::4] / ww
        # 预测anchors/proposals的高度回归参数
        dh = rel_codes[:, 3::4] / wh
        # 限制最大值，防止输入到torch.exp()的值过大
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)
        # 计算预测框的中心坐标和宽高
        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]
        # 计算预测框的坐标(xmin, ymin, xmax, ymax)
        # xmin
        pred_boxes1 = pred_ctr_x - torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
        # ymin
        pred_boxes2 = pred_ctr_y - torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
        # xmax
        pred_boxes3 = pred_ctr_x + torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
        # ymax
        pred_boxes4 = pred_ctr_y + torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
        # 堆叠并展平结果
        pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=2).flatten(1)
        return pred_boxes


class Matcher(object):
    """
    匹配器
    """
    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2
    __annotations__ = {
        'BELOW_LOW_THRESHOLD': int,
        'BETWEEN_THRESHOLDS': int,
    }

    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        # type: (float, float, bool) -> None
        """

        :param high_threshold: 大于或等于此值的质量值为候选匹配 0.7
        :param low_threshold: 用于将匹配分为三个级别的较低质量阈值:
                                                            1) 匹配 >= high_threshold
                                                            2) BETWEEN_THRESHOLDS 匹配在[low_threshold, high_threshold)
                                                            3) BELOW_LOW_THRESHOLD 匹配在[0, low_threshold)
        :param allow_low_quality_matches: 如果为True，则为只有低质量匹配候选项的预测生成额外匹配
        """
        self.BELOW_LOW_THRESHOLD = -1
        self.BETWEEN_THRESHOLDS = -2
        assert low_threshold <= high_threshold  # 判断低的一定比高的小
        self.high_threshold = high_threshold  # 0.7
        self.low_threshold = low_threshold    # 0.3
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        """

        :param match_quality_matrix: MxN张量，包含M个gt元素和N个预测元素之间的成对质量(IoU)默认都为0
        :return: IOU小于低阈值为-1，IoU在两个阈值之间为-2，IOU大于高阈值为0
        """
        # 检查输入是否为空
        if match_quality_matrix.numel() == 0:
            if match_quality_matrix.shape[0] == 0:
                raise ValueError("训练期间不支持空目标")
            else:
                raise ValueError("训练期间不支持空提议框")

        # match_quality_matrix是 M(gt) x N(预测) 的矩阵
        # 对每个预测框，找到与其IoU最大的gt框，在真实框维度（第 0 维）上取最大值
        matched_vals, matches = match_quality_matrix.max(dim=0)
        # 如果需要保留低质量匹配，则保存原始匹配结果
        if self.allow_low_quality_matches:
            all_matches = matches.clone()
        else:
            all_matches = None

        # 将低质量的匹配标记为负值，IOU小于低阈值
        below_low_threshold = matched_vals < self.low_threshold
        # IoU在两个阈值之间
        between_thresholds = (matched_vals >= self.low_threshold) & (matched_vals < self.high_threshold)
        # iou小于low_threshold的matches索引置为-1
        matches[below_low_threshold] = self.BELOW_LOW_THRESHOLD
        # iou在[low_threshold, high_threshold]之间的matches索引置为-2
        matches[between_thresholds] = self.BETWEEN_THRESHOLDS
        # 如果需要，设置低质量匹配
        if self.allow_low_quality_matches:
            assert all_matches is not None
            self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)

        return matches

    def set_low_quality_matches_(self, matches, all_matches, match_quality_matrix):
        """
        为只有低质量匹配的预测生成额外匹配。对于每个gt框，找到与其IOU最大的预测框集合；对于该集合中的每个预测框，如果它未被匹配，则将其匹配到与其质量值最高的gt框。
        :param matches:
        :param all_matches:
        :param match_quality_matrix:
        :return:
        """
        # 对于每个gt框，找到与其IOU最大的预测框，在锚框维度（第 1 维）上取最大值
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        # 找到最高质量的匹配(包括低质量匹配)
        gt_pred_pairs_of_highest_quality = torch.where(torch.eq(match_quality_matrix, highest_quality_foreach_gt[:, None]))
        # 获取预测框索引
        pre_inds_to_update = gt_pred_pairs_of_highest_quality[1]
        # 更新匹配结果，保留这些预测框与gt的最大IOU匹配
        matches[pre_inds_to_update] = all_matches[pre_inds_to_update]


def smooth_l1_loss(input, target, beta: float = 1. / 9, size_average: bool = True):
    """
    PyTorch的smooth_l1_loss，但带有额外的beta参数
    :param input: 预测值，正样本的预测的bbox回归参数[31, 4]维度
    :param target: 目标值，正样本的真实的bbox回归目标，gt与anchor计算的中心点偏移量、高宽偏移量[31, 4]维度
    :param beta: 控制平滑区域的阈值
    :param size_average: 是否对损失取平均
    :return:
    """
    # 计算绝对差值
    n = torch.abs(input - target)
    # 判断差值是否小于beta
    cond = torch.lt(n, beta)
    # 根据条件计算损失
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    # 根据size_average决定返回均值或总和
    if size_average:
        return loss.mean()
    return loss.sum()
