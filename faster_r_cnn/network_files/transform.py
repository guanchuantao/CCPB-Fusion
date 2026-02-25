import math
import torch
from torch import nn, Tensor
from typing import List, Tuple, Dict, Optional
import torchvision
from .image_list import ImageList


# 这里使用注解，编译忽略，只有在特定场合才使用
@torch.jit.unused
def _resize_image_onnx(image, self_min_size, self_max_size):
    # type: (Tensor, float, float) -> Tensor
    """
    对图像进行给定区间的缩放
    :param image: pytorch的一个tensor图像
    :param self_min_size: 期望图像缩放后达到的最小尺寸
    :param self_max_size: 限制图像缩放后最大尺寸
    :return: 缩放后的图像
    """
    from torch.onnx import operators  # 导入包，用于帮助实现和 ONNX 格式兼容的计算等操作
    # 切片操作，[channel, height, width]，获取数组倒数第二个维度和最后一个维度，同时转换为tensor格式
    im_shape = operators.shape_as_tensor(image)[-2:]
    # 这里分别获取[高，宽]两个数据里面，最小的和最大的，并且转换为浮点型
    min_size = torch.min(im_shape).to(dtype=torch.float32)
    max_size = torch.max(im_shape).to(dtype=torch.float32)
    # 计算缩放因子，它通过取两个比值（self_min_size / min_size 和 self_max_size / max_size）中的最小值来确定。
    # 其中 self_min_size / min_size 可以理解为基于期望最小尺寸来计算的缩放比例，
    # self_max_size / max_size 则是基于最大尺寸限制来计算的缩放比例，
    # 取最小值确保最终的缩放结果既满足最小尺寸要求又不会超出最大尺寸限制。
    scale_factor = torch.min(self_min_size / min_size, self_max_size / max_size)
    # image[None] 操作，这是在图像张量最前面添加了一个维度，将其形状从比如 (C, H, W)变为 (1, C, H, W)
    # 使用双线性插值（mode="bilinear"）的方式对图像进行缩放
    # recompute_scale_factor=True 表示重新计算缩放因子（这里虽然前面已经计算了，但设置这个参数可以让函数根据输入的一些情况再次确认调整缩放因子，以保证更好的缩放效果等）
    # align_corners=False 是双线性插值中的一种常见设置，用于确定插值过程中角点的对齐方式（不同的设置会影响插值结果的准确性等）
    image = torch.nn.functional.interpolate(
        image[None], scale_factor=scale_factor, mode="bilinear", recompute_scale_factor=True,
        align_corners=False)[0]

    return image


def _resize_image(image, self_min_size, self_max_size):
    # type: (Tensor, float, float) -> Tensor
    """
    对图像进行指定区间的缩放
    :param image: pytorch的一个tensor图像
    :param self_min_size: 期望图像缩放后达到的最小尺寸
    :param self_max_size: 限制图像缩放后最大尺寸
    :return: 缩放后的图像
    """
    # 切片操作，[channel, height, width]，获取数组倒数第二个维度和最后一个维度，同时转换为tensor格式
    im_shape = torch.tensor(image.shape[-2:])
    # 这里分别获取[高，宽]两个数据里面，最小的和最大的，并且转换为浮点型
    min_size = float(torch.min(im_shape))
    max_size = float(torch.max(im_shape))
    # 根据指定最小边长和图片最小边长计算缩放比例
    scale_factor = self_min_size / min_size

    # 如果使用该缩放比例计算的图片最大边长大于指定的最大边长
    if max_size * scale_factor > self_max_size:
        # 将缩放比例设为指定最大边长和图片最大边长之比
        scale_factor = self_max_size / max_size
    # interpolate利用插值的方法缩放图片
    # image[None]操作是在最前面添加batch维度[C, H, W] -> [1, C, H, W]
    # bilinear只支持4D Tensor
    image = torch.nn.functional.interpolate(
        image[None], scale_factor=scale_factor, mode="bilinear", recompute_scale_factor=True,
        align_corners=False)[0]

    return image


# 这里使用注解，编译忽略，只有在特定场合才使用
@torch.jit.unused
def _onnx_batch_images(images, size_divisible=32):
    # type: (List[Tensor], int) -> Tensor
    """
    0填充图像边缘
    :param images: 输入的一批图片
    :param size_divisible: 将图像高和宽调整到该数的整数倍
    :return: 填充之后的，打包成一个batch后的tensor数据
    """
    # 空列表，用于存储这批图像在各个维度上的最大尺寸信息
    max_size = []
    # 通过循环遍历图像张量的各个维度
    for i in range(images[0].dim()):
        # 计算这批图像在各个维度（如通道、高度、宽度等）上的最大尺寸信息
        max_size_i = torch.max(torch.stack([img.shape[i] for img in images]).to(torch.float32)).to(torch.int64)
        max_size.append(max_size_i)
    stride = size_divisible
    # 针对高度和宽度这两个维度，先将对应维度的最大尺寸（max_size[1] 和 max_size[2]）转换为 torch.float32 类型
    # 然后除以 stride，再通过 torch.ceil 函数向上取整（例如，如果计算结果是 3.2，向上取整后变为 4）
    # 最后乘以 stride 得到调整后的尺寸，使其能被 size_divisible 整除
    # 并将结果再转换回 torch.int64 类型更新到 max_size 列表中对应的位置
    # 这样就完成了对高度和宽度维度尺寸的调整，使其符合后续填充等操作的要求
    max_size[1] = (torch.ceil((max_size[1].to(torch.float32)) / stride) * stride).to(torch.int64)
    max_size[2] = (torch.ceil((max_size[2].to(torch.float32)) / stride) * stride).to(torch.int64)
    # 将 max_size 列表转换为元组形式，元组形式更便于作为参数传递以及进行一些索引操作等
    max_size = tuple(max_size)
    # 空列表，用于存储填充后的每张图像的张量
    padded_imgs = []
    # 遍历输入的每张图像张量
    for img in images:
        # 通过列表推导式，计算出在每个维度上需要填充的大小
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        # 对图像进行填充
        padded_img = torch.nn.functional.pad(img, [0, padding[2], 0, padding[1], 0, padding[0]])
        # 将填充后的图像张量添加到 padded_imgs 列表中
        padded_imgs.append(padded_img)
    # 填充后的图像张量沿着一个新的维度（(C, H, W) 形式变为 (batch_size, C, H, W) ）进行堆叠
    return torch.stack(padded_imgs)


def resize_boxes(boxes, original_size, new_size):
    """
    对目标进行图像缩放比例进行缩放（这里可以缩放，也可以复原）
    :param boxes:
    :param original_size: 图像缩放前的尺寸
    :param new_size: 图像缩放后的尺寸
    :return:
    """
    # 这里根据缩放后的图像尺寸和缩放前的图像尺寸，计算缩放比率
    ratios = [torch.tensor(s, dtype=torch.float32, device=boxes.device) /
              torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
              for s, s_orig in zip(new_size, original_size)]
    # 解构赋值
    ratios_height, ratios_width = ratios
    # 沿着指定的维度拆分张量，boxes可能携带批次(batch_size, 4)
    x_min, y_min, x_max, y_max = boxes.unbind(1)
    # 缩放
    x_min = x_min * ratios_width
    x_max = x_max * ratios_width
    y_min = y_min * ratios_height
    y_max = y_max * ratios_height
    # 这里将缩放后的值，按照维度为1进行恢复到(batch_size, 4)
    return torch.stack((x_min, y_min, x_max, y_max), dim=1)


def torch_choice(k):
    # type: (list[int]) -> int
    """
    随机选择一个数返回
    :param k: 整数列表
    :return: 从给定的整数列表 k 中随机选择一个元素并返回
    """
    # 从给定的整数列表 k 中随机选择一个元素并返回，这里获取随机编号，类似于random.choice
    # 但是此处这样做是为了让该函数更加容易被TorchScrip编译
    index = int(torch.empty(1).uniform_(0., float(len(k))).item())
    return k[index]


def max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    """
    找到列表里面每一个维度最大值
    :param the_list: 每一个图像的维度信息[(2,3,3),(3,3,3)]
    :return: 列表中每一列的最大的值
    """
    # 先取一个值，然后作为初始值，这样可以不断和后面进行比较，就能找到最大值
    maxes = the_list[0]
    # 遍历第二个及往后数据，因为第一个被拿来初始化了
    for sublist in the_list[1:]:
        # 取出索引和元素值，这里索引会是：0，1，2
        for index, item in enumerate(sublist):
            # 将上一个最大值与当前值最大值比较，然后将大的赋值给maxes
            maxes[index] = max(maxes[index], item)
    return maxes


def batch_images(images, size_divisible=32):
    # type: (List[Tensor], int) -> Tensor
    """
    将图像填充之后返回
    :param images:
    :param size_divisible: （标准单元）将图像的高和宽调整该数的整数倍
    :return: 打包成一个batch的tensor数据
    """
    # 模型导出为 ONNX 格式、或者执行某些需要记录模型运算流程以进行后续分析或转换的操作时
    if torchvision._is_tracing():
        # 进行填充之后打包成一个batch后的tensor数据
        return _onnx_batch_images(images, size_divisible)
    # 分别计算一个batch中所有图片中的最大channel, height, width
    max_size = max_by_axis([list(image.shape) for image in images])
    # 将标准单元改为浮点型，方便后面运算的时候能够保留出小数，从而能够向上取整
    stride = float(size_divisible)
    # 将height向上调整到stride的整数倍
    max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
    # 将width向上调整到stride的整数倍
    max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)
    # 增加一个维度作为图像批次[batch, channel, height, width]
    batch_shape = [len(images)] + max_size
    # 这里初始化一个形状为batch_shape，全部值为0的新张量，属性（CPU or GPU等等）与images相同
    batched_imgs = images[0].new_full(batch_shape, 0)
    # 对原始图片和新的尺寸为0的张量进行遍历
    for img, pad_img in zip(images, batched_imgs):
        # 这里将每一个images里面的图像复制到每一个batched_imgs里面，然后按照左上角对齐，这样就完成了对图片的填充，从而使得每一张图片形状一样
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    return batched_imgs


class GeneralizedRCNNTransform(nn.Module):

    def __init__(self, min_size, max_size, image_mean, image_std):
        """
        :param min_size: 图像的最小边长范围
        :param max_size: 图像的最大边长范围
        :param image_mean: 图像在标准化处理中的均值
        :param image_std: 图像在标准化处理中的方差
        """
        super(GeneralizedRCNNTransform, self).__init__()
        # 这里使用 isinstance() 是用来判断 min_size 是否是 (list, tuple) 类型
        if not isinstance(min_size, (list, tuple)):
            # 如果不是的话就把整数或者字符串转换为包含该字符的元祖
            min_size = (min_size,)
        self.min_size = min_size  # 指定全局变量图像的最小边长范围
        self.max_size = max_size  # 指定全局变量图像的最大边长范围
        self.image_mean = image_mean  # 指定全局变量图像在标准化处理中的均值
        self.image_std = image_std  # 指定全局变量图像在标准化处理中的方差

    def normalize(self, image):
        """
        对图像进行标准化
        :param image: 图像
        :return: 标准化之后的图像
        """
        # 获取image的数据类型 和 image的GPU还是CPU
        dtype, device = image.dtype, image.device
        # 这里将图像的均值和方差 按照图像的类型和运行内存类别，将其转换为tensor格式
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        # 这里使用 [:, None, None] 是为了扩展，额外扩展的值为 1 ，[:, None, None]就会变成[:, 1, 1]
        return (image - mean[:, None, None]) / std[:, None, None]

    def resize(self, image, target):
        # type: (Tensor, Optional[Dict[str, Tensor]]) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]
        """
        对图像和目标进行缩放
        :param image: 输入的图片
        :param target: 输入图片的相关信息（包括bboxes信息）
        :return: 缩放后的图片和缩放bboxes后的图片相关信息
        """
        # 这里image的shape是[channel, height, width]，切片操作，获取数组倒数第二个维度和最后一个维度
        h, w = image.shape[-2:]
        # 训练模式
        if self.training:
            # 指定输入图片的最小边长
            size = float(torch_choice(self.min_size))
        else:
            # 测试模式
            # 指定输入图片的最小边长
            size = float(self.min_size[-1])
        # 对图像进行缩放
        if torchvision._is_tracing():
            # 模型导出为 ONNX 格式、或者执行某些需要记录模型运算流程以进行后续分析或转换的操作时
            image = _resize_image_onnx(image, size, float(self.max_size))
        else:
            image = _resize_image(image, size, float(self.max_size))
        # 如果目标为空直接返回，否则对目标也进行缩放
        if target is None:
            return image, target
        # 取出目标里面的边界框
        bbox = target["boxes"]
        # 这里根据图像缩放情况对目标进行缩放
        bbox = resize_boxes(bbox, [h, w], image.shape[-2:])
        # 赋值给目标
        target["boxes"] = bbox
        return image, target

    def postprocess(self, result, image_shapes, original_image_size):
        # type: (List[Dict[str, Tensor]], List[Tuple[int, int]], List[Tuple[int, int]]) -> List[Dict[str, Tensor]]
        """
        将预测后的图像和目标复原
        :param result:
        :param image_shapes:
        :param original_image_size:
        :return:
        """
        # 训练模式直接返回结果
        if self.training:
            return result
        # 对每一个图片的预测结果，并且将目标还原到原来图像应有的尺寸
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_size)):
            # 这里先把目标取出来
            boxes = pred["boxes"]
            # 这里把目标从现在尺寸映射到原始尺寸(resize_boxes可以缩放，也可以复原)
            boxes = resize_boxes(boxes, im_s, o_im_s)
            # 复原的目标填入结果集中
            result[i]["boxes"] = boxes
        return result

    def forward(self,
                images,  # type: List[Tensor]
                targets=None  # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[ImageList, Optional[List[Tuple[str, Tensor]]]]
        """
        对图像变换的前向传播
        :param images:
        :param target:
        :return:
        """
        # 这里将传入的图片取出来
        images = [img for img in images]
        # 对图像进行遍历
        for i in range(len(images)):
            # 取出图片
            image = images[i]
            # 如果目标列表不为空，那么取出图片对应的目标
            target_index = targets[i] if targets is not None else None
            # 这里检查图像尺寸，如果不是3通道，报错打印日志
            if image.dim() != 3:
                raise ValueError("图像尺寸有误，不是3通道，该图像尺寸为{}".format(image.shape))
            # 前向传播标准化
            image = self.normalize(image)
            # 前向传播缩放图像和目标
            image, target_index = self.resize(image, target_index)
            # 将标准化和缩放后的图像重新传递给图像列表
            images[i] = image
            # 如果该循环的图片有对应的目标，目标不为空的话，那么将同时缩放的目标复制给原目标列表
            if targets is not None and target_index is not None:
                targets[i] = target_index
        # 这里记录一下图像经过缩放后的尺寸（是为了方便后面进行还原）
        image_sizes = [img.shape[-2:] for img in images]
        # 前向传播打包图像成一个batch
        images = batch_images(images)
        # 用能够添加类型注释型的函数生成空列表，用来后面存储图像缩放后的尺寸
        image_sizes_list = torch.jit.annotate(List[Tuple[int, int]], [])
        # 这里对图像缩放后的尺寸列表进行遍历
        for image_size in image_sizes:
            # 如果尺寸长度不足2那么报错
            assert len(image_size) == 2
            # 这里将每一个图像的高和宽作为一个元祖，然后添加到列表中
            image_sizes_list.append((image_size[0], image_size[1]))
        # 这里将图像的缩放后的尺寸与图像放在一起
        image_list = ImageList(images, image_sizes_list)
        # 返回图像、目标
        return image_list, targets
