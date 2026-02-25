


import time
import os
import datetime
import torch
import transforms
from my_dataset import MyVOCDataSet
from backbone import resnet50_fpn_backbone
from network_files import FasterRCNN, FastRCNNPredictor
import train_utils.train_eval_utils as utils
from train_utils import GroupedBatchSampler, create_aspect_ratio_groups, init_distributed_mode, save_on_master, mkdir


def create_model(num_classes):
    """
    创建Faster R-CNN模型
    :param num_classes: 类别数（包括背景）
    :return: 初始化的Faster R-CNN模型
    """
    # 创建ResNet50-FPN骨干网络
    # norm_layer: 归一化层使用BatchNorm2d，如果显存很小，建议使用默认的FrozenBatchNorm2d
    # trainable_layers: 可训练层数（这里设置为3层）
    # trainable_layers包括['layer4', 'layer3', 'layer2', 'layer1', 'conv1']， 5代表全部训练
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d, trainable_layers=5)
    # 创建Faster R-CNN模型（初始类别数为91 - COCO类别数）
    model = FasterRCNN(backbone=backbone, num_classes=91)
    # 载入预训练模型权重
    # https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
    weights_dict = torch.load("./backbone/fasterrcnn_resnet50_fpn_coco.pth", map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
    # 打印缺失和意外的键（如果有）
    if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)

    # 获取分类头的输入特征维度
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # 替换预训练的分类头为新的分类头（适应自定义类别数）
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def main(args):
    """主训练函数"""
    # 初始化分布式训练环境
    init_distributed_mode(args)
    print(args)     # 打印参数
    # 设置训练设备
    device = torch.device(args.device)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # 创建结果文件名（包含时间戳），用来保存coco_info的文件
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    # 数据加载
    print("加载数据")
    # 数据转换定义（训练和验证使用不同的转换）
    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }
    # 检查VOC数据集路径
    VOC_root = args.data_path
    if os.path.exists(os.path.join(VOC_root, "VOCdevkit")) is False:
        raise FileNotFoundError("VOCdevkit dose not in path:'{}'.".format(VOC_root))
    # 加载训练数据集
    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> train.txt
    train_dataset = MyVOCDataSet(voc_root=VOC_root,
                                 year="2012",
                                 txt_type="train.txt",
                                 transforms=data_transform["train"])
    # 加载验证数据集
    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> val.txt
    val_dataset = MyVOCDataSet(voc_root=VOC_root,
                               year="2012",
                               txt_type="val.txt",
                               transforms=data_transform["val"])
    print("数据加载完成")
    # 分布式训练设置采样器
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        # 单机训练使用随机采样器（训练）和顺序采样器（验证）
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        test_sampler = torch.utils.data.SequentialSampler(val_dataset)
    # 宽高比分组采样（如果启用）
    if args.aspect_ratio_group_factor >= 0:
        # 创建宽高比分组
        group_ids = create_aspect_ratio_groups(train_dataset, k=args.aspect_ratio_group_factor)
        # 创建分组批采样器
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        # 普通批采样器
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)

    # 创建训练数据加载器
    data_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_sampler=train_batch_sampler,
                    num_workers=args.workers,                   # 工作进程数
                    collate_fn=train_dataset.collate_fn         # 自定义批处理函数
    )

    # 创建验证数据加载器
    data_loader_test = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=1,               # 验证批次大小为1
                    sampler=test_sampler,
                    num_workers=args.workers,
                    collate_fn=train_dataset.collate_fn
    )

    print("创建模型")
    # 类别数 = 背景类 + 目标类别数
    model = create_model(num_classes=args.num_classes + 1)
    weights_path = "./multi_train/model_best.pth"
    assert os.path.exists(weights_path), "{} file does not exist.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)            # 移至设备

    # 分布式训练使用同步批归一化（如果启用）
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # 非分布式模型引用（用于保存）
    model_without_ddp = model
    if args.distributed:
        # 分布式数据并行包装
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # 获取需要训练的参数
    params = [p for p in model.parameters() if p.requires_grad]
    # 创建SGD优化器
    optimizer = torch.optim.SGD(
                    params,
                    lr=args.lr,                         # 学习率
                    momentum=args.momentum,             # 动量
                    weight_decay=args.weight_decay      # 权重衰减
    )

    # 混合精度训练梯度缩放器（如果启用）
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # 学习率调度器（多步衰减）
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                                                        optimizer,
                                                        milestones=args.lr_steps,           # 学习率衰减的epoch
                                                        gamma=args.lr_gamma                 # 衰减因子
    )

    # 如果传入resume参数，即上次训练的权重地址，则接着上次的参数训练
    if args.resume:
        # 加载检查点（在CPU上加载）
        checkpoint = torch.load(args.resume, map_location='cpu')  # 读取之前保存的权重文件(包括优化器以及学习率策略)
        # 加载模型状态
        model_without_ddp.load_state_dict(checkpoint['model'])
        # 加载优化器状态
        optimizer.load_state_dict(checkpoint['optimizer'])
        # 加载学习率调度器状态
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        # 设置起始epoch
        args.start_epoch = checkpoint['epoch'] + 1
        # 加载混合精度缩放器状态（如果启用）
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

    # 仅测试模式（不训练）
    if args.test_only:
        utils.evaluate(model, data_loader_test, device=device)
        return

    # 初始化训练记录
    train_loss = []         # 训练损失记录
    learning_rate = []      # 学习率记录
    val_map = []            # 验证集mAP记录

    print("开始训练")
    best_map = 0.0  # 初始化为0
    start_time = time.time()
    # 开始训练循环
    for epoch in range(args.start_epoch, args.epochs):
        # 分布式训练设置epoch（确保每个epoch数据顺序不同）
        # if args.distributed:
        #     train_sampler.set_epoch(epoch)

        # # 训练一个epoch
        # mean_loss, lr = utils.train_one_epoch(
        #                             model,
        #                             optimizer,
        #                             data_loader,
        #                             device,
        #                             epoch,
        #                             args.print_freq,
        #                             warmup=True,
        #                             scaler=scaler
        # )

        # # 记录训练损失和学习率
        # train_loss.append(mean_loss.item())
        # learning_rate.append(lr)

        # # 更新学习率
        # lr_scheduler.step()

        # 在验证集上评估
        coco_info = utils.evaluate(model, data_loader_test, device=device)
        # 记录验证集mAP（这里使用Pascal mAP）
    #     val_map.append(coco_info)  # pascal mAP

    #     # 写入结果文件
    #     with open(results_file, "a") as f:
    #         # 写入的数据包括coco指标还有loss和learning rate
    #         result_info = [f"{coco_info:.4f}"] + [f"{mean_loss.item():.6f}"] + [f"{lr:.6f}"]
    #         txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
    #         f.write(txt + "\n")

    #     # 保存模型检查点
    #     if args.output_dir:
    #         # 只在主节点上执行保存权重操作
    #         save_files = {
    #             'model': model_without_ddp.state_dict(),        # 模型权重
    #             'optimizer': optimizer.state_dict(),            # 优化器状态
    #             'lr_scheduler': lr_scheduler.state_dict(),      # 学习率调度器状态
    #             'args': args,                                   # 训练参数
    #             'epoch': epoch                                  # 当前epoch
    #         }
    #         # 保存混合精度缩放器状态（如果启用）
    #         if args.amp:
    #             save_files["scaler"] = scaler.state_dict()
    #         # 主进程保存模型
    #         if coco_info > best_map:
    #             best_map = coco_info
    #             best_epoch = epoch
    #             save_on_master(save_files, os.path.join(args.output_dir, f'model_best.pth'))
    #             print(f"New best model saved with mAP: {best_map:.4f} at epoch {best_epoch}")
    # # 计算总训练时间
    # total_time = time.time() - start_time
    # total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    # print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # 训练文件的根目录(VOCdevkit)
    parser.add_argument('--data-path', default='./', help='dataset')
    # 训练设备类型
    parser.add_argument('--device', default='cuda', help='device')
    # 检测目标类别数(不包含背景)
    parser.add_argument('--num-classes', default=3, type=int, help='num_classes')
    # 每块GPU上的batch_size
    parser.add_argument('-b', '--batch-size', default=16, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=40, type=int, metavar='N',
                        help='number of total epochs to run')
    # 数据加载以及预处理的线程数
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    # 学习率，这个需要根据gpu的数量以及batch_size进行设置0.02 / 8 * num_GPU
    parser.add_argument('--lr', default=0.02, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    # SGD的momentum参数
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    # SGD的weight_decay参数
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # 针对torch.optim.lr_scheduler.StepLR的参数
    parser.add_argument('--lr-step-size', default=8, type=int, help='decrease lr every step-size epochs')
    # 针对torch.optim.lr_scheduler.MultiStepLR的参数
    parser.add_argument('--lr-steps', default=[20, 30, 40], nargs='+', type=int, help='decrease lr every step-size epochs')
    # 针对torch.optim.lr_scheduler.MultiStepLR的参数
    parser.add_argument('--lr-gamma', default=0.8, type=float, help='decrease lr by a factor of lr-gamma')
    # 训练过程打印信息的频率
    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')
    # 文件保存地址
    parser.add_argument('--output-dir', default='./multi_train', help='path where to save')
    # 基于上次的训练结果接着训练
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    # 不训练，仅测试
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    # 开启的进程数(注意不是线程)
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--sync-bn", dest="sync_bn", help="Use sync batch norm", type=bool, default=False)
    # 是否使用混合精度训练(需要GPU支持混合精度)
    parser.add_argument("--amp", default=False, help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    # 如果指定了保存文件地址，检查文件夹是否存在，若不存在，则创建
    if args.output_dir:
        mkdir(args.output_dir)

    main(args)
