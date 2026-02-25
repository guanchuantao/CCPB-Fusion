


from collections import defaultdict, deque
import datetime
import pickle
import time
import errno
import os

import torch
import torch.distributed as dist


class SmoothedValue(object):
    """
    平滑值跟踪器，用于记录一系列值并提供窗口平滑值或全局平均值
    """
    def __init__(self, window_size=20, fmt=None):
        """
        初始化平滑值跟踪器
        :param window_size: 滑动窗口大小
        :param fmt: 格式化字符串
        """
        if fmt is None:
            # 默认格式
            fmt = "{value:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)      # 双端队列，作为滑动窗口
        self.total = 0.0        # 总值
        self.count = 0          # 计数
        self.fmt = fmt          # 格式化字符串

    def update(self, value, n=1):
        """
        更新值
        :param value:
        :param n:
        :return:
        """
        self.deque.append(value)        # 添加到滑动窗口
        self.count += n                 # 更新计数
        self.total += value * n         # 更新总值

    def synchronize_between_processes(self):
        """
        在分布式进程间同步计数和总值
        """
        # 如果不是分布式环境则返回
        if not is_dist_avail_and_initialized():
            return
        # 创建张量并同步
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()           # 同步所有进程
        dist.all_reduce(t)       # 所有进程求和
        # 更新值
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):  # @property 是装饰器，这里可简单理解为增加median属性(只读)
        """计算滑动窗口中的中位数"""
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        """计算滑动窗口中的平均值"""
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        """计算全局平均值"""
        return self.total / self.count

    @property
    def max(self):
        """获取滑动窗口中的最大值"""
        return max(self.deque)

    @property
    def value(self):
        """获取最新值"""
        return self.deque[-1]

    def __str__(self):
        """字符串表示"""
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def all_gather(data):
    """
    在任意可pickle的数据上执行all_gather操作（不仅是张量）
    :param data: 任意可pickle的对象
    :return:
            list[data]: 从每个rank收集的数据列表
    """
    # 获取世界大小（进程数）
    world_size = get_world_size()
    if world_size == 1:     # 如果是单进程
        return [data]

    # 将数据序列化为字节张量
    buffer = pickle.dumps(data)         # 序列化
    storage = torch.ByteStorage.from_buffer(buffer)     # 创建存储
    tensor = torch.ByteTensor(storage).to("cuda")       # 创建CUDA张量

    # 获取每个rank的张量大小
    local_size = torch.tensor([tensor.numel()], device="cuda")      # 本地大小
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]       # 大小列表
    dist.all_gather(size_list, local_size)          # 收集所有rank的大小
    size_list = [int(size.item()) for size in size_list]        # 转换为整数列表
    max_size = max(size_list)       # 最大大小

    # 从所有rank接收张量
    tensor_list = []    # 张量列表
    for _ in size_list:
        # 创建空张量占位
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))

    # 如果本地大小小于最大大小，填充张量
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)

    # 收集所有rank的张量
    dist.all_gather(tensor_list, tensor)
    # 反序列化数据
    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]      # 获取字节数据
        data_list.append(pickle.loads(buffer))              # 反序列化

    return data_list


def reduce_dict(input_dict, average=True):
    """
    在分布式环境中减少字典中的值
    :param input_dict: 需要减少的字典
    :param average: 是否进行平均（否则求和）
    :return: 减少后的字典
    """
    world_size = get_world_size()
    if world_size < 2:  # 单进程直接返回
        return input_dict

    with torch.no_grad():  # 不需要梯度计算
        names = []      # 键列表
        values = []     # 值列表

        # 对键排序以确保跨进程一致性
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])

        # 堆叠值并执行all_reduce
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)     # 求和

        if average:     # 如果需要平均
            values /= world_size

        # 重建字典
        reduced_dict = {k: v for k, v in zip(names, values)}
        return reduced_dict


class MetricLogger(object):
    """
    指标记录器，用于记录和打印训练指标
    """
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)        # 使用默认字典存储平滑值
        self.delimiter = delimiter      # 分隔符

    def update(self, **kwargs):
        """更新指标"""
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()        # 张量转换为标量
            assert isinstance(v, (float, int))      # 确保是数值类型
            self.meters[k].update(v)                # 更新平滑值

    def __getattr__(self, attr):
        """属性访问重载"""
        if attr in self.meters:
            return self.meters[attr]        # 返回对应的平滑值
        if attr in self.__dict__:
            return self.__dict__[attr]      # 返回自身属性
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        """字符串表示"""
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))      # 格式化为字符串
        return self.delimiter.join(loss_str)        # 用分隔符连接

    def synchronize_between_processes(self):
        """在进程间同步所有平滑值"""
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        """添加新的平滑值记录器"""
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        """定期记录指标"""
        i = 0
        if not header:
            header = ""         # 默认空头部

        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')          # 迭代时间平滑值
        data_time = SmoothedValue(fmt='{avg:.4f}')          # 数据加载时间平滑值

        # 格式化数字空间
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"

        # 构建日志消息格式
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([header,
                                           '[{0' + space_fmt + '}/{1}]',
                                           'eta: {eta}',
                                           '{meters}',
                                           'time: {time}',
                                           'data: {data}',
                                           'max mem: {memory:.0f}'])
        else:
            log_msg = self.delimiter.join([header,
                                           '[{0' + space_fmt + '}/{1}]',
                                           'eta: {eta}',
                                           '{meters}',
                                           'time: {time}',
                                           'data: {data}'])

        MB = 1024.0 * 1024.0    # MB换算
        # 遍历可迭代对象
        for obj in iterable:
            data_time.update(time.time() - end)     # 更新数据时间
            yield obj       # 返回对象（用于循环）
            iter_time.update(time.time() - end)     # 更新迭代时间

            # 定期打印日志
            if i % print_freq == 0 or i == len(iterable) - 1:
                # 计算预计剩余时间
                eta_second = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=eta_second))

                # 打印日志
                if torch.cuda.is_available():
                    print(log_msg.format(i, len(iterable),
                                         eta=eta_string,
                                         meters=str(self),
                                         time=str(iter_time),
                                         data=str(data_time),
                                         memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(i, len(iterable),
                                         eta=eta_string,
                                         meters=str(self),
                                         time=str(iter_time),
                                         data=str(data_time)))
            i += 1
            end = time.time()       # 更新结束时间
        # 打印总时间
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(header,
                                                         total_time_str,
                                                         total_time / len(iterable)))


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    """创建学习率预热调度器"""
    def f(x):
        """根据step数返回一个学习率倍率因子"""
        if x >= warmup_iters:  # 当迭代数大于给定的warmup_iters时，倍率因子为1
            return 1
        alpha = float(x) / warmup_iters
        # 迭代过程中倍率因子从warmup_factor -> 1
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def mkdir(path):
    """创建目录（如果不存在）"""
    try:
        os.makedirs(path)       # 创建目录
    except OSError as e:
        if e.errno != errno.EEXIST:     # 忽略目录已存在的错误
            raise


def setup_for_distributed(is_master):
    """
    设置分布式环境下的打印行为
    :param is_master: 是否为主进程
    :return:
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print       # 保存原始打印函数

    def print(*args, **kwargs):
        """自定义打印函数"""
        force = kwargs.pop('force', False)      # 获取force参数
        if is_master or force:          # 仅主进程或强制打印
            builtin_print(*args, **kwargs)

    __builtin__.print = print       # 覆盖内置打印函数


def is_dist_avail_and_initialized():
    """检查分布式环境是否可用并已初始化"""
    if not dist.is_available():         # 检查是否可用
        return False
    if not dist.is_initialized():       # 检查是否初始化
        return False
    return True


def get_world_size():
    """获取进程总数"""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """获取当前进程的rank"""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    """检查当前进程是否为主进程（rank 0）"""
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    """仅在主进程上保存模型"""
    if is_main_process():
        torch.save(*args, **kwargs)         # 保存模型


def init_distributed_mode(args):
    """
    初始化分布式模式
    :param args: 命令行参数
    :return:
    """
    # 检查环境变量
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:      # SLURM集群支持
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True         # 标记为分布式模式

    # 设置当前GPU设备
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'      # 使用NCCL后端

    # 打印初始化信息
    print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)
    # 初始化进程组
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    # 同步所有进程
    torch.distributed.barrier()
    # 设置分布式打印
    setup_for_distributed(args.rank == 0)

