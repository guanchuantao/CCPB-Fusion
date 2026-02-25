
from torch.utils.data import Dataset
import os
from lxml import etree
import json
from PIL import Image
import torch

'''
要求：
要写自己的Dataset，首先需要参考pytorch官方文档
官方文档要求：需要继承torch.utils.data里面的Dataset类；
需要实现两个方法：__getitem__  获取图片和图片的信息
__len__ 获取数据集的长度
额外的实现一个方法get_height_and_width 获取图片宽和高
'''


class MyVOCDataSet(Dataset):
    """ 解析voc2012格式数据集 """

    def __init__(self, voc_root, year="2012", txt_type="train.txt", transforms=None):
        """
        :param voc_root: VOC文件夹的路径（可以包含VOCdevkit，也可以不包含VOCdevkit）
        :param year: 对应VOCdevkit的数据年份
        :param txt_type: 指定训练集的文件名
        :param transforms: 对应读取图像的变换操作
        """

        # 主文件目录
        self.root_dir = voc_root
        # 如果主目录包含VOCdevkit字符，说明指定了对的路径，则直接拼接
        if "VOCdevkit" in voc_root:
            self.root = os.path.join(self.root_dir, f"VOC{year}")
        else:
            # 如果不包含VOCdevkit，那么就直接拼接当前文件项目的相对路径
            self.root = os.path.join(self.root_dir, "VOCdevkit", f"VOC{year}")
        # 拼接主目录和图片目录名称，即为图片所在的文件项目的相对路径
        self.image_root = os.path.join(self.root, "JPEGImages")
        # 拼接主目录和图片标签名称，即为图片对应标签文件夹的相对路径
        self.annotations_root = os.path.join(self.root, "Annotations")
        # 拼接主目录和训练集txt的目录
        txt_root = os.path.join(self.root, "ImageSets", "Main", txt_type)
        # 判断txt文件是否在目录中，如果没有则抛出异常
        assert os.path.exists(txt_root), f"{txt_root}在文件夹中没有，找不到"
        # 通过打开txt文件，然后将每一个xml文件名和标签文件夹对应相对路径连接，这样就获取了每一个xml文件的路径（过程为空则抛出错误）
        with open(txt_root) as read_name:
            xml_list = [each_line.strip() for each_line in read_name.readlines() if len(each_line.strip()) > 0]
        # 存储xml列表
        self.xml_list = []
        # 存储xml内容
        self.xml_list_data = []
        # 存储每个样本是否有目标
        self.has_target_list = []

        # 这里对每一个xml文件进行遍历解析
        for xml_path_each in xml_list:
            xml_path = os.path.join(self.annotations_root, xml_path_each + ".xml")
            jpg_path = os.path.join(self.image_root, xml_path_each + ".jpg")
            # 这里解析，如果找不到对应的文件，则打印信息日志
            if not os.path.exists(jpg_path):
                # 自定义新对象的数据，without_target:0(没有目标)，1（有目标）
                print(f"图片文件 {jpg_path} 不存在，跳过")
                continue
            # 这里首先读取对应的xml的内容
            # 检查XML文件是否存在
            if os.path.exists(xml_path):
                # 这里首先读取对应的xml的内容
                with open(xml_path) as fild_xml:
                    xml_read_str = fild_xml.read()
                # 这里判断xml文件里面是否含有object内容，如果有再继续后面提取内容
                if "object" in xml_read_str:
                    # 这里将xml内容转换为树结构
                    xml_tree = etree.fromstring(xml_read_str)
                    # 通过重新写的方法来获取解析树结构
                    xml_data = self.get_xml_message_to_dict(xml_tree)["annotation"]
                    # 将树结构内容存储在全局变量中
                    self.xml_list_data.append(xml_data)
                    # 存储含有内容的xml文件名
                    self.xml_list.append(xml_path)
                    self.has_target_list.append(True)
                    # 这里根据xml_list长度判断是否含有xml内容
                    assert len(self.xml_list) > 0 and len(
                        self.xml_list_data) > 0, f"文本数据分类集里面{txt_root}没有xml内容"
                else:
                    print(f"图片：{xml_path_each}没有对应的object，需要更改")
            else:
                # 无目标的情况
                self._add_no_target_sample(xml_path_each)
        # 目标类别json文件
        json_class_path = './pascal_voc_classes.json'
        # 这里判断一下json文件是否存在
        assert os.path.exists(json_class_path), f"目标类别文件{json_class_path}找不到"
        # 将json文件只读打开
        with open(json_class_path, 'r') as f:
            # 这里将打开的json按照json格式解析，解析为字典格式
            self.class_dict = json.load(f)
        # 这里将图像处理方法直接赋值给全局变量
        self.transforms = transforms

    def _add_no_target_sample(self, image_name):
        """添加无目标样本"""
        # 自定义新对象的数据，without_target:0(没有目标)
        new_obj = {
            'filename': image_name + ".jpg",
            'without_target': '0',
            'size': {
                'width': '512',
                'height': '512',
                'depth': '3'
            }
        }
        # 将树结构内容存储在全局变量中
        self.xml_list_data.append(new_obj)
        # 存储含有内容的xml文件名（使用虚拟路径）
        self.xml_list.append(os.path.join(self.annotations_root, image_name + ".xml"))
        self.has_target_list.append(False)

    """这里为输入对应的id编号，返回对应的图像，对应的xml文件中的目标边界框"""

    def __getitem__(self, idx):
        # 取出对应的xml路径
        xml_path = self.xml_list[idx]
        # 取出对应的解析xml内容
        xml_data = self.xml_list_data[idx]
        # 取出对应图片的路径
        img_path = os.path.join(self.image_root, xml_data["filename"])
        # 打开图片
        image = Image.open(img_path)
        # 接下来判断当前图片是否为jpeg格式
        # if image.format != "JPEG":
        #     raise ValueError(f"图片{image}格式不是jpg，后面进行不了")
        boxes = []
        labels = []
        is_crowd = []
        without_target = []
        image_id = torch.tensor([idx])
        # 遍历data信息，读取xy的值
        if "object" in xml_data:
            for xml_obj in xml_data["object"]:
                x_min = float(xml_obj["bndbox"]["xmin"])
                y_min = float(xml_obj["bndbox"]["ymin"])
                x_max = float(xml_obj["bndbox"]["xmax"])
                y_max = float(xml_obj["bndbox"]["ymax"])
                # 这里需要再判断一下，是否xy为0
                if x_max <= x_min or y_max <= y_min:
                    print(f"当前图片{xml_path}目标标签值有误，需要更改")
                # 将xy的值添加进box中
                boxes.append([x_min, y_min, x_max, y_max])
                # 根据当前object的类别名称判断出类别编号
                labels.append(self.class_dict[xml_obj["name"]])
                # 这里判断是否有难检测字段，有(为1)，则添加进iscrowd里面
                if "difficult" in xml_obj:
                    is_crowd.append(int(xml_obj["difficult"]))
                else:
                    # 没有则为（0）
                    is_crowd.append(0)
            # 将数据处理完之后，需要将其转换为torch模式的tensor形式，方便后面运算
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            is_crowd = torch.as_tensor(is_crowd, dtype=torch.int64)
            without_target = torch.as_tensor(without_target, dtype=torch.int64)
            # 这里计算面积
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            # 将信息存放到字典里面
            tar_get = {"boxes": boxes,
                       "labels": labels,
                       "image_id": image_id,
                       "is_crowd": is_crowd,
                       "area": area,
                       "without_target": without_target,
                       "name": xml_data["filename"]
                       }
        else:
            without_target.append(0)
            without_target = torch.as_tensor(without_target, dtype=torch.int64)
            # 将信息存放到字典里面
            tar_get = {"without_target": without_target,
                       "image_id": image_id,
                       "name": xml_data["filename"]
                       }
        # 这里判断是否有图像处理操作，如果有，那么就将图像目标进行图像处理
        if self.transforms is not None:
            image, tar_get = self.transforms(image, tar_get)
        # 返回图片信息和目标信息
        return image, tar_get

    """这里直接返回xml列表的长度，代表标签文件有多少个"""

    def __len__(self):
        return len(self.xml_list)

    """这里是将传入进来的xml内容转换为字典（递归方法）"""

    def get_xml_message_to_dict(self, xml):
        # 这里为后续递归做准备，如果递归过来的xml只有一层，没有下一层，那么直接将这一层xml的标题作为标题，内容作为内容返回
        if len(xml) == 0:
            return {xml.tag: xml.text}
        # 这里定义空的集合，用来存放提取xml里面的标题和对应内容
        result_all = {}
        # 这里遍历每一层xml，进行层层提取
        for child_xml in xml:
            # 这里使用递归，如果
            child_xml_result = self.get_xml_message_to_dict(child_xml)
            # 这里通过是否为object来判断是否需要使用列表来存储信息
            if child_xml.tag != "object":
                # 这里存储信息为xml里面不是object的其他结构，以标题为字典，如果多层结构也为列表
                result_all[child_xml.tag] = child_xml_result[child_xml.tag]
            else:
                # 这里为object的操作，需要考虑多个object
                if child_xml.tag not in result_all:
                    # 这里首先根据result_all里面是否含有object，如果没有则定义一个字典
                    result_all[child_xml.tag] = []
                # 这里将递归结果里面的object内容添加到result_all里面的object
                result_all[child_xml.tag].append(child_xml_result[child_xml.tag])
        # 这里返回的内容是传入过来xml的主标题为字典，树结构内容为值
        return {xml.tag: result_all}

    """这里是为了获取图片的宽和高"""

    def get_height_and_width(self, idx):
        # 取出对应的解析xml内容
        xml_data = self.xml_list_data[idx]
        return int(xml_data["size"]["height"]), int(xml_data["size"]["height"])

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))

    def coco_index(self, idx):
        """
        该方法是专门为pycocotools统计标签信息准备，不对图像和标签作任何处理
        由于不用去读取图片，可大幅缩减统计时间

        Args:
            idx: 输入需要获取图像的索引
        """
        # read xml
        xml_path = self.xml_list[idx]
        # 取出对应的解析xml内容
        xml_data = self.xml_list_data[idx]
        image_id = torch.tensor([idx])
        without_target = []
        if "object" in xml_data:
            with open(xml_path) as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str)
            data = self.get_xml_message_to_dict(xml)["annotation"]
            data_height = int(data["size"]["height"])
            data_width = int(data["size"]["width"])
            boxes = []
            labels = []
            iscrowd = []
            for obj in data["object"]:
                without_target.append(1)
                xmin = float(obj["bndbox"]["xmin"])
                xmax = float(obj["bndbox"]["xmax"])
                ymin = float(obj["bndbox"]["ymin"])
                ymax = float(obj["bndbox"]["ymax"])
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(self.class_dict[obj["name"]])
                iscrowd.append(int(obj["difficult"]))
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
            without_target = torch.as_tensor(without_target, dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd
            target["without_target"] = without_target
        else:
            without_target.append(0)
            without_target = torch.as_tensor(without_target, dtype=torch.int64)
            data_height = 512
            data_width = 512
            target = {"without_target": without_target,
                      "image_id": image_id
                      }

        return (data_height, data_width), target
