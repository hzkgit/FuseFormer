import os
import random
import torch
import numpy as np
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
from core.utils import create_random_shape_with_random_motion
from core.utils import Stack, ToTorchFormatTensor, GroupRandomHorizontalFlip


class Dataset(torch.utils.data.Dataset):
    def __init__(self, args: dict, split='train'):
        self.args = args
        self.split = split
        self.sample_length = args['sample_length']
        self.size = self.w, self.h = (args['w'], args['h'])

        if args['name'] == 'YouTubeVOS':
            vid_lst_prefix = os.path.join(args['data_root'], args['name'], split+'/JPEGImages')
            vid_lst = os.listdir(vid_lst_prefix)
            self.video_names = [os.path.join(vid_lst_prefix, name) for name in vid_lst]

        self._to_tensors = transforms.Compose([
            Stack(),  # 在某个维度上堆叠起来
            ToTorchFormatTensor(), ])  # 将数据转换为PyTorch张量（Tensor）

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('Loading error in video {}'.format(self.video_names[index]))
            item = self.load_item(0)
        return item

    def load_item(self, index):
        video_name = self.video_names[index]  # 传入一个随机索引，获取视频帧位置
        all_frames = [os.path.join(video_name, name) for name in sorted(os.listdir(video_name))]  # 获取文件夹内所有视频帧
        all_masks = create_random_shape_with_random_motion(
            len(all_frames), imageHeight=self.h, imageWidth=self.w)  # 创建随机掩码(这个随机掩码是一个运动物体) TODO 这里需要改成：掩码是静止的物体
        ref_index = get_ref_index(len(all_frames), self.sample_length)  # 随机获取sample_length个帧 或 随机获取连续的sample_length个帧
        # read video frames
        frames = []
        masks = []
        for idx in ref_index:
            img = Image.open(all_frames[idx]).convert('RGB')
            img = img.resize(self.size)
            frames.append(img)
            masks.append(all_masks[idx])  # 掩码是0和1组成，1代表不可见区域，0代表可见区域
        if self.split == 'train':
            frames = GroupRandomHorizontalFlip()(frames)  # 水平翻转
        # To tensors
        frame_tensors = self._to_tensors(frames)*2.0 - 1.0  # 将范围从[0,1]=>[0,2]=>[-1,1]
        mask_tensors = self._to_tensors(masks)
        return frame_tensors, mask_tensors


def get_ref_index(length, sample_length):
    if random.uniform(0, 1) > 0.5:
        ref_index = random.sample(range(length), sample_length)  # 随机抽取sample_length个不重复的元素
        ref_index.sort()  # 排序，确保索引按升序排列
    else:
        pivot = random.randint(0, length-sample_length)
        ref_index = [pivot+i for i in range(sample_length)]  # 从原序列中选取了一个连续的子序列[pivot,pivot+sample_length]
    return ref_index
