import os
import random
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
from core.utils import create_random_shape_with_random_motion, separate_masks, paths_to_images, match_no_occlusion
from core.utils import Stack, ToTorchFormatTensor, GroupRandomHorizontalFlip



class Dataset(torch.utils.data.Dataset):
    def __init__(self, args: dict, split='train'):
        self.args = args
        self.split = split
        self.sample_length = args['sample_length']
        self.size = self.w, self.h = (args['w'], args['h'])

        if args['name'] == 'YouTubeVOS':
            vid_lst_prefix = os.path.join(args['data_root'], args['name'], split+'/JPEGImages')  # 获取所有子文件夹所在路径
            vid_lst = os.listdir(vid_lst_prefix)  # 得到所有子文件夹
            self.video_names = [os.path.join(vid_lst_prefix, name) for name in vid_lst]  # 得到子文件夹内所有二级文件夹所在路径
        if args['name'] == 'CASIA':
            # 获取RGB
            vid_lst_prefix = os.path.join(args['data_root'], args['name'], split+'/JPEGImages')
            vid_lst = os.listdir(vid_lst_prefix)
            self.video_names = [os.path.join(vid_lst_prefix, name) for name in vid_lst]
            # 获取人体mask
            mask_vid_lst_prefix = os.path.join(args['data_root'], args['name'], split + '/Mask')
            mask_vid_lst = os.listdir(mask_vid_lst_prefix)
            self.mask_video_names = [os.path.join(mask_vid_lst_prefix, name) for name in mask_vid_lst]
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
        if self.args['name'] == 'YouTubeVOS':
            video_name = self.video_names[index]  # 传入一个随机索引，获取视频帧位置
            all_frames = [os.path.join(video_name, name) for name in sorted(os.listdir(video_name))]  # 获取文件夹内所有视频帧
            all_masks = create_random_shape_with_random_motion(
                len(all_frames), imageHeight=self.h, imageWidth=self.w)  # 创建随机掩码(这个随机掩码是一个运动物体)
            ref_index = get_ref_index(len(all_frames), self.sample_length)  # 随机获取sample_length个帧 或 随机获取连续的sample_length个帧

        if self.args['name'] == 'CASIA':
            video_name = self.video_names[index]  # 传入一个随机索引，获取一个二级子文件夹路径（也即一个视频）
            all_frames = [os.path.join(video_name, name) for name in sorted(os.listdir(video_name))]  # 获取二级文件夹内所有视频帧
            mask_video_name = self.mask_video_names[index]
            mask_all_frames = [os.path.join(mask_video_name, name) for name in sorted(os.listdir(mask_video_name))]

            all_masks = create_random_shape_with_random_motion(
                len(all_frames), imageHeight=self.h, imageWidth=self.w)  # 在帧上创建随机掩码(这个随机掩码是一个运动物体)

            # 根据all_masks和mask_all_frames是否有交集，来将所有帧进行二分类（遮挡帧和非遮挡帧）
            intersection, difference, difference_masks = separate_masks(all_masks, paths_to_images(
                mask_all_frames, imageHeight=self.h, imageWidth=self.w))

            # 获取图像序号
            ref_index = get_ref_index_by_class(difference_masks, self.sample_length, intersection, difference)  # 随机获取sample_length个帧 或 随机获取连续的sample_length个帧  TODO 这里可以改成我的那个匹配算法

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


# 按帧类型采样
def get_ref_index_by_class(difference_masks, sample_length, intersection, difference):
    ref_index = []
    if len(intersection) == 0:
        # 如果不存在遮挡帧,则随机选择sample_length个未遮挡帧
        ref_index = random.sample(difference, sample_length)
        ref_index.sort()
        return ref_index
    if len(difference) == 0:
        # 如果不存在未遮挡帧,则随机选择sample_length个遮挡帧
        ref_index = random.sample(intersection, sample_length)
        ref_index.sort()
        return ref_index
    if random.uniform(0, 1) > 0.5:
        index = random.choice(intersection)  # 从遮挡帧列表中随机抽取1个序号
        ref_index.append(index)
        if len(difference) < sample_length-1:
            # 如果未遮挡帧不够sample_length-1个，则少的部分重复采样
            ref_index.extend(difference)
            for _ in range(sample_length-1-len(difference)):
                random_choice = random.choice(difference)
                ref_index.append(random_choice)
        else:
            occlusion_mask = difference_masks[index]
            no_occlusion_masks = [difference_masks[i] for i in difference]
            indexes = match_no_occlusion(occlusion_mask, no_occlusion_masks, sample_length-1)  # 从非遮挡帧中选择相似度较高的前sample_length-1个
            ref_index.extend(indexes)  # 序列中第一个就是遮挡帧序号 后面是相似的非遮挡帧序号
        return ref_index
    else:
        if len(difference) < sample_length:
            # 如果未遮挡帧不够sample_length个，则随机选择sample_length-len(difference)个遮挡帧
            ref_index.extend(difference)
            temp_index = random.sample(intersection, sample_length-len(difference))
            ref_index.extend(temp_index)
        else:
            ref_index = random.sample(difference, sample_length)
            ref_index.sort()
        return ref_index
