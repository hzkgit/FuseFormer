# -*- coding: utf-8 -*-
import cv2
from PIL import Image
import numpy as np
import importlib
import os
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from core.utils import create_random_shape_with_random_motion, separate_masks, paths_to_images, match_no_occlusion
import random

from core.utils import Stack, ToTorchFormatTensor

parser = argparse.ArgumentParser(description="FuseFormer")
parser.add_argument("-v", "--video", type=str, required=True)
parser.add_argument("-hm", "--human_mask", type=str, required=False)
parser.add_argument("-m", "--mask",   type=str, required=True)
parser.add_argument("-c", "--ckpt",   type=str, required=True)
parser.add_argument("--model", type=str, default='fuseformer')
parser.add_argument("--width", type=int, default=432)
parser.add_argument("--height", type=int, default=240)
parser.add_argument("--outw", type=int, default=432)
parser.add_argument("--outh", type=int, default=240)
parser.add_argument("--step", type=int, default=10)
parser.add_argument("--num_ref", type=int, default=-1)
parser.add_argument("--neighbor_stride", type=int, default=5)
parser.add_argument("--savefps", type=int, default=24)
parser.add_argument("--use_mp4", action='store_true')
args = parser.parse_args()


w, h = args.width, args.height
ref_length = args.step  # ref_step
num_ref = args.num_ref
neighbor_stride = args.neighbor_stride
default_fps = args.savefps

_to_tensors = transforms.Compose([
    Stack(),
    ToTorchFormatTensor()])


# sample reference frames from the whole video 
def get_ref_index(f, neighbor_ids, length):
    ref_index = []
    if num_ref == -1:
        for i in range(0, length, ref_length):  # 每隔10次取一次
            if not i in neighbor_ids:  # 不取临近帧  # 临近帧：[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
                ref_index.append(i)  # [0, 30, 40, 50, 60, 70, 80, 90]
    else:
        start_idx = max(0, f - ref_length * (num_ref//2))
        end_idx = min(length, f + ref_length * (num_ref//2))
        for i in range(start_idx, end_idx+1, ref_length):
            if not i in neighbor_ids:
                if len(ref_index) > num_ref:
                #if len(ref_index) >= 5-len(neighbor_ids):
                    break
                ref_index.append(i)
    return ref_index


# read frame-wise masks 
def read_mask(mpath):
    masks = []
    mnames = os.listdir(mpath)
    mnames.sort()
    for m in mnames: 
        m = Image.open(os.path.join(mpath, m))
        m = m.resize((w, h), Image.NEAREST)
        m = np.array(m.convert('L'))
        m = np.array(m > 0).astype(np.uint8)
        # cv2.dilate:m 中的白色区域（假设为前景）将沿着十字方向进行膨胀，每个白色像素点周围3x3区域内的其他像素点（如果之前是黑色）将变为白色，迭代4次
        m = cv2.dilate(m, cv2.getStructuringElement(
            cv2.MORPH_CROSS, (3, 3)), iterations=4)
        masks.append(Image.fromarray(m*255))
    return masks


#  read frames from video 
def read_frame_from_videos(args):
    vname = args.video
    frames = []
    if args.use_mp4:
        vidcap = cv2.VideoCapture(vname)
        success, image = vidcap.read()
        count = 0
        while success:
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            frames.append(image.resize((w,h)))
            success, image = vidcap.read()
            count += 1
    else:
        lst = os.listdir(vname)
        lst.sort()
        fr_lst = [vname+'/'+name for name in lst]
        for fr in fr_lst:
            image = cv2.imread(fr)
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            frames.append(image.resize((w,h)))
    return frames


# 获取Image对象的掩码
def get_mask(hm_path):
    masks = []
    mnames = os.listdir(hm_path)
    mnames.sort()
    for m in mnames:
        m = Image.open(os.path.join(hm_path, m))
        m = m.resize((w, h), Image.NEAREST)
        masks.append(m.convert('L'))
    return masks


# 按帧类型采样
def get_ref_index_by_class(index, difference_masks, sample_length, intersection, difference):
    ref_index = []
    ref_index.append(index)
    if len(intersection) == 0:
        # 如果不存在遮挡帧,则随机选择sample_length个未遮挡帧
        ref_index = random.sample(difference, sample_length-1)
        return ref_index
    if len(difference) == 0:
        # 如果不存在未遮挡帧,则随机选择sample_length个遮挡帧
        ref_index = random.sample(intersection, sample_length-1)
        return ref_index
    if len(difference) < sample_length-1:
        # 如果未遮挡帧不够sample_length-1个，则少的部分重复采样
        ref_index.extend(difference)
        for _ in range(sample_length-1-len(difference)):
            random_choice = random.choice(difference)
            ref_index.append(random_choice)
        return ref_index
    else:
        occlusion_mask = difference_masks[index]
        no_occlusion_masks = [difference_masks[i] for i in difference]
        indexes = match_no_occlusion(occlusion_mask, no_occlusion_masks, sample_length-1)  # 从非遮挡帧中选择相似度较高的前sample_length-1个
        ref_index.extend(indexes)  # 序列中第一个就是遮挡帧序号 后面是相似的非遮挡帧序号
        return ref_index


def main_worker():
    # set up models 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = importlib.import_module('model.' + args.model)
    model = net.InpaintGenerator().to(device)
    model_path = args.ckpt
    data = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(data['netG'])
    print('loading from: {}'.format(args.ckpt))
    model.eval()

    # prepare datset, encode all frames into deep space 
    frames = read_frame_from_videos(args)  # 获取图片对象List
    video_length = len(frames)
    imgs = _to_tensors(frames).unsqueeze(0)*2-1  # # (93,3,240,432)=>(1,93,3,240,432)
    frames = [np.array(f).astype(np.uint8) for f in frames]  # 将Image对象全部转为ndarray

    masks = read_mask(args.mask)  # 获取掩码对象List
    binary_masks = [np.expand_dims((np.array(m) != 0).astype(np.uint8), 2) for m in masks]  # 将Image对象全部转为ndarray
    masks = _to_tensors(masks).unsqueeze(0)  # ndarray转Tensor
    imgs, masks = imgs.to(device), masks.to(device)
    comp_frames = [None]*video_length  # 全为None的list，用于保存最终输出所有预测帧
    print('loading videos and masks from: {}'.format(args.video))

    # completing holes by spatial-temporal transformers(每次读取10个连续帧加参考帧输入模型)
    for f in range(0, video_length, neighbor_stride):  # 从（0，93）每隔5帧取一次f，
        neighbor_ids = [i for i in range(max(0, f-neighbor_stride), min(video_length, f+neighbor_stride+1))]  # f=15时，neighbor_ids=[10到20]
        ref_ids = get_ref_index(f, neighbor_ids, video_length)  # [0, 30, 40, 50, 60, 70, 80, 90]
        print(f, len(neighbor_ids), len(ref_ids))
        # len_temp = len(neighbor_ids) + len(ref_ids)
        selected_imgs = imgs[:1, neighbor_ids+ref_ids, :, :, :]  # 取出临近帧和参考帧图像
        selected_masks = masks[:1, neighbor_ids+ref_ids, :, :, :]  # # 取出临近帧和参考帧掩码
        with torch.no_grad():
            masked_imgs = selected_imgs*(1-selected_masks)  # 构造遮挡图
            pred_img = model(masked_imgs)  # 输入模型
            pred_img = (pred_img + 1) / 2  # 将范围从 [-1, 1] 转换到 [0, 1]
            pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy()*255  # 将范围从 [0, 1] 转换到 [0, 255] 并转为(19,240,432,3)
            for i in range(len(neighbor_ids)):  # 0到11
                idx = neighbor_ids[i]
                img = np.array(pred_img[i]).astype(np.uint8)*binary_masks[idx] + frames[idx] * (1-binary_masks[idx])  # 预测出的被遮挡部分+原有未遮挡部分
                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else:
                    comp_frames[idx] = comp_frames[idx].astype(np.float32)*0.5 + img.astype(np.float32)*0.5  # 现有图像与新的融合图像融合
    name = args.video.strip().split('/')[-1]
    writer = cv2.VideoWriter(f"{name}_result.mp4", cv2.VideoWriter_fourcc(*"mp4v"), default_fps, (args.outw, args.outh))
    for f in range(video_length):
        comp = np.array(comp_frames[f]).astype(np.uint8)*binary_masks[f] + frames[f] * (1-binary_masks[f])  # 预测出的被遮挡部分+原有未遮挡部分
        if w != args.outw:
            comp = cv2.resize(comp, (args.outw, args.outh), interpolation=cv2.INTER_LINEAR)
        writer.write(cv2.cvtColor(np.array(comp).astype(np.uint8), cv2.COLOR_BGR2RGB))  # 生成视频
    writer.release()
    print('Finish in {}'.format(f"{name}_result.mp4"))


def my_main_worker():
    # set up models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = importlib.import_module('model.' + args.model)
    model = net.InpaintGenerator().to(device)
    data = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(data['netG'])
    print('loading from: {}'.format(args.ckpt))
    model.eval()

    # prepare datset, encode all frames into deep space
    frames = read_frame_from_videos(args)  # 获取图片对象List
    human_masks = get_mask(args.human_mask)  # 获取Image对象的人体掩码List
    masks_temp = get_mask(args.mask)  # 获取Image对象的掩码对象List

    # 根据all_masks和mask_all_frames是否有交集，来将所有帧进行二分类（遮挡帧和非遮挡帧）
    intersection, difference, difference_masks = separate_masks(masks_temp, human_masks)

    video_length = len(frames)
    imgs = _to_tensors(frames).unsqueeze(0) * 2 - 1
    masks = read_mask(args.mask)  # 获取掩码对象List
    binary_masks = [np.expand_dims((np.array(m) != 0).astype(np.uint8), 2) for m in masks]  # 将Image对象全部转为ndarray
    masks = _to_tensors(masks).unsqueeze(0)  # ndarray转Tensor
    imgs, masks = imgs.to(device), masks.to(device)
    comp_frames = [None] * video_length  # 全为None的list，用于保存最终输出所有预测帧

    video_length = len(frames)
    frames = [np.array(f).astype(np.uint8) for f in frames]  # 将Image对象全部转为ndarray
    name = args.video.strip().split('/')[-1]
    writer = cv2.VideoWriter(f"{name}_result.mp4", cv2.VideoWriter_fourcc(*"mp4v"), default_fps, (args.outw, args.outh))
    for f in range(0, video_length):  # 逐帧开始解析
        # 获取5帧
        ref_index = get_ref_index_by_class(f, difference_masks, neighbor_stride, intersection, difference)

        selected_imgs = imgs[:1, ref_index, :, :, :]  # 取出临近帧和参考帧图像
        selected_masks = masks[:1, ref_index, :, :, :]  # # 取出临近帧和参考帧掩码
        with torch.no_grad():
            masked_imgs = selected_imgs * (1 - selected_masks)  # 构造遮挡图
            pred_img = model(masked_imgs)  # 输入模型
            pred_img = (pred_img + 1) / 2  # 将范围从 [-1, 1] 转换到 [0, 1]
            pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255  # 将范围从 [0, 1] 转换到 [0, 255] 并转为(19,240,432,3)
            comp_frames[f] = frames[f] * (1 - binary_masks[f]) + binary_masks[f] * np.array(pred_img)  # 未遮挡部分仍用原来的，遮挡部分用预测的
            print(f)
        comp = cv2.resize(comp_frames[f][0], (args.outw, args.outh), interpolation=cv2.INTER_LINEAR)
        writer.write(cv2.cvtColor(np.array(comp).astype(np.uint8), cv2.COLOR_BGR2RGB))  # 生成视频
    writer.release()
    print('Finish in {}'.format(f"{name}_result.mp4"))


if __name__ == '__main__':
    main_worker()
    # my_main_worker()
