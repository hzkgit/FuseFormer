import os
import cv2
import random
import numpy as np
from PIL import Image, ImageOps

import torch
import matplotlib
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib import pyplot as plt
import cv2
import torchvision
from sklearn.metrics import jaccard_score  # 用于计算IoU
matplotlib.use('agg')


# ###########################################################################
# ###########################################################################


class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, img_group, is_flow=False):
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            if self.is_flow:
                for i in range(0, len(ret), 2):
                    # invert flow pixel values when flipping
                    ret[i] = ImageOps.invert(ret[i])
            return ret
        else:
            return img_group


class Stack(object):
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        mode = img_group[0].mode
        if mode == '1':
            img_group = [img.convert('L') for img in img_group]
            mode = 'L'
        if mode == 'L':
            return np.stack([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif mode == 'RGB':
            if self.roll:
                return np.stack([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                return np.stack(img_group, axis=2)
        else:
            raise NotImplementedError(f"Image mode {mode}")


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """

    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # numpy img: [L, C, H, W]
            img = torch.from_numpy(pic).permute(2, 3, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        img = img.float().div(255) if self.div else img.float()
        return img


# ##########################################
# ##########################################

def create_random_shape_with_random_motion(video_length, imageHeight=240, imageWidth=432):
    # get a random shape
    height = random.randint(imageHeight//3, imageHeight-1)  # height的范围被限制在原图像高度的1/3到图像高度减1之间
    width = random.randint(imageWidth//3, imageWidth-1)  # 同上
    edge_num = random.randint(6, 8)  # 随机形状的边数，取值范围在6到8之间
    ratio = random.randint(6, 8)/10  # 随机的比率ratio，用于控制多边形顶点的随机性或形状的复杂度
    region = get_random_shape(  # 根据给定的参数生成一个随机形状区域
        edge_num=edge_num, ratio=ratio, height=height, width=width)
    region_width, region_height = region.size
    # get random position
    x, y = random.randint(
        0, imageHeight-region_height), random.randint(0, imageWidth-region_width)
    velocity = get_random_velocity(max_speed=3)  # 返回一个随机速度
    m = Image.fromarray(np.zeros((imageHeight, imageWidth)).astype(np.uint8))  # 创建了一个与原始图像大小相同的全零数组
    m.paste(region, (y, x, y+region.size[0], x+region.size[1]))  # 将之前生成的形状区域region粘贴到掩码图像m上。粘贴的位置由(y, x)坐标确定
    masks = [m.convert('L')]  # 将掩码图像m转换为灰度图像（单通道），并存储到列表masks中
    # return fixed masks
    if random.uniform(0, 1) > 0.5:  # 决定是否对视频的每一帧应用相同的掩码
        return masks*video_length
    # return moving masks
    for _ in range(video_length-1):
        x, y, velocity = random_move_control_points(  # 更新形状的位置和速度
            x, y, imageHeight, imageWidth, velocity, region.size, maxLineAcceleration=(3, 0.5), maxInitSpeed=3)
        m = Image.fromarray(
            np.zeros((imageHeight, imageWidth)).astype(np.uint8))  # 创建了一个与原始图像大小相同的全零数组
        m.paste(region, (y, x, y+region.size[0], x+region.size[1]))  # 将生成的形状区域region粘贴到掩码图像m新的位置上
        masks.append(m.convert('L'))
    return masks  # 包含了形状从初始位置随机移动到结束的整个过程的遮罩图像序列


def get_random_shape(edge_num=9, ratio=0.7, width=432, height=240):
    '''
      There is the initial point and 3 points per cubic bezier curve. 
      Thus, the curve will only pass though n points, which will be the sharp edges.
      The other 2 modify the shape of the bezier curve.
      edge_num, Number of possibly sharp edges
      points_num, number of points in the Path
      ratio, (0, 1) magnitude of the perturbation from the unit circle, 
    '''
    points_num = edge_num*3 + 1
    angles = np.linspace(0, 2*np.pi, points_num)
    codes = np.full(points_num, Path.CURVE4)
    codes[0] = Path.MOVETO
    # Using this instad of Path.CLOSEPOLY avoids an innecessary straight line
    verts = np.stack((np.cos(angles), np.sin(angles))).T * \
        (2*ratio*np.random.random(points_num)+1-ratio)[:, None]
    verts[-1, :] = verts[0, :]
    path = Path(verts, codes)
    # draw paths into images
    fig = plt.figure()
    ax = fig.add_subplot(111)
    patch = patches.PathPatch(path, facecolor='black', lw=2)
    ax.add_patch(patch)
    ax.set_xlim(np.min(verts)*1.1, np.max(verts)*1.1)
    ax.set_ylim(np.min(verts)*1.1, np.max(verts)*1.1)
    ax.axis('off')  # removes the axis to leave only the shape
    fig.canvas.draw()
    # convert plt images into numpy images
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape((fig.canvas.get_width_height()[::-1] + (3,)))
    plt.close(fig)
    # postprocess
    data = cv2.resize(data, (width, height))[:, :, 0]
    data = (1 - np.array(data > 0).astype(np.uint8))*255
    corrdinates = np.where(data > 0)
    xmin, xmax, ymin, ymax = np.min(corrdinates[0]), np.max(
        corrdinates[0]), np.min(corrdinates[1]), np.max(corrdinates[1])
    region = Image.fromarray(data).crop((ymin, xmin, ymax, xmax))
    return region


def random_accelerate(velocity, maxAcceleration, dist='uniform'):
    speed, angle = velocity
    d_speed, d_angle = maxAcceleration
    if dist == 'uniform':
        speed += np.random.uniform(-d_speed, d_speed)
        angle += np.random.uniform(-d_angle, d_angle)
    elif dist == 'guassian':
        speed += np.random.normal(0, d_speed / 2)
        angle += np.random.normal(0, d_angle / 2)
    else:
        raise NotImplementedError(
            f'Distribution type {dist} is not supported.')
    return (speed, angle)


def get_random_velocity(max_speed=3, dist='uniform'):
    if dist == 'uniform':
        speed = np.random.uniform(max_speed)
    elif dist == 'guassian':
        speed = np.abs(np.random.normal(0, max_speed / 2))
    else:
        raise NotImplementedError(
            f'Distribution type {dist} is not supported.')
    angle = np.random.uniform(0, 2 * np.pi)
    return (speed, angle)


def random_move_control_points(X, Y, imageHeight, imageWidth, lineVelocity, region_size, maxLineAcceleration=(3, 0.5), maxInitSpeed=3):
    region_width, region_height = region_size
    speed, angle = lineVelocity
    X += int(speed * np.cos(angle))
    Y += int(speed * np.sin(angle))
    lineVelocity = random_accelerate(
        lineVelocity, maxLineAcceleration, dist='guassian')
    if ((X > imageHeight - region_height) or (X < 0) or (Y > imageWidth - region_width) or (Y < 0)):
        lineVelocity = get_random_velocity(maxInitSpeed, dist='guassian')
    new_X = np.clip(X, 0, imageHeight - region_height)
    new_Y = np.clip(Y, 0, imageWidth - region_width)
    return new_X, new_Y, lineVelocity


# 求掩码图A和B的交集和差集
def paths_to_images(mask_paths, imageHeight=240, imageWidth=432):
    """将掩码图像路径列表转换为PIL.Image对象列表"""
    images = []
    for path in mask_paths:
        img = Image.open(path)
        # 如果掩码图像是灰度或二值图像，可以确保其模式为L（8-bit pixels, black and white）
        if img.mode != 'L':
            img = img.convert('L')
        img = img.resize((imageWidth, imageHeight), Image.LANCZOS)
        images.append(img)
    return images


def convert_image_to_mask(image):
    """将PIL Image对象转换为黑白掩码"""
    # 将图像转换为灰度
    grayscale_image = image.convert('L')
    # 使用阈值处理，将图像转换为二进制掩码
    threshold = 128
    binary_mask = np.array(grayscale_image) > threshold
    return binary_mask


def separate_masks(images_A, images_B):
    # images_A：随机生成遮挡物掩码
    # images_B：人体掩码
    C_indices = []  # 存放与B有交集的A的掩码图像的索引
    D_indices = []  # 存放与B无交集的A的掩码图像的索引
    difference_masks = []  # # 合成后的掩码序列
    B_length = len(images_B)

    # 确保不会越界
    min_length = min(len(images_A), B_length)
    for i in range(min_length):
        bin_maskA = convert_image_to_mask(images_A[i])
        bin_maskB = convert_image_to_mask(images_B[i])

        # 计算A+B（交集）
        intersection = np.logical_and(bin_maskA, bin_maskB)
        # 计算B-A（差集）
        difference = np.logical_and(bin_maskB, np.logical_not(bin_maskA))

        difference = np.where(difference, 1, 0)  # 将True和False改为1和0

        difference_masks.append(difference)

        if np.any(intersection):
            C_indices.append(i)
        else:
            D_indices.append(i)

    # 处理images_A多出来的部分
    for i in range(B_length, len(images_A)):
        D_indices.append(i)

    return C_indices, D_indices, difference_masks


#  read frames from video
def read_frame_from_videos(frame_root):
    if frame_root.endswith(('mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI')): # input video path
        video_name = os.path.basename(frame_root)[:-4]
        vframes, aframes, info = torchvision.io.read_video(filename=frame_root, pts_unit='sec')  # RGB
        frames = list(vframes.numpy())
        frames = [Image.fromarray(f) for f in frames]
        fps = info['video_fps']
    else:
        video_name = os.path.basename(frame_root)
        frames = []
        fr_lst = sorted(os.listdir(frame_root))
        for fr in fr_lst:
            frame = cv2.imread(os.path.join(frame_root, fr))
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(frame)
        fps = None
    size = frames[0].size

    return frames, fps, size, video_name


def computeIOU(mask1, mask2):
    # 计算交并集
    mask1_bool = mask1 > 0
    mask2_bool = mask2 > 0
    intersection = np.logical_and(mask1_bool, mask2_bool).astype(int)
    union = np.logical_or(mask1_bool, mask2_bool).astype(int)
    iou = np.sum(intersection) / np.sum(union)
    return iou


def bbox_iou(b1, b2):
    '''
    b: (x1,y1,x2,y2)
    '''
    lx = max(b1[0], b2[0])
    rx = min(b1[2], b2[2])
    uy = max(b1[1], b2[1])
    dy = min(b1[3], b2[3])
    if rx <= lx or dy <= uy:
        return 0.
    else:
        interArea = (rx-lx)*(dy-uy)
        a1 = float((b1[2] - b1[0]) * (b1[3] - b1[1]))
        a2 = float((b2[2] - b2[0]) * (b2[3] - b2[1]))
        return interArea / (a1 + a2 - interArea)


def crop_padding(img, roi, pad_value):
    '''
    img: HxW or HxWxC np.ndarray
    roi: (x,y,w,h)
    pad_value: [b,g,r]
    '''
    need_squeeze = False
    if len(img.shape) == 2:
        img = img[:,:,np.newaxis]
        need_squeeze = True
    assert len(pad_value) == img.shape[2]
    x,y,w,h = roi
    x,y,w,h = int(x),int(y),int(w),int(h)
    H, W = img.shape[:2]
    output = np.tile(np.array(pad_value), (h, w, 1)).astype(img.dtype)
    if bbox_iou((x,y,x+w,y+h), (0,0,W,H)) > 0:
        output[max(-y,0):min(H-y,h), max(-x,0):min(W-x,w), :] = img[max(y,0):min(y+h,H), max(x,0):min(x+w,W), :]
    if need_squeeze:
        output = np.squeeze(output)
    return output


def mask_to_bbox(mask):
    mask = (mask == 1)
    # ~表示取反,True=>False,False=>True
    if np.all(~mask):
        return [0, 0, 0, 0]
    assert len(mask.shape) == 2
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]  # 返回了行和列中True值的索引，然后取第一个和最后一个索引作为边界框的起始和结束坐标
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [cmin.item(), rmin.item(), cmax.item() + 1 - cmin.item(), rmax.item() + 1 - rmin.item()] # xywh


def crop_img(modal):
    modal = np.array(modal)
    bbox = mask_to_bbox(modal)
    # 图像截取
    centerx = bbox[0] + bbox[2] / 2.  # 中心点x坐标
    centery = bbox[1] + bbox[3] / 2.  # 中心点y坐标
    size = max([np.sqrt(bbox[2] * bbox[3] * 3.), bbox[2] * 1.1, bbox[3] * 1.1])
    new_bbox = [int(centerx - size / 2.), int(centery - size / 2.), int(size), int(size)]
    modal = cv2.resize(crop_padding(modal, new_bbox, pad_value=(0,)), (320, 320),
                       interpolation=cv2.INTER_NEAREST)  # 按新的box截取出modal
    return modal


def align2(modal_test, prior_test):
    # 计算四个端点对齐时的iou
    modal_temp = modal_test.copy().astype(np.float32)
    modal_temp[modal_temp == 255] = 1
    modal_bbox = mask_to_bbox(modal_temp)
    x, y, w, h = [modal_bbox[0], modal_bbox[1], modal_bbox[2], modal_bbox[3]]

    prior_temp = prior_test.copy().astype(np.float32)
    prior_temp[prior_temp == 255] = 1
    prior_bbox = mask_to_bbox(prior_temp)
    x1, y1, w1, h1 = [prior_bbox[0], prior_bbox[1], prior_bbox[2], prior_bbox[3]]

    modal_temp[modal_temp == 1] = 255
    prior_temp[prior_temp == 1] = 255

    # 先平移提到(0，0)点
    prior_temp = cv2.warpAffine(prior_temp, np.float32([[1, 0, -x1], [0, 1, -y1]]),
                                prior_test.T.shape)  # 这一步出问题了，改成.T就好了
    iou_arr = np.zeros((2, 2))
    # 以左上角为准对齐
    left_top = cv2.warpAffine(prior_temp, np.float32([[1, 0, x], [0, 1, y]]), prior_test.T.shape)
    iou_arr[0][0] = computeIOU(modal_test, left_top)
    # 以左下角为准对齐
    left_down = cv2.warpAffine(prior_temp, np.float32([[1, 0, x], [0, 1, y + h - h1]]), prior_test.T.shape)
    iou_arr[1][0] = computeIOU(modal_test, left_down)
    # 以右上角为准对齐
    right_top = cv2.warpAffine(prior_temp, np.float32([[1, 0, x + w - w1], [0, 1, y]]), prior_test.T.shape)
    iou_arr[0][1] = computeIOU(modal_test, right_top)
    # 以右下角为准对齐
    right_down = cv2.warpAffine(prior_temp, np.float32([[1, 0, x + w - w1], [0, 1, y + h - h1]]), prior_test.T.shape)
    iou_arr[1][1] = computeIOU(modal_test, right_down)

    # print(iou_arr)

    max_index = np.argmax(iou_arr)
    row = max_index // iou_arr.shape[1]
    col = max_index % iou_arr.shape[1]

    # print(f"最大值是{iou_arr[row, col]}，位于第{row + 1}行，第{col + 1}列。")

    if iou_arr[row, col] == iou_arr[0][0]:
        prior = left_top
    if iou_arr[row, col] == iou_arr[1][0]:
        prior = left_down
    if iou_arr[row, col] == iou_arr[0][1]:
        prior = right_top
    if iou_arr[row, col] == iou_arr[1][1]:
        prior = right_down

    return iou_arr[row, col], prior


def match_no_occlusion(occlusion_mask, no_occlusion_masks, num):
    # 基于jaccard计算相似度并排序（从小到大）
    mask1_crop = crop_img(occlusion_mask)
    similarity = []
    for no_occlusion_mask in no_occlusion_masks:
        mask_crop = crop_img(no_occlusion_mask)
        score = jaccard_score(mask1_crop.flatten(), mask_crop.flatten(), average='weighted')
        similarity.append(score)
    sorted_indices = np.argsort(similarity)
    indexes = sorted_indices[::-1][:10]

    # 计算这10个掩码与目标掩码的iou值
    values = []
    for j in indexes:
        no_occlusion_mask = no_occlusion_masks[j]
        prior = crop_img(no_occlusion_mask)
        max_value, prior = align2(mask1_crop, prior)  # 计算iou值
        values.append(max_value)
    sorted_values_indices = np.argsort(values)
    index_arr = sorted_values_indices[::-1][:num]
    return index_arr



# ##############################################
# ##############################################

if __name__ == '__main__':

    trials = 10
    for _ in range(trials):
        video_length = 10
        # The returned masks are either stationary (50%) or moving (50%)
        masks = create_random_shape_with_random_motion(
            video_length, imageHeight=240, imageWidth=432)

        for m in masks:
            cv2.imshow('mask', np.array(m))
            cv2.waitKey(500)

