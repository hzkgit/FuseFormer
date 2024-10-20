# 数据集格式化(将CASIA数据集中所有二级子文件夹放到一个文件夹中，且只需要图像，作为训练集)

import os
import shutil


def move_subfolders_to_target(root_folder, target_folder):
    # 确保目标目录存在
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 遍历一级和二级子文件夹
    i = 0
    for root, dirs, files in os.walk(root_folder):
        # 检查当前目录是否为二级子文件夹（通过目录深度判断）
        if root.count(os.sep) == root_folder.count(os.sep) + 2:
            # 构建目标子文件夹的路径
            parts = root.rsplit('\\', 1)[1]
            if parts.startswith('00') or parts.startswith('45'):
                target_subfolder = os.path.join(target_folder, "JPEGImages", os.path.basename(root+'_'+str(i)))
                # 如果目标子文件夹不存在，则创建
                if not os.path.exists(target_subfolder):
                    os.makedirs(target_subfolder)
                # 复制二级子文件夹下的所有内容到目标子文件夹
                for file in files:
                    shutil.copy2(os.path.join(root, file), os.path.join(target_subfolder, file))
                i = i+1


def move_subfolders_to_target(root_folder, target_folder, path):
    # 确保目标目录存在
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 遍历一级和二级子文件夹
    i = 0
    for root, dirs, files in os.walk(root_folder):
        # 检查当前目录是否为二级子文件夹（通过目录深度判断）
        if root.count(os.sep) == root_folder.count(os.sep) + 2:
            # 构建目标子文件夹的路径
            parts = root.rsplit('\\', 1)[1]
            if parts.startswith('00') or parts.startswith('45'):
                target_subfolder = os.path.join(target_folder, path, os.path.basename(root+'_'+str(i)))
                # 如果目标子文件夹不存在，则创建
                if not os.path.exists(target_subfolder):
                    os.makedirs(target_subfolder)
                # 复制二级子文件夹下的所有内容到目标子文件夹
                for file in files:
                    shutil.copy2(os.path.join(root, file), os.path.join(target_subfolder, file))
                i = i+1


if __name__ == '__main__':
    # 指定根目录和目标目录
    root_folder = r'F:\ML\FuseFormer\data\CASIA\DatasetA\gaitdb'
    target_folder = r'F:\ML\FuseFormer\data\CASIA\train'
    move_subfolders_to_target(root_folder, target_folder, "JPEGImages")

    # 指定根目录和目标目录
    root_folder = r'F:\ML\FuseFormer\data\CASIA\DatasetA\silhouettes'
    target_folder = r'F:\ML\FuseFormer\data\CASIA\train'
    move_subfolders_to_target(root_folder, target_folder, "Mask")
