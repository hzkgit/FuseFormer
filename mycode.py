import torch
import torch.nn as nn

if __name__ == '__main__':
    # 1:
    # import torch
    # from torchvision.utils import make_grid
    # import matplotlib.pyplot as plt
    # # 假设batch_of_images是一个四维Tensor，形状为 (batch_size, channels, height, width)
    # batch_of_images = torch.randn(4, 3, 64, 64)  # 示例批量Tensor
    # # 创建网格
    # grid = make_grid(batch_of_images, nrow=2)  # nrow指定每行的图像数量
    #
    # # 转换为NumPy数组并显示
    # plt.imshow(grid.numpy().transpose((1, 2, 0)))
    # plt.axis('off')
    # plt.show()

    # # 2: nn.Unfold函数
    # import torch
    # import torch.nn as nn
    #
    # # 创建一个示例输入张量
    # input_tensor = torch.randn(1, 3, 3, 3)  # 批处理大小为1，通道数为3，高度和宽度均为5
    #
    # # 初始化nn.Unfold模块
    # unfold_module = nn.Unfold(kernel_size=(2, 2), stride=1, padding=0)  # 3*4*4
    #
    # # 使用nn.Unfold模块处理输入张量
    # output_tensor = unfold_module(input_tensor)  #
    #
    # # 打印输入和输出张量的形状
    # print("Input Tensor Shape:", input_tensor.shape)
    # print(input_tensor)
    # print("Output Tensor Shape:", output_tensor.shape)
    # print(output_tensor)
    # print(output_tensor.permute(0, 2, 1))

    # x = torch.Tensor([[[[1, 2, 3, 4],
    #                     [5, 6, 7, 8],
    #                     [9, 10, 11, 12],
    #                     [13, 14, 15, 16]]]])
    # print(x)
    # unfold_module = nn.Unfold(kernel_size=(2, 2), padding=0, stride=2)
    # x = unfold_module(x)
    # print(x)
    # print(x.size())

    # # 3: nn.Fold函数
    # import torch
    # import torch.nn as nn
    #
    # # 假设的输入参数
    # output_size = (5, 5)  # 我们想要重建的输出图像尺寸
    # kernel_size = 3  # 卷积核的尺寸
    # stride = 1  # 移动步长
    # padding = 1  # 输入边界的填充
    # dilation = 1  # 卷积核内的空洞大小
    #
    # # 初始化Fold模块
    # fold = nn.Fold(output_size=output_size, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    #
    # # 生成模拟输入张量
    # # 假设我们有1个样本，9个通道（因为3x3的卷积核），每通道25个数值（这里是简化版本）
    # input_data = torch.rand(1, 9, 25)
    #
    # # 执行Fold操作
    # print(input_data)
    # reconstructed_image = fold(input_data)
    #
    # # 输出结果检查
    # print(reconstructed_image)
    # print(f'Reconstructed Image Size: {reconstructed_image.size()}')  # (1,1,5,5)

    x = torch.Tensor([[[[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16]]]])
    unfold = nn.Unfold(kernel_size=(2, 2), padding=0, stride=2)
    fold = nn.Fold(output_size=(3, 3), kernel_size=(2, 2), padding=0, stride=1)
    x = unfold(x)
    print(x)
    x = fold(x)
    print(x)
    print(x.size())
