import torch
import torch.nn as nn

class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()
        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()  # 二元交叉熵损失
        elif type == 'lsgan':
            self.criterion = nn.MSELoss()
        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:  # 是判别器
                if is_real:  # 是真实的数据
                    outputs = -outputs  # outputs(1,16,128,4,7)，判别器网络输出的特征图，判别器根据特征图来计算损失
                return self.criterion(1 + outputs).mean()  # 加1的操作可以使原本可能负的小数值变为正值(这样就能使用ReLu函数了)
            else:  # 是生成器
                return (-outputs).mean()
        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(
                outputs)
            loss = self.criterion(outputs, labels)
            return loss


