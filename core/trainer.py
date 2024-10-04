import os
import glob
import logging
import importlib
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter
import torch.distributed as dist

from core.dataset import Dataset
from core.loss import AdversarialLoss


class Trainer():
    def __init__(self, config):
        self.config = config
        self.epoch = 0
        self.iteration = 0

        # setup data set and data loader DataLoader通过调用Dataset的__getitem__方法来获取数据
        self.train_dataset = Dataset(config['data_loader'], split='train')
        self.train_sampler = None  # 默认使用SequentialSampler
        self.train_args = config['trainer']
        if config['distributed']:
            self.train_sampler = DistributedSampler(  # 如果是分布式训练则使用DistributedSampler
                self.train_dataset,
                num_replicas=config['world_size'], 
                rank=config['global_rank'])
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_args['batch_size'] // config['world_size'],
            shuffle=(self.train_sampler is None),  # 如果train_sampler为空则随机采样，即使用RandomSampler
            num_workers=self.train_args['num_workers'], 
            sampler=self.train_sampler)

        # set loss functions 
        self.adversarial_loss = AdversarialLoss(type=self.config['losses']['GAN_LOSS'])
        self.adversarial_loss = self.adversarial_loss.to(self.config['device'])
        self.l1_loss = nn.L1Loss()

        # setup models including generator and discriminator
        net = importlib.import_module('model.'+config['model']['net'])
        self.netG = net.InpaintGenerator()
        self.netG = self.netG.to(self.config['device'])
        if not self.config['model']['no_dis']:
            self.netD = net.Discriminator(
                in_channels=3, use_sigmoid=config['losses']['GAN_LOSS'] != 'hinge')
            self.netD = self.netD.to(self.config['device'])
        self.optimG = torch.optim.Adam(
            self.netG.parameters(), 
            lr=config['trainer']['lr'],
            betas=(self.config['trainer']['beta1'], self.config['trainer']['beta2']))
        if not self.config['model']['no_dis']:
            self.optimD = torch.optim.Adam(
                self.netD.parameters(), 
                lr=config['trainer']['lr'],
                betas=(self.config['trainer']['beta1'], self.config['trainer']['beta2']))
        self.load()

        if config['distributed']:
            self.netG = DDP(
                self.netG, 
                device_ids=[self.config['local_rank']], 
                output_device=self.config['local_rank'],
                broadcast_buffers=True, 
                find_unused_parameters=True)
            if not self.config['model']['no_dis']:
                self.netD = DDP(
                    self.netD, 
                    device_ids=[self.config['local_rank']], 
                    output_device=self.config['local_rank'],
                    broadcast_buffers=True, 
                    find_unused_parameters=False)

        # set summary writer
        self.dis_writer = None
        self.gen_writer = None
        self.summary = {}
        if self.config['global_rank'] == 0 or (not config['distributed']):
            self.dis_writer = SummaryWriter(
                os.path.join(config['save_dir'], 'dis'))
            self.gen_writer = SummaryWriter(
                os.path.join(config['save_dir'], 'gen'))

    # get current learning rate
    def get_lr(self):
        return self.optimG.param_groups[0]['lr']

     # learning rate scheduler, step
    def adjust_learning_rate(self):
        decay = 0.1**(min(self.iteration,
                          self.config['trainer']['niter']) // self.config['trainer']['niter'])
        new_lr = self.config['trainer']['lr'] * decay
        if new_lr != self.get_lr():
            for param_group in self.optimG.param_groups:
                param_group['lr'] = new_lr
            if not self.config['model']['no_dis']:
                for param_group in self.optimD.param_groups:
                    param_group['lr'] = new_lr

    # add summary
    def add_summary(self, writer, name, val):
        if name not in self.summary:
            self.summary[name] = 0
        self.summary[name] += val
        if writer is not None and self.iteration % 100 == 0:
            writer.add_scalar(name, self.summary[name]/100, self.iteration)
            self.summary[name] = 0

    # load netG and netD
    def load(self):
        model_path = self.config['save_dir']
        if os.path.isfile(os.path.join(model_path, 'latest.ckpt')):
            latest_epoch = open(os.path.join(
                model_path, 'latest.ckpt'), 'r').read().splitlines()[-1]
        else:
            ckpts = [os.path.basename(i).split('.pth')[0] for i in glob.glob(
                os.path.join(model_path, '*.pth'))]
            ckpts.sort()
            latest_epoch = ckpts[-1] if len(ckpts) > 0 else None
        if latest_epoch is not None:
            gen_path = os.path.join(
                model_path, 'gen_{}.pth'.format(str(latest_epoch).zfill(5)))
            dis_path = os.path.join(
                model_path, 'dis_{}.pth'.format(str(latest_epoch).zfill(5)))
            opt_path = os.path.join(
                model_path, 'opt_{}.pth'.format(str(latest_epoch).zfill(5)))
            if self.config['global_rank'] == 0:
                print('Loading model from {}...'.format(gen_path))
            data = torch.load(gen_path, map_location=self.config['device'])
            self.netG.load_state_dict(data['netG'])
            if not self.config['model']['no_dis']:
                data = torch.load(dis_path, map_location=self.config['device'])
                self.netD.load_state_dict(data['netD'])
            data = torch.load(opt_path, map_location=self.config['device'])
            self.optimG.load_state_dict(data['optimG'])
            if not self.config['model']['no_dis']:
                self.optimD.load_state_dict(data['optimD'])
            self.epoch = data['epoch']
            self.iteration = data['iteration']
        else:
            if self.config['global_rank'] == 0:
                print(
                    'Warnning: There is no trained model found. An initialized model will be used.')

    # save parameters every eval_epoch
    def save(self, it):
        if self.config['global_rank'] == 0:
            gen_path = os.path.join(
                self.config['save_dir'], 'gen_{}.pth'.format(str(it).zfill(5)))
            dis_path = os.path.join(
                self.config['save_dir'], 'dis_{}.pth'.format(str(it).zfill(5)))
            opt_path = os.path.join(
                self.config['save_dir'], 'opt_{}.pth'.format(str(it).zfill(5)))
            print('\nsaving model to {} ...'.format(gen_path))
            if isinstance(self.netG, torch.nn.DataParallel) or isinstance(self.netG, DDP):
                netG = self.netG.module
                if not self.config['model']['no_dis']:
                    netD = self.netD.module
            else:
                netG = self.netG
                if not self.config['model']['no_dis']:
                    netD = self.netD
            torch.save({'netG': netG.state_dict()}, gen_path)
            if not self.config['model']['no_dis']:
                torch.save({'netD': netD.state_dict()}, dis_path)
                torch.save({'epoch': self.epoch,
                            'iteration': self.iteration,
                            'optimG': self.optimG.state_dict(),
                            'optimD': self.optimD.state_dict()}, opt_path)
            else:
                torch.save({'epoch': self.epoch,
                            'iteration': self.iteration,
                            'optimG': self.optimG.state_dict()}, opt_path)
            os.system('echo {} > {}'.format(str(it).zfill(5),
                                            os.path.join(self.config['save_dir'], 'latest.ckpt')))

    # train entry
    def train(self):
        # 迭代（iteration）指的是模型在训练过程中对批量数据进行一次前向传播和一次反向传播的完整过程，包括计算损失函数和更新模型参数
        pbar = range(int(self.train_args['iterations']))
        if self.config['global_rank'] == 0:
            pbar = tqdm(pbar, initial=self.iteration, dynamic_ncols=True, smoothing=0.01)

        os.makedirs('logs', exist_ok=True)
        logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='logs/{}.log'.format(self.config['save_dir'].split('/')[-1]),
                    filemode='w')
        
        while True:
            self.epoch += 1
            if self.config['distributed']:
                self.train_sampler.set_epoch(self.epoch)

            self._train_epoch(pbar)  # 调用一次epoch（epoch指的是整个训练数据集被模型遍历一次的过程）
            if self.iteration > self.train_args['iterations']:
                break
        print('\nEnd training....')

    # process input and calculate loss every training epoch
    def _train_epoch(self, pbar):
        device = self.config['device']

        for frames, masks in self.train_loader:
            self.adjust_learning_rate()
            self.iteration += 1

            frames, masks = frames.to(device), masks.to(device)  # frames(8,2,3,240,432) masks(8,2,1,240,432)
            b, t, c, h, w = frames.size()
            masked_frame = (frames * (1 - masks).float())  # masked_frame：保留原帧中可见区域，不可见区域全部设为0
            pred_img = self.netG(masked_frame)  # 输入数据到模型
            frames = frames.view(b*t, c, h, w)  # (16,3,240,432)
            masks = masks.view(b*t, 1, h, w)
            comp_img = frames*(1.-masks) + masks*pred_img  # frames*(1.-masks)：取原帧中可见部分  masks*pred_img：取预测图中的不可见部分

            gen_loss = 0
            dis_loss = 0

            if not self.config['model']['no_dis']:
                # discriminator adversarial loss  进入判别器网络
                real_vid_feat = self.netD(frames)  # 先将原图通过判别器 real_vid_feat=([1, 16, 128, 4, 7])
                fake_vid_feat = self.netD(comp_img.detach())  # 再将合成后的图通过判别器 fake_vid_feat=([1, 16, 128, 4, 7])
                dis_real_loss = self.adversarial_loss(real_vid_feat, True, True)
                dis_fake_loss = self.adversarial_loss(fake_vid_feat, False, True)
                dis_loss += (dis_real_loss + dis_fake_loss) / 2
                self.add_summary(
                    self.dis_writer, 'loss/dis_vid_fake', dis_fake_loss.item())  # 记录损失值到文件，保存位置：checkpoints下
                self.add_summary(
                    self.dis_writer, 'loss/dis_vid_real', dis_real_loss.item())
                # self.optimD.zero_grad():将优化器中存储的所有参数的梯度缓存重置为零。在深度学习模型中，每次前向传播之后，都需要执行反向传播来计算梯度。
                # 这些梯度随后将用于更新模型的参数（权重和偏置）。如果不显式地清除旧的梯度，新的梯度将会累加上次的计算结果，导致错误的累积效应。
                self.optimD.zero_grad()
                # 计算损失张量dis_loss相对于模型中所有带有requires_grad=True属性的参数的梯度,这些梯度将被存储在参数的.grad属性中，并用于后续的参数更新步骤
                dis_loss.backward()
                # 用于执行优化器的参数更新步骤 在训练模型时，一旦计算出损失的梯度（通过loss.backward()），接下来就需要使用优化器来更新模型的参数，以最小化损失函数
                # 当调用self.optimD.step()时,优化器会根据存储在模型参数梯度（.grad属性）中的信息来更新这些参数
                # 注意：这里的优化器只更新判别器网络 ，因为定义时只传入了netD的参数，self.optimD = torch.optim.Adam(self.netD.parameters(), ...)
                self.optimD.step()

                # generator adversarial loss
                gen_vid_feat = self.netD(comp_img)  # 将合成图输入生成器
                gan_loss = self.adversarial_loss(gen_vid_feat, True, False)
                gan_loss = gan_loss * self.config['losses']['adversarial_weight']
                gen_loss += gan_loss  # 生成器损失值
                self.add_summary(
                    self.gen_writer, 'loss/gan_loss', gan_loss.item())

            # generator l1 loss
            hole_loss = self.l1_loss(pred_img*masks, frames*masks)  # 将“预测出的不可见部分”与“原图中真实部分”计算L1损失
            hole_loss = hole_loss / torch.mean(masks) * self.config['losses']['hole_weight']
            gen_loss += hole_loss  # 预测结果与真实结果间的损失值
            self.add_summary(
                self.gen_writer, 'loss/hole_loss', hole_loss.item())

            valid_loss = self.l1_loss(pred_img*(1-masks), frames*(1-masks))  # 将“预测出的可见部分”与“原图中真实部分”计算L1损失
            valid_loss = valid_loss / torch.mean(1-masks) * self.config['losses']['valid_weight']
            gen_loss += valid_loss  # 预测结果（可见部分）与真实结果间的损失值
            self.add_summary(
                self.gen_writer, 'loss/valid_loss', valid_loss.item())
            
            self.optimG.zero_grad()
            gen_loss.backward()
            self.optimG.step()  # 更新生成器的参数

            # console logs 输出日志
            if self.config['global_rank'] == 0:
                pbar.update(1)
                if not self.config['model']['no_dis']:
                    pbar.set_description((
                        f"d: {dis_loss.item():.3f}; g: {gan_loss.item():.3f};"
                        f"hole: {hole_loss.item():.3f}; valid: {valid_loss.item():.3f}")
                    )
                else:
                    pbar.set_description((
                        f"hole: {hole_loss.item():.3f}; valid: {valid_loss.item():.3f}")
                    )

                if self.iteration % self.train_args['log_freq'] == 0:
                    if not self.config['model']['no_dis']:
                        logging.info('[Iter {}] d: {:.4f}; g: {:.4f}; hole: {:.4f}; valid: {:.4f}'.format(self.iteration, dis_loss.item(), gan_loss.item(), hole_loss.item(), valid_loss.item()))
                    else:
                        logging.info('[Iter {}] hole: {:.4f}; valid: {:.4f}'.format(self.iteration, hole_loss.item(), valid_loss.item()))
            # saving models 保存模型参数
            if self.iteration % self.train_args['save_freq'] == 0:
                self.save(int(self.iteration//self.train_args['save_freq']))
            if self.iteration > self.train_args['iterations']:
                break