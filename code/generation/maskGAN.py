import torch.nn as nn
import torch
import numpy as np
import models
import torch.nn.functional as F
import torchvision
import os
import logging
import math
from torch.utils.data import DataLoader
from torch.autograd import Variable


generator_models_path = './model/stage_1_model/generator_models/'
discriminator_models_path = './model/stage_1_model/discriminator_models/'


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data)
    elif classname.find('LayerNorm') != -1:
        nn.init.kaiming_normal_(m.weight.data)

def compute_gradient_penalty(D,real_samples, fake_samples,device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1,1,1))).to(device).expand_as(real_samples)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones_like(d_interpolates)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad_l2norm = gradients.norm(2, dim=[1,2,3])
    gradient_penalty = torch.mean((grad_l2norm - 1) ** 2)
    return gradient_penalty

def L2distance(x, y):
	return torch.sqrt(torch.sum((x - y)**2,dim=1))

def norm_distance(fea1,fea2):
    nfea1 = fea1 / torch.linalg.norm(fea1, dim=1).view(fea1.shape[0], 1)
    nfea2 = fea2 / torch.linalg.norm(fea2, dim=1).view(fea2.shape[0], 1)
    num = fea1.shape[0]
    dis = torch.sum(L2distance(nfea1,nfea2),dim=0)/num
    return dis

def min_distance(fea1,fea2,input_num,train_num):
    nfea1 = fea1 / torch.linalg.norm(fea1, dim=1).view(fea1.shape[0], 1)
    nfea2 = fea2 / torch.linalg.norm(fea2, dim=1).view(fea2.shape[0], 1)
    dis = L2distance(nfea1,nfea2)
    num = fea1.shape[0]
    row = num//(input_num+train_num)
    dis_min = torch.ones([row,1])
    n = dis_min.shape[0]
    for i in range(n):
        dis_batch = dis[i*(input_num+train_num):(i+1)*(input_num+train_num)]
        dis_min[i,:] = torch.min(dis_batch)
    return dis_min

def dis(fea1,fea2,):
    nfea1 = fea1 / torch.linalg.norm(fea1, dim=1).view(fea1.shape[0], 1)
    nfea2 = fea2 / torch.linalg.norm(fea2, dim=1).view(fea2.shape[0], 1)
    dis = L2distance(nfea1,nfea2)
    return dis

def projcet_function(perturbation,C_norm):
    pert_norm = torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1)
    num = perturbation.shape[0]
    pert = torch.zeros_like(perturbation)
    for k in range(num):
        if pert_norm[k] > C_norm:
            C = pert_norm[k] / C_norm
            pert[k] = perturbation[k] / C
        else:
            pert[k] = perturbation[k]
    return pert

def margin_loss(dis_min,margin):
    num = dis_min.shape[0]
    loss = 0
    for i in range(num):
        if dis_min[i] >= margin:
            continue
        else:
            loss += (margin-dis_min[i])
    return loss


class maskGAN_Attack:
    def __init__(self,
                 device,
                 target_model,
                 input_nc,
                 image_nc,
                 epsilon):
        self.device = device
        self.target_model = target_model
        self.input_nc = input_nc
        self.output_nc = image_nc
        self.epsilon = epsilon

        self.netG = models.Generator(input_nc, image_nc).to(device)
        self.netDisc = models.Discriminator(image_nc).to(device)

        # initialize all weights
        self.netG.apply(weights_init)
        #self.netDisc.apply(weights_init)

        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=0.0001,betas=[0.5,0.9])
        self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                            lr=0.0001,betas=[0.5,0.9])
        if not os.path.exists(generator_models_path):
            os.makedirs(generator_models_path)
        if not os.path.exists(discriminator_models_path):
            os.makedirs(discriminator_models_path)

    def train_batch(self, train_images):
        # optimize D
        lambda_gp = 10
        for i in range(1):
            # self.netG.eval()
            # self.netDisc.train()
            self.optimizer_D.zero_grad()
            perturbation = self.netG(train_images)
            adv_images = 2*torch.clamp(perturbation+(train_images+1.0)/2.0,0,1) -1

            pred_real = self.netDisc(train_images)
            #标签反转和软标签
            true_num = math.floor(pred_real.shape[0]*0.95)
            false_num = pred_real.shape[0]-true_num
            real_label_true = 0.1*torch.rand(true_num,device=self.device)+0.9
            real_label_false = 0.1*torch.rand(false_num,device=self.device)
            real_label = torch.cat((real_label_true,real_label_false))

            loss_D_real = F.binary_cross_entropy_with_logits(pred_real, real_label)

            loss_D_real.backward()
            pred_fake = self.netDisc(adv_images.detach())

            fake_label_true = 0.1 * torch.rand(true_num, device=self.device)
            fake_label_false = 0.1 * torch.rand(false_num, device=self.device)+0.9
            fake_label = torch.cat((fake_label_true, fake_label_false))

            loss_D_fake = F.binary_cross_entropy_with_logits(pred_fake, fake_label)

            loss_D_fake.backward()



            loss_D_GAN = loss_D_fake + loss_D_real
            self.optimizer_D.step()

        # optimize G
        for i in range(1):
            self.optimizer_G.zero_grad()
            pred_fake = self.netDisc(adv_images)
            loss_G_fake = F.binary_cross_entropy_with_logits(pred_fake, torch.ones_like(pred_fake,device=self.device))
            loss_G_fake.backward(retain_graph = True)


            #pixel loss
            E = self.epsilon/256
            E_tensor = E*torch.ones_like(perturbation).to(self.device)
            pixel_loss = torch.max(torch.abs(perturbation),E_tensor)
            pixel_loss = torch.mean(pixel_loss)
            # calculate perturbation norm
            C = 3
            C_tensor = C*torch.ones(perturbation.shape[0],1).to(self.device)
            loss_perturb = torch.max(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1),C_tensor)
            loss_perturb = torch.mean(loss_perturb)

            # cal adv loss
            self.target_model.eval()
            real_feature = self.target_model.forward(train_images*10.)
            fake_feature = self.target_model.forward(adv_images*10.)
            loss_adv = torch.mean(torch.cosine_similarity(real_feature,fake_feature,dim=1))

            adv_lambda = 10
            pert_lambda = 1
            loss_G = adv_lambda*loss_adv + pert_lambda*loss_perturb


            loss_G.backward()
            self.optimizer_G.step()

        return loss_D_GAN.item(), loss_G_fake.item(), loss_perturb.item(), loss_adv.item(),pixel_loss.item()

    def train(self, train_dataloader, epochs):
        logger = get_logger('./log/stage_1/train_loss_log.log')
        logger.info('start training!')
        for epoch in range(1, epochs+1):

            if epoch == 200:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=0.0001,betas=[0.5,0.9])
                self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                                    lr=0.0001,betas=[0.5,0.9])
            if epoch == 300:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=0.0001,betas=[0.5,0.9])
                self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                                    lr=0.0001,betas=[0.5,0.9])
            loss_D_sum = 0
            loss_G_fake_sum = 0
            loss_perturb_sum = 0
            loss_adv_sum = 0
            loss_pixel_sum = 0
            for i, data in enumerate(train_dataloader, start=0):
                train_images = data
                train_images = train_images.to(self.device)
                loss_D_batch, loss_G_fake_batch, loss_perturb_batch, loss_adv_batch,loss_pixel_batch = self.train_batch(train_images)
                loss_D_sum += loss_D_batch
                loss_G_fake_sum += loss_G_fake_batch
                loss_perturb_sum += loss_perturb_batch
                loss_adv_sum += loss_adv_batch
                loss_pixel_sum += loss_pixel_batch
            # print statistics
            num_batch = len(train_dataloader)
            logger.info("epoch %d:\nloss_D: %.3f, loss_G_fake: %.3f,\
             \nloss_perturb: %.3f, loss_adv: %.3f, loss_pixel: %.3f \n" %
                  (epoch, loss_D_sum/num_batch, loss_G_fake_sum/num_batch,
                   loss_perturb_sum/num_batch, loss_adv_sum/num_batch, loss_pixel_sum/num_batch))

            # save generator
            if epoch%5==0:
                netG_file_name = generator_models_path + 'netG_epoch_' + str(epoch) + '.pth'
                torch.save(self.netG.state_dict(), netG_file_name)
                netD_file_name = discriminator_models_path + 'netD_epoch_' + str(epoch) + '.pth'
                torch.save(self.netDisc.state_dict(), netD_file_name)
        logger.info('finish training!')