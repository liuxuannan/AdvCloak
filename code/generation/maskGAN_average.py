import torch.nn as nn
import torch
import numpy as np
import models
import torch.nn.functional as F
import torchvision
import os
from torch.utils.data import DataLoader
from torch.autograd import Variable
import logging
import math
import cvxpy as cp

models_path = './model/stage_2_model/generator_models/'
pretrained_generator_path = './model/stage_1_model/generator_models/netG_stage_1.pth'
pretrained_discriminator_path = './model/stage_1_model/discriminator_models/netD_stage_1.pth'

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
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1,1,1))).to(device)
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
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
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

def margin_loss(dis_min,margin):
    num = dis_min.shape[0]
    loss = 0
    for i in range(num):
        if dis_min[i] >= margin:
            continue
        else:
            loss += (margin-dis_min[i])
    return loss

class sim_average(nn.Module):
    def forward(self,fea1,fea2):
        sim = torch.mean(torch.cosine_similarity(fea1, fea2, dim=1))
        return sim

def intra_class_sim(fea1):
    nfea1 = fea1 / torch.linalg.norm(fea1, dim=1).view(fea1.shape[0], 1)
    fea1_ave = torch.mean(nfea1,dim=0)
    sim = torch.cosine_similarity(nfea1, fea1_ave, dim=1)
    return sim
# class sim_average(nn.Module):
#     def forward(self,fea1,fea2):
#         sim = torch.cosine_similarity(fea1, fea2, dim=1)
#         p = sim/(torch.sum(sim))
#         sim_p = torch.sum(torch.mul(sim,p))
#         return sim_p

class affine_hull_cvx(nn.Module):
    def forward(self, fea1, fea2):
        nfea1 = fea1 / torch.linalg.norm(fea1, dim=1).view(fea1.shape[0], 1)
        nfea2 = fea2 / torch.linalg.norm(fea2, dim=1).view(fea2.shape[0], 1)
        # nfea2 --> A, nfea1 --> y, caculate x.
        # Using cvx to calculate variable x

        A = nfea2.detach().cpu().numpy()
        XX = torch.tensor(np.zeros((nfea1.shape[0],nfea1.shape[0])), dtype=torch.float32, device=torch.device("cuda:0"))
        for i in range(nfea1.shape[0]):
            y = nfea1[i].detach().cpu().numpy()

            x = cp.Variable(nfea1.shape[0])
            objective = cp.Minimize(cp.sum_squares(x @ A - y))
            constraints = [sum(x)==1]
            prob = cp.Problem(objective, constraints)
            prob.solve()
            x_tensor = torch.tensor(x.value, dtype=torch.float32, device=torch.device("cuda:0"))
            XX[i]= x_tensor
        #embed()
        num = nfea1.shape[0]
        sim = torch.mean(torch.cosine_similarity(torch.mm(XX.detach(), nfea2), nfea1, dim=1))
        return sim

class convex_hull_cvx_dyn(nn.Module):
    def forward(self, fea1, fea2, lower = 0.0, upper = 1.0):
        nfea1 = fea1 / torch.linalg.norm(fea1, dim=1).view(fea1.shape[0], 1)
        nfea2 = fea2 / torch.linalg.norm(fea2, dim=1).view(fea2.shape[0], 1)
        # nfea2 --> A, nfea1 --> y, caculate x.
        # Using cvx to calculate variable x
        lowerbound = lower
        upperbound = upper
        A = nfea2.detach().cpu().numpy()
        XX = torch.tensor(np.zeros((nfea1.shape[0], nfea1.shape[0])), dtype=torch.float32,
                          device=torch.device("cuda:0"))
        for i in range(nfea1.shape[0]):
            y = nfea1[i].detach().cpu().numpy()

            x = cp.Variable(nfea1.shape[0])
            # embed()
            objective = cp.Minimize(cp.sum_squares(x @ A - y))
            #   objective = cp.Minimize(cp.sum(cp.norm(x @ A - y, axis=0)))
            constraints = [sum(x) == 1, lowerbound <= x, x <= upperbound]
            prob = cp.Problem(objective, constraints)
            prob.solve()
            x_tensor = torch.tensor(x.value, dtype=torch.float32, device=torch.device("cuda:0"))
            XX[i] = x_tensor
        num = nfea1.shape[0]
        sim = torch.mean(torch.cosine_similarity(torch.mm(XX.detach(), nfea2), nfea1, dim=1))
        #embed()
        return sim

class maskGAN_Attack:
    def __init__(self,
                 device,
                 target_model,
                 input_nc,
                 image_nc,
                 epsilon,
                 loss_type,
                 upper,
                 lower,
                 nter):
        self.device = device
        self.target_model = target_model
        self.input_nc = input_nc
        self.output_nc = image_nc
        self.epsilon = epsilon
        self.loss_type = loss_type
        self.netG = models.Generator(input_nc, image_nc).to(device)
        self.netDisc = models.Discriminator(image_nc).to(device)
        self.upper = upper
        self.lower = lower
        self.nter = nter
        if loss_type == 0:
            self.LossFunction = sim_average()
        elif loss_type == 7:  # center
            self.LossFunction = convex_hull_cvx_dyn()
        elif loss_type == 8:
            self.LossFunction = affine_hull_cvx()
        elif loss_type == 9:  # convexhull，inducehull
            self.LossFunction = convex_hull_cvx_dyn()

        # initialize all weights
        #self.netG.apply(weights_init)
        #self.netDisc.apply(weights_init)
        self.netG.load_state_dict(torch.load(pretrained_generator_path))
        self.netDisc.load_state_dict(torch.load(pretrained_discriminator_path))
        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=0.00000002,betas=[0.5,0.9])
        self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                            lr=0.00000002,betas=[0.5,0.9])


        if not os.path.exists(models_path):
            os.makedirs(models_path)

    def train_batch(self, real_images,train_num,num_shot,epoch):
        # optimize D
        lambda_gp = 10
        for i in range(1):
            self.optimizer_D.zero_grad()
            perturbation = self.netG(real_images)
            mask = torch.zeros([num_shot,3,112,112]).to(self.device)
            adv_images = torch.zeros_like(real_images).to(self.device)
            for i in range(num_shot):
                mask[i,:,:,:] = torch.mean(perturbation[i*train_num:(i+1)*train_num,:,:,:],dim=0)
                adv_images[i * train_num:(i + 1) * train_num, :, :, :] = \
                    2 * torch.clamp(mask[i]+(real_images[i * train_num:(i + 1) * train_num, :, :, :]+1.0)/2.0,0,1) -1

            pred_real = self.netDisc(real_images)
            # 标签反转和软标签
            true_num = math.floor(pred_real.shape[0] * 0.95)
            false_num = pred_real.shape[0] - true_num
            real_label_true = 0.1 * torch.rand(true_num, device=self.device) + 0.9
            real_label_false = 0.1 * torch.rand(false_num, device=self.device)
            real_label = torch.cat((real_label_true, real_label_false))
            loss_D_real = F.binary_cross_entropy_with_logits(pred_real, real_label)
            #loss_D_real = F.binary_cross_entropy_with_logits(pred_real, torch.ones_like(pred_real,device=self.device))
            loss_D_real.backward()

            pred_fake = self.netDisc(adv_images.detach())
            fake_label_true = 0.1 * torch.rand(true_num, device=self.device)
            fake_label_false = 0.1 * torch.rand(false_num, device=self.device) + 0.9
            fake_label = torch.cat((fake_label_true, fake_label_false))
            loss_D_fake = F.binary_cross_entropy_with_logits(pred_fake, fake_label)
            #loss_D_fake = F.binary_cross_entropy_with_logits(pred_fake, torch.zeros_like(pred_fake,device=self.device))
            loss_D_fake.backward()

            loss_D_GAN = loss_D_fake + loss_D_real
            self.optimizer_D.step()

        # optimize G
        for i in range(1):
            self.optimizer_G.zero_grad()
            # cal G's loss in GAN
            pred_fake = self.netDisc(adv_images)
            loss_G_fake = F.binary_cross_entropy_with_logits(pred_fake, torch.ones_like(pred_fake, device=self.device))
            loss_G_fake.backward(retain_graph=True)

            # calculate perturbation norm
            C = 3
            C_tensor = C * torch.ones(mask.shape[0], 1).to(self.device)
            loss_perturb = torch.max(torch.norm(mask.view(mask.shape[0], -1), 2, dim=1), C_tensor)
            # loss_perturb = torch.max(loss_perturb - C, torch.zeros(1, device=self.device))
            loss_perturb = torch.mean(loss_perturb)

            # cal adv loss
            self.target_model.eval()
            real_feature = self.target_model.forward(real_images * 10.)
            fake_feature = self.target_model.forward(adv_images * 10.)
            real_feature_1 = real_feature[0:train_num,:]
            real_feature_2 = real_feature[train_num:num_shot*train_num,:]
            fake_feature_1 = fake_feature[0:train_num,:]
            fake_feature_2 = fake_feature[train_num:num_shot*train_num,:]

            if self.loss_type == 9:
                if epoch < self.nter:
                    loss_adv_1 = self.LossFunction(fake_feature_1, real_feature_1, 1 / train_num, 1 / train_num)
                    loss_adv_2 = self.LossFunction(fake_feature_2, real_feature_2, 1 / train_num, 1 / train_num)
                    loss_adv = (loss_adv_1 + loss_adv_2) / 2
                else:
                    loss_adv_1 = self.LossFunction(fake_feature_1, real_feature_1, self.lower, self.upper)
                    loss_adv_2 = self.LossFunction(fake_feature_2, real_feature_2, self.lower, self.upper)
                    loss_adv = (loss_adv_1+loss_adv_2)/2
            elif self.loss_type == 7: # center
                loss_adv_1 = self.LossFunction(fake_feature_1, real_feature_1, 1 / train_num, 1 / train_num)
                loss_adv_2 = self.LossFunction(fake_feature_2, real_feature_2, 1 / train_num, 1 / train_num)
                loss_adv = (loss_adv_1 + loss_adv_2) / 2
            else:
                loss_adv_1 = self.LossFunction(fake_feature_1, real_feature_1)
                loss_adv_2 = self.LossFunction(fake_feature_2, real_feature_2)
                loss_adv = (loss_adv_1 + loss_adv_2) / 2
            # cal intra-class sim loss
            # loss_anti = 0
            #   loss_anti += torch.cosine_similarity()

            adv_lambda = 15
            pert_lambda = 1
            loss_G = adv_lambda * loss_adv + pert_lambda*loss_perturb
            loss_G.backward()
            self.optimizer_G.step()

        return loss_D_GAN.item(), loss_G_fake.item(), loss_perturb.item(), loss_adv.item()

    def train(self, train_dataloader, epochs,train_num, num_shot):
        logger = get_logger('./log/stage_2/train_loss_log.log')
        logger.info('start training!')
        for epoch in range(1, epochs+1):

            loss_D_sum = 0
            loss_G_fake_sum = 0
            loss_perturb_sum = 0
            loss_adv_sum = 0
            for i, data in enumerate(train_dataloader, start=0):
                train_images = data.to(self.device)
                loss_D_batch, loss_G_fake_batch, loss_perturb_batch, loss_adv_batch = self.train_batch(train_images, train_num,num_shot,epoch)
                loss_D_sum += loss_D_batch
                loss_G_fake_sum += loss_G_fake_batch
                loss_perturb_sum += loss_perturb_batch
                loss_adv_sum += loss_adv_batch

            # print statistics
            num_batch = len(train_dataloader)
            logger.info("epoch %d:\nloss_D: %.3f, loss_G_fake: %.3f,\
             \nloss_perturb: %.3f, loss_adv: %.3f, \n" %
                  (epoch, loss_D_sum/num_batch, loss_G_fake_sum/num_batch,
                   loss_perturb_sum/num_batch, loss_adv_sum/num_batch))

            # save generator
            if epoch%5==0:
                netG_file_name = models_path + 'netG_epoch_' + str(epoch) + '.pth'
                torch.save(self.netG.state_dict(), netG_file_name)
        logger.info('finish training!')

