import torch
import models
import numpy as np
import os
import cv2
from torch.utils.data import DataLoader
#import sklearn.preprocessing
import datetime
import argparse
import sys
import torch.nn as nn
from backbone.model_irse import IR_50


def L2distance(x, y):
	return torch.sqrt(torch.sum((x - y)**2,dim=1))

class inverse_mse(nn.Module):
    def forward(self,fea1,fea2):
        nfea1 = fea1 / torch.linalg.norm(fea1, dim = 1).view(fea1.shape[0],1)
        nfea2 = fea2 / torch.linalg.norm(fea2, dim = 1).view(fea2.shape[0],1)
        num = fea1.shape[0]
        dis = -torch.sum(L2distance(nfea1,nfea2),dim=0)/num
        return dis

def projcet_function(perturbation,C_norm):
    pert_norm = torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1)
    num = perturbation.shape[0]
    pert = torch.zeros_like(perturbation)
    for k in range(num):
        C = pert_norm[k] / C_norm
        pert[k] = perturbation[k] / C
    return pert


def mask_generation(train_img_list_path,data_dir,pretrained_generator_path,train_num,input_num,DEVICE,BATCH_SIZE,n, mask_out, norm):
    img_list = open(train_img_list_path)
    files = img_list.readlines()
    img_num = len(files)
    IMG = np.ones((img_num, 3, 112, 112), dtype='float32')
    # load the generator of adversarial examples
    gen_input_nc = 3
    image_nc = 3
    pretrained_G = models.Generator(gen_input_nc, image_nc).to(DEVICE)
    pretrained_G.load_state_dict(torch.load(pretrained_generator_path))
    pretrained_G.eval()
    for i in range(img_num):
        name = files[i]
        img_name = os.path.join(data_dir, name)
        img_name = img_name.split('\n')[0]
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        IMG[i, :, :, :] = img
    IMG = torch.from_numpy(IMG)
    IMG = IMG - 127.5
    IMG = IMG * 0.0078125

    img_dataloader = DataLoader(IMG,batch_size=train_num*BATCH_SIZE,shuffle=False,num_workers=1)
    for i,data in enumerate(img_dataloader,start=0):
        data = data.to(DEVICE)
        mask = torch.zeros([BATCH_SIZE,3,112,112]).to(DEVICE)
        perturbation = pretrained_G(data)
        for j in range(BATCH_SIZE):
            mask[j, :, :, :] = torch.mean(perturbation[(j*train_num):(j*train_num+input_num),:,:,:], dim=0)
        mask = mask*256
        mask= projcet_function(mask,norm)
        mask = mask.detach().cpu().numpy()
        for j in range(BATCH_SIZE):
            noise_j = mask[j]
            savenpy = os.path.join(mask_out, 'mask_id%d.npy'% (i*BATCH_SIZE+j))
            np.save(savenpy, noise_j)
            noise_j = noise_j.astype(np.uint8)
            noise_j = np.transpose(noise_j, (1, 2, 0))
            savebmp = os.path.join(mask_out, 'mask_id%d.bmp' % (i*BATCH_SIZE+j))
            cv2.imwrite(savebmp, noise_j[..., ::-1])





def main(args):
    print(args)
    time1 = datetime.datetime.now()
    DEVICE = torch.device("cuda:0")
    mask_generation(args.query_train_image_list,args.query_image_dir,args.pretrained_generator,args.train_img_per_id,args.input_img_per_id,DEVICE,args.batch_size,args.input_img_per_id,args.mask_out, args.norm)
    time2 = datetime.datetime.now()
    print("time consumed: ", time2 - time1)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, help='', default=1)
    parser.add_argument('--train_img_per_id', type=int, help='', default=10)
    parser.add_argument('--input_img_per_id', type=int, help='', default=10)
    parser.add_argument('--query_image_dir', default='../data', help='image path')
    parser.add_argument('--query_train_image_list', default='../data/list/privacy_train_v3_10.lst',
                        help='image path')
    parser.add_argument('--pretrained_generator',default='./models/stage_2_model/netG_stage_2.pth',help='generating mask')
    parser.add_argument('--mask_out', default = './mask_out', help='msk path')
    parser.add_argument('--step', type=int, default=50, help='loss type')
    parser.add_argument('--start_id', type=int, default=-1, help='loss type')
    parser.add_argument('--end_id', type=int, default=500, help='loss type')
    parser.add_argument('--norm', type=float, default=1200, help='loss type')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))






