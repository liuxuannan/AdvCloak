import torch
from backbone.model_irse import IR_50
import os
import sys
import datetime
import argparse
import cv2
import numpy as np
from torch.utils.data import DataLoader
from maskGAN_average import maskGAN_Attack

image_nc = 3
input_nc = 3
def main(args):
    print(args)

    time1 = datetime.datetime.now()
    DEVICE = torch.device("cuda:0")
    # initialize the model
    target_model_root =args.pretrained
    target_model = IR_50([112, 112]),
    target_model = target_model[0]
    target_model.load_state_dict(torch.load(target_model_root))
    #multi-GPU setting
    device_ids = [0]
    target_model = target_model.cuda(device=device_ids[0])

    train_list = open(args.train_list)
    train_files = train_list.readlines()
    train_num = 48000

#加载数据
    train_img = np.ones((train_num, 3, 112, 112), dtype='float32')
    for i in range(train_num):
        name = train_files[i]
        img_name_pre = name.split('\n')[0]
        img_name = args.data_dir + img_name_pre
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        train_img[i, :, :, :] = img


    train_img = torch.from_numpy(train_img)  # from_numpy将numpy数组转换成tensor


    #归一化
    train_img = train_img - 127.5
    train_img = train_img * 0.0078125


    train_dataloader = DataLoader(train_img, batch_size=args.train_num*args.num_shot, shuffle=False, num_workers=1)

    maskGAN = maskGAN_Attack(DEVICE,
                           target_model,
                           input_nc,
                           image_nc,
                           args.alpha,
                           args.loss_type,
                           args.upper,
                           args.lower,
                           args.nter
                             )

    maskGAN.train(train_dataloader,args.epoch,args.train_num,args.num_shot)
    time2  = datetime.datetime.now()
    print("time consumed: ", time2 - time1)







def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', default='./model/source_model/IR_50-ArcFace-casia/Backbone_IR_50_Epoch_73_Batch_138000_Time_2020-05-07-23-48_checkpoint.pth', help='training set directory')
    parser.add_argument('--train_list', default='../data/CASIA-WebFace_112/list_16.lst',
                        help='training set directory')
    parser.add_argument('--data_dir', default='../data',
                        help='training set directory')
    parser.add_argument('--train_num', type=int, help='training num for a person', default=16)
    parser.add_argument('--num_shot', type=int, default=2)
    parser.add_argument('--epoch', type=int, help='', default=100)
    parser.add_argument('--alpha', type=float, default=8, help='loss type')
    parser.add_argument('--loss_type', type=int, default=0, help='loss type')
    parser.add_argument('--upper', type=float, help='', default=1.0)
    parser.add_argument('--lower', type=float, help='', default=0.0)
    parser.add_argument('--nter', type=int, help='', default=40)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
