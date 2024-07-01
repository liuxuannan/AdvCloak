import os
import numpy as np
import timeit
#import sklearn
from sklearn.metrics import roc_curve, auc
import cv2
import mxnet as mx
import sys
import argparse
#from prettytable import PrettyTable
import scipy.io as scio
import math
import heapq
import datetime
import logging


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

def projcet_function(perturbation,C_norm):
    pert_norm = np.linalg.norm(perturbation.reshape(1, -1), ord=2, axis=1,keepdims=True)
    print(pert_norm)
    C = pert_norm / C_norm
    pert = perturbation / C
    return pert

def get_image_feature(img_dir, img_list_path, pretrained, gpu_id, batch_size, img_type = None, lenth = None,img_per_tmp = None,
                      mask = None, img_save = None):
    img_list = open(img_list_path)
    #print('loading', model_path, model_num)
    ctx = mx.gpu(gpu_id)
    vec = pretrained.split(',')
    sym, arg_params, aux_params = mx.model.load_checkpoint(vec[0], int(vec[1]))
    all_layers = sym.get_internals()
    sym = all_layers['fc1_output']
    image_size = (112, 112)
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    model.bind(for_training=False, data_shapes=[('data', (batch_size, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)

    files = img_list.readlines()
    #print('files:', len(files))
    img_feats = []

    if lenth is not None:
        file_lenth = lenth
    else:
        file_lenth = len(files)
    start = 0
    while True:
        if start % 10000 ==0:
            print("processing", start)
        end = min(start + batch_size, file_lenth)
        if start >= end:
            break
        input_blob = np.zeros((batch_size, 3, image_size[0], image_size[1]), dtype=np.uint8)
        for i in range(start, end):
            img_name = files[i]
            #print(img_name)
            if img_type == 'lfw':
                img_name = img_name.split('/')
                a = img_name[0]
                b = img_name[1].split('\n')[0]
                out_dir = os.path.join(img_dir, "%s" % (a))
                img_name = os.path.join(out_dir, "%s" % (b))
            elif img_type == 'MF2':
                image_name = img_name.split()[1]
                img_name = os.path.join(img_dir, image_name)
            else:
                img_name = os.path.join(img_dir, img_name.split('\n')[0])

            #print(img_name)
            img = cv2.imread(img_name)
            if img is None:
                print(img_name)
                print(img)

            if mask is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mask_img_npy = os.path.join(mask, 'mask_id%d.npy'% int(i/img_per_tmp))
                adv_noise = np.load(mask_img_npy)

                if len(adv_noise.shape) == 4:
                    adv_noise = adv_noise[0]
                adv_noise = np.transpose(adv_noise, (1, 2, 0))
                img_float = img.astype(np.float64)
                adv_img = np.maximum(np.minimum(img_float + adv_noise, 255), 0)
                img = adv_img.astype(dtype=np.uint8)
                img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                if img_save:
                    name = files[i].split('\n')[0]
                    tmp_n, img_n = name.split('/')[1],name.split('/')[2]
                    img_n = img_n.split('.')[0] + '.bmp'
                    save_path = os.path.join(img_save, tmp_n)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    save_name = os.path.join(save_path, img_n)
                    cv2.imwrite(save_name, img)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.transpose(img, (2, 0, 1))  # 3*112*112, RGB
            input_blob[i-start] = img
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        model.forward(db, is_train=False)
        feat = model.get_outputs()[0].asnumpy()
        for i in range(end - start):
            fea = feat[i]
            fea = fea.flatten()
            img_feats.append(fea)
        start = end
    img_feats = np.array(img_feats).astype(np.float32)
    img_feats = sklearn.preprocessing.normalize(img_feats)
    return img_feats

def verification(query_img_feats_mask, query_img_feats, gallery_noise_feats, img_per_tmp = 10):
    print(query_img_feats_mask.shape)
    print(query_img_feats.shape)
    print(gallery_noise_feats.shape)

    query_img_num = int(query_img_feats_mask.shape[0] / img_per_tmp)
    positive_num = query_img_num * int(img_per_tmp * (img_per_tmp - 1) / 2)
    negative_num = query_img_feats_mask.shape[0] * gallery_noise_feats.shape[0]
    print(query_img_num, positive_num, negative_num)

    score_positive = np.zeros((query_img_num, int(img_per_tmp * (img_per_tmp - 1) / 2)))
    score_negative = np.zeros((query_img_feats_mask.shape[0], gallery_noise_feats.shape[0]))

    for id in range(query_img_num):
        pair = 0
        for i in range(img_per_tmp):
            for j in range(i, img_per_tmp):
                if i == j:
                    continue
                else:
                    query_feat = query_img_feats_mask[id * img_per_tmp + i]
                    target_feat = query_img_feats[id * img_per_tmp + j]
                    similarity = np.dot(query_feat, target_feat.T)
                    #print(id, i, j, id * img_per_tmp + pair)
                    score_positive[id, pair] = similarity
                    pair += 1


    for id in range(query_img_num):
        for i in range(img_per_tmp):
            query_feat = query_img_feats_mask[id * img_per_tmp + i]
            similarity = np.dot(query_feat, gallery_noise_feats.T)
            score_negative[id * img_per_tmp + i,:] = similarity

    
    score_positive = score_positive.reshape((score_positive.shape[0]*score_positive.shape[1],))
    score_negative = score_negative.reshape((score_negative.shape[0]*score_negative.shape[1],))
    score = np.concatenate((score_positive, score_negative))

    pair_label = np.zeros((len(score),))
    pair_label[0:len(score_positive)] = 1

    fpr, tpr, _ = roc_curve(pair_label, score)
    roc_auc = auc(fpr, tpr)
    fpr = np.flipud(fpr)
    tpr = np.flipud(tpr)  # select largest tpr at same fpr

    
    tpr_fpr_row = []
    x_labels = [10 ** -7, 10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]
    for fpr_iter in np.arange(len(x_labels)):
        _, min_index = min(list(zip(abs(fpr - x_labels[fpr_iter]), range(len(fpr)))))
        tpr_fpr_row.append('%.2f' % (tpr[min_index] * 100))
    print(tpr_fpr_row)


def evaluation(gpu_id,query_img_feats_mask, query_img_feats, gallery_noise_feats, logger, img_per_tmp = 10):
    print(query_img_feats_mask.shape)
    print(query_img_feats.shape)
    print(gallery_noise_feats.shape)
    query_img_num = int(query_img_feats_mask.shape[0] / img_per_tmp)
    query_num = query_img_num * img_per_tmp * (img_per_tmp - 1)
    gallery_num = gallery_noise_feats.shape[0]
    print(query_num, gallery_num)

    query_img_feats_mask = mx.nd.array(query_img_feats_mask, ctx=mx.gpu(gpu_id))
    query_img_feats = mx.nd.array(query_img_feats, ctx=mx.gpu(gpu_id))
    gallery_noise_feats = mx.nd.array(gallery_noise_feats, ctx=mx.gpu(gpu_id))

    correct_num_top1 = 0
    correct_num_top5 = 0
    correct_num_top10 = 0
    for id in range(query_img_num):
        for i in range(img_per_tmp):
            for j in range(img_per_tmp):
                if i == j:
                    continue
                else:
                    query_feat = query_img_feats_mask[id * img_per_tmp + i]
                    target_feat = mx.nd.zeros((1, 512), ctx=mx.gpu(gpu_id))
                    target_feat[0] = query_img_feats[id * img_per_tmp + j]
                    
                    gallery_feat = mx.nd.concat(target_feat, gallery_noise_feats, dim=0)

                    ''' 
                    time_start = time.time()
                    similarity1 = np.dot(query_feat, gallery_feat.T)
                    top_inds1 = np.argsort(-similarity1)
                    time_end = time.time()
                    print("numpy dot+sort", time_end - time_start)
                    '''
                    # time_start = time.time()
                    similarity = mx.nd.dot(query_feat, mx.nd.transpose(gallery_feat))
                    top_inds = mx.nd.argsort(-similarity)
                    top_inds = top_inds.asnumpy()
                    # time_end = time.time()
                    # print("mxnet dot+sort", time_end - time_start)

                    # print(top_inds1, top_inds2)
                    # top_inds = top_inds2
                    if top_inds[0] == 0:
                        correct_num_top1 += 1
                    # else:
                    # print("wrong",id, i, j)
                    if 0 in top_inds[0:5]:
                        correct_num_top5 += 1
                    if 0 in top_inds[0:10]:
                        correct_num_top10 += 1
    logger.info("acc top1 = %1.5f, protect top1 = %1.5f" % (correct_num_top1 / float(query_num), 1.0-correct_num_top1 / float(query_num)) )
    logger.info("acc top5 = %1.5f, protect top5 = %1.5f" % (correct_num_top5 / float(query_num), 1.0- correct_num_top5 / float(query_num)) )
    logger.info("acc top10 = %1.5f, protect top10 = %1.5f" % (correct_num_top10 / float(query_num), 1.0-correct_num_top10 / float(query_num)) )


def test(query_image_dir, query_train_image_list, query_test_image_list, test_img_per_id, gallery_noise_dir,
         gallery_noise_list, pretrained, gpu, batch_size, mask = None, img_train_save = None, img_test_save = None, distract_lenth = None, gallery_image_type = None, log = None):
    query_train_img_feats_mask = get_image_feature(query_image_dir, query_train_image_list, pretrained,
                                              gpu, batch_size, img_type = None, img_per_tmp = 10, mask = mask, img_save = None)
    query_train_img_feats = get_image_feature(query_image_dir, query_train_image_list, pretrained,
                                                   gpu, batch_size)
    query_test_img_feats_mask = get_image_feature(query_image_dir, query_test_image_list, pretrained,
                                              gpu, batch_size, img_type = None, img_per_tmp = test_img_per_id, mask = mask, img_save = None)
    query_test_img_feats = get_image_feature(query_image_dir, query_test_image_list, pretrained,
                                                  gpu, batch_size)
    #gallery_noise_feats = get_image_feature(gallery_noise_dir, gallery_noise_list, pretrained,
    #                              gpu, batch_size, img_type = 'lfw')
    gallery_noise_feats = get_image_feature(gallery_noise_dir, gallery_noise_list, pretrained,
                                            gpu, batch_size, img_type=gallery_image_type, lenth = distract_lenth)
    
    logger = get_logger(log)
    logger.info('start training!')
    print('train verification:')
    verification(query_train_img_feats_mask, query_train_img_feats, gallery_noise_feats, img_per_tmp=10)
    print('test verification:')
    verification(query_test_img_feats_mask, query_test_img_feats, gallery_noise_feats, img_per_tmp=test_img_per_id)
    logger.info('train result:')
    evaluation(gpu, query_train_img_feats_mask, query_train_img_feats, gallery_noise_feats, logger, img_per_tmp = 10)
    logger.info('test result:')
    evaluation(gpu, query_test_img_feats_mask, query_test_img_feats, gallery_noise_feats, logger, img_per_tmp = test_img_per_id)
    logger.info('finish training!')
def main(args):
    print(args)
    time1 = datetime.datetime.now()
    test(args.query_image_dir, args.query_train_image_list, args.query_test_image_list, args.test_img_per_id,
    args.gallery_noise_dir, args.gallery_noise_list,
    args.pretrained, args.gpu, args.batch_size, args.msk_dir, args.img_train_save, args.img_test_save, args.lenth, args.gallery_image_type, args.log_save)
    time2 = datetime.datetime.now()
    print("time consumed: ", time2 - time1)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, help='', default=0)
    parser.add_argument('--batch-size', type=int, help='', default=50)
    parser.add_argument('--lenth', type=int, help='', default=10000)
    parser.add_argument('--test_img_per_id', type=int, help='', default=5)
    parser.add_argument('--gallery_image_type', default='MF2', help='query image type')
    parser.add_argument('--query_image_dir', default='../data', help='image path')
    parser.add_argument('--query_train_image_list', default='../data/list/privacy_train_v3_10.lst',
                        help='image path')
    parser.add_argument('--query_test_image_list', default='../data/list/privacy_test_v3_5.lst',
                        help='image path')
    parser.add_argument('--gallery_noise_dir', default='../data/Distractor', help='image path')
    parser.add_argument('--gallery_noise_list', default='../data/Distractor/lst',
                        help='image path')
    parser.add_argument('--pretrained', type=str, help='', default='./target_model/r50_webface_arc_bs/model,146')
    parser.add_argument('--msk_dir', default = '../generation/mask_out', help='msk path')
    parser.add_argument('--img_train_save', default = None, help='msk path')
    parser.add_argument('--img_test_save', default= None, help='msk path')
    parser.add_argument('--log_save', default='./test_log/loss_log.log', help='log path')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

