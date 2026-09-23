# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
from PIL import Image
import sys
import collections
import time
from datetime import timedelta

from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
import cupy as cp 
from cuml.cluster import KMeans as cuKMeans
from cuml.cluster import AgglomerativeClustering as cuAgglomerativeClustering

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from clustercontrast import datasets
from clustercontrast import models
from clustercontrast.models.cm import ClusterMemory
from clustercontrast.trainers import ClusterContrastTrainer
from clustercontrast.evaluators import Evaluator, extract_features
from clustercontrast.utils.data import IterLoader
from clustercontrast.utils.data import transforms as T
from clustercontrast.utils.data.preprocessor import Preprocessor, Preprocessor_color, Preprocessor_gmm, Preprocessor_color_gmm
from clustercontrast.utils.logging import Logger
from clustercontrast.utils.serialization import load_checkpoint, save_checkpoint
from clustercontrast.utils.faiss_rerank import compute_jaccard_distance
from clustercontrast.utils.data.sampler import RandomMultipleGallerySampler, RandomMultipleGallerySamplerNoCam
import os
import torch.utils.data as data
from torch.autograd import Variable
import math
from ChannelAug import ChannelAdap, ChannelAdapGray, ChannelRandomErasing, ChannelExchange, Gray
from collections import Counter
start_epoch = best_mAP = 0

def get_data(name, data_dir,trial=0):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root, trial=trial)
    return dataset

class channel_select(object):
    def __init__(self,channel=0):
        self.channel = channel
    def __call__(self, img):
        if self.channel == 3:
            img_gray = img.convert('L')
            np_img = np.array(img_gray, dtype=np.uint8)
            img_aug = np.dstack([np_img, np_img, np_img])
            img_PIL=Image.fromarray(img_aug, 'RGB')
        else:
            np_img = np.array(img, dtype=np.uint8)
            np_img = np_img[:,:,self.channel]
            img_aug = np.dstack([np_img, np_img, np_img])
            img_PIL=Image.fromarray(img_aug, 'RGB')
        return img_PIL

def get_train_loader_ir(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None, no_cam=False, train_transformer=None, train_transformer1=None):
    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        if no_cam:
            sampler = RandomMultipleGallerySamplerNoCam(train_set, num_instances)
        else:
            sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    if train_transformer1 is None:
        train_loader = IterLoader(
            DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                    batch_size=batch_size, num_workers=workers, sampler=sampler,
                    shuffle=False, pin_memory=True, drop_last=True), length=iters)
    else:
        train_loader = IterLoader(
            DataLoader(Preprocessor_color(train_set, root=dataset.images_dir, transform=train_transformer, transform1=train_transformer1),
                       batch_size=batch_size, num_workers=workers, sampler=sampler,
                       shuffle=False, pin_memory=True, drop_last=True), length=iters)
    return train_loader

def get_train_loader_color(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None, no_cam=False, train_transformer=None, train_transformer1=None):
    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        if no_cam:
            sampler = RandomMultipleGallerySamplerNoCam(train_set, num_instances)
        else:
            sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    if train_transformer1 is None:
        train_loader = IterLoader(
            DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                       batch_size=batch_size, num_workers=workers, sampler=sampler,
                       shuffle=False, pin_memory=True, drop_last=True), length=iters)
    else:
        train_loader = IterLoader(
            DataLoader(Preprocessor_color(train_set, root=dataset.images_dir, transform=train_transformer, transform1=train_transformer1),
                       batch_size=batch_size, num_workers=workers, sampler=sampler,
                       shuffle=False, pin_memory=True, drop_last=True), length=iters)
    return train_loader

def get_test_loader(dataset, height, width, batch_size, workers, testset=None, test_transformer=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    if test_transformer is None:
        test_transformer = T.Compose([
            T.Resize((height, width), interpolation=3),
            T.ToTensor(),
            normalizer
        ])

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)
    return test_loader

def create_model(args):
    model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout,
                          num_classes=0, pooling_type=args.pooling_type)
    # use CUDA
    model.cuda()
    model = nn.DataParallel(model)
    return model

class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size = (144,288)):

        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_gall_feat(model, gall_loader, ngall):
    pool_dim=2048
    net = model
    net.eval()
    print ('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat_pool = np.zeros((ngall, pool_dim))
    gall_feat_fc = np.zeros((ngall, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label ) in enumerate(gall_loader):
            batch_num = input.size(0)
            flip_input = fliplr(input)
            input = Variable(input.cuda())
            feat_fc = net( input,input, 2)
            flip_input = Variable(flip_input.cuda())
            feat_fc_1 = net( flip_input,flip_input, 2)
            feature_fc = (feat_fc.detach() + feat_fc_1.detach())/2
            fnorm_fc = torch.norm(feature_fc, p=2, dim=1, keepdim=True)
            feature_fc = feature_fc.div(fnorm_fc.expand_as(feature_fc))
            gall_feat_fc[ptr:ptr+batch_num,: ]   = feature_fc.cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return gall_feat_fc
    
def extract_query_feat(model,query_loader,nquery):
    pool_dim=2048
    net = model
    net.eval()
    print ('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat_pool = np.zeros((nquery, pool_dim))
    query_feat_fc = np.zeros((nquery, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label ) in enumerate(query_loader):
            batch_num = input.size(0)
            flip_input = fliplr(input)
            input = Variable(input.cuda())
            feat_fc = net( input, input,1)
            flip_input = Variable(flip_input.cuda())
            feat_fc_1 = net( flip_input,flip_input, 1)
            feature_fc = (feat_fc.detach() + feat_fc_1.detach())/2
            fnorm_fc = torch.norm(feature_fc, p=2, dim=1, keepdim=True)
            feature_fc = feature_fc.div(fnorm_fc.expand_as(feature_fc))
            query_feat_fc[ptr:ptr+batch_num,: ]   = feature_fc.cpu().numpy()
            
            ptr = ptr + batch_num         
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return query_feat_fc

def pairwise_distance(features_q, features_g):
    x = torch.from_numpy(features_q)
    y = torch.from_numpy(features_g)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(1, -2, x, y.t())
    return dist_m.numpy()

def process_test_regdb(img_dir, trial = 1, modal = 'visible'):
    if modal=='visible':
        input_data_path = img_dir + 'idx/test_visible_{}'.format(trial) + '.txt'
    elif modal=='thermal':
        input_data_path = img_dir + 'idx/test_thermal_{}'.format(trial) + '.txt'
    
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [img_dir + '/' + s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        
    return file_image, np.array(file_label)

def eval_regdb(distmat, q_pids, g_pids, max_rank = 20):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0. # number of valid query
    
    # only two cameras
    q_camids = np.ones(num_q).astype(np.int32)
    g_camids = 2* np.ones(num_g).astype(np.int32)
    
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()

        # compute mINP
        # refernece Deep Learning for Person Re-identification: A Survey and Outlook
        pos_idx = np.where(raw_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        inp = cmc[pos_max_idx]/ (pos_max_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    return all_cmc, mAP, mINP

def main():
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
    log_s1_name = 'RegDB_s1_colearning1'
    log_s2_name = 'RegDB_s2_colearning1'
    # main_worker_stage1_robust_loss(args, log_s1_name)                  # Stage 1
    # main_worker_stage2_robust_loss(args, log_s1_name, log_s2_name)     # Stage 2
    # main_worker_stage1_colearning(args, log_s1_name)                   # Stage 1
    main_worker_stage2_colearning(args, log_s1_name, log_s2_name)        # Stage 2

def main_worker_stage1(args,log_s1_name):
    logs_dir_root = osp.join(args.logs_dir+'/'+log_s1_name)
    trial = args.trial
    # global start_epoch, best_mAP
    start_epoch =0
    best_mAP =0
    args.logs_dir = osp.join(logs_dir_root,str(trial))
    start_time = time.monotonic()

    # cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, str(trial)+'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset_ir = get_data('regdb_ir', args.data_dir,trial=trial)
    dataset_rgb = get_data('regdb_rgb', args.data_dir,trial=trial)

    test_loader_ir = get_test_loader(dataset_ir, args.height, args.width, args.batch_size, args.workers)
    test_loader_rgb = get_test_loader(dataset_rgb, args.height, args.width, args.batch_size, args.workers)
    # Create model
    model = create_model(args)

    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    # Trainer
    trainer = ClusterContrastTrainer(model)

    for epoch in range(args.epochs):
        with torch.no_grad():
            if epoch == 0:
                # DBSCAN cluster
                ir_eps = 0.3
                print('IR Clustering criterion: eps: {:.3f}'.format(ir_eps))
                cluster_ir = DBSCAN(eps=ir_eps, min_samples=4, metric='precomputed', n_jobs=-1)
                rgb_eps = 0.3
                print('RGB Clustering criterion: eps: {:.3f}'.format(rgb_eps))
                cluster_rgb = DBSCAN(eps=rgb_eps, min_samples=4, metric='precomputed', n_jobs=-1)

            print('==> Create pseudo labels for unlabeled RGB data')

            cluster_loader_rgb = get_test_loader(dataset_rgb, args.height, args.width,
                                             args.batch_size, args.workers,
                                             testset=sorted(dataset_rgb.train))
            features_rgb, _ = extract_features(model, cluster_loader_rgb, print_freq=50, mode=1)
            del cluster_loader_rgb,
            features_rgb = torch.cat([features_rgb[f].unsqueeze(0) for f, _, _ in sorted(dataset_rgb.train)], 0)

            
            print('==> Create pseudo labels for unlabeled IR data')
            cluster_loader_ir = get_test_loader(dataset_ir, args.height, args.width,
                                             args.batch_size, args.workers,
                                             testset=sorted(dataset_ir.train))
            features_ir, _ = extract_features(model, cluster_loader_ir, print_freq=50, mode=2)
            del cluster_loader_ir
            features_ir = torch.cat([features_ir[f].unsqueeze(0) for f, _, _ in sorted(dataset_ir.train)], 0)

            rerank_dist_ir = compute_jaccard_distance(features_ir, k1=args.k1, k2=args.k2,search_option=3)#rerank_dist_all_jacard[features_rgb.size(0):,features_rgb.size(0):]#
            pseudo_labels_ir = cluster_ir.fit_predict(rerank_dist_ir)
            del rerank_dist_ir

        # generate new dataset and calculate cluster centers
        @torch.no_grad()
        def generate_cluster_features(labels, features):
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])

            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]

            centers = torch.stack(centers, dim=0)
            return centers

        cluster_features_ir = generate_cluster_features(pseudo_labels_ir, features_ir)        
        rerank_dist_rgb = compute_jaccard_distance(features_rgb, k1=args.k1, k2=args.k2,search_option=3)#rerank_dist_all_jacard[:features_rgb.size(0),:features_rgb.size(0)]#
        pseudo_labels_rgb = cluster_rgb.fit_predict(rerank_dist_rgb)
        del rerank_dist_rgb
        
        num_cluster_ir = len(set(pseudo_labels_ir)) - (1 if -1 in pseudo_labels_ir else 0)
        num_cluster_rgb = len(set(pseudo_labels_rgb)) - (1 if -1 in pseudo_labels_rgb else 0)

        cluster_features_rgb = generate_cluster_features(pseudo_labels_rgb, features_rgb)
        memory_ir = ClusterMemory(model.module.num_features, num_cluster_ir, temp=args.temp,
                               momentum=args.momentum, use_hard=args.use_hard).cuda()
        memory_rgb = ClusterMemory(model.module.num_features, num_cluster_rgb, temp=args.temp,
                               momentum=args.momentum, use_hard=args.use_hard).cuda()
        memory_ir.features = F.normalize(cluster_features_ir, dim=1).cuda()
        memory_rgb.features = F.normalize(cluster_features_rgb, dim=1).cuda()

        trainer.memory_ir = memory_ir
        trainer.memory_rgb = memory_rgb

        pseudo_labeled_dataset_ir = []
        ir_label=[]
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_ir.train), pseudo_labels_ir)):
            if label != -1:
                pseudo_labeled_dataset_ir.append((fname, label.item(), cid))
                ir_label.append(label.item())
        print('==> Statistics for IR epoch {}: {} clusters'.format(epoch, num_cluster_ir))

        pseudo_labeled_dataset_rgb = []
        rgb_label=[]
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_rgb.train), pseudo_labels_rgb)):
            if label != -1:
                pseudo_labeled_dataset_rgb.append((fname, label.item(), cid))
                rgb_label.append(label.item())

        print('==> Statistics for RGB epoch {}: {} clusters'.format(epoch, num_cluster_rgb))

        ########################
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        height=args.height
        width=args.width
        train_transformer_rgb = T.Compose([
            T.Resize((height, width), interpolation=3),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            normalizer,
            ChannelRandomErasing(probability = 0.5)
        ])
        
        train_transformer_rgb1 = T.Compose([
            T.Resize((height, width), interpolation=3),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.5),
            T.ToTensor(),
            normalizer,
            ChannelRandomErasing(probability = 0.5),
            ChannelExchange(gray = 2)
        ])

        transform_thermal = T.Compose( [
            T.Resize((height, width), interpolation=3),
            T.Pad(10),
            T.RandomCrop((288, 144)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalizer,
            ChannelRandomErasing(probability = 0.5),
            ChannelAdapGray(probability =0.5)])

        train_loader_ir = get_train_loader_ir(args, dataset_ir, args.height, args.width,
                                        args.batch_size, args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset_ir, no_cam=args.no_cam, train_transformer=transform_thermal)

        train_loader_rgb = get_train_loader_color(args, dataset_rgb, args.height, args.width,
                                        128, args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset_rgb, no_cam=args.no_cam, train_transformer=train_transformer_rgb, train_transformer1=train_transformer_rgb1)

        train_loader_ir.new_epoch()
        train_loader_rgb.new_epoch()

        trainer.train(epoch, train_loader_ir, train_loader_rgb, optimizer,
                      print_freq=args.print_freq, train_iters=len(train_loader_ir))

        if epoch>=1 and ( (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1)):
##############################
            args.test_batch=64
            args.img_w=args.width
            args.img_h=args.height
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            transform_test = T.Compose([
                T.ToPILImage(),
                T.Resize((args.img_h,args.img_w)),
                T.ToTensor(),
                normalize,
            ])
            mode='all'
            data_path='/home/liyongxiang/code/ReID/dataset/RegDB/'
            query_img, query_label = process_test_regdb(data_path, trial=trial, modal='visible')
            gall_img, gall_label = process_test_regdb(data_path, trial=trial, modal='thermal')

            gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
            gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
            nquery = len(query_label)
            ngall = len(gall_label)
            queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
            query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
            query_feat_fc = extract_query_feat(model,query_loader,nquery)
            # for trial in range(1):
            ngall = len(gall_label)
            gall_feat_fc = extract_gall_feat(model, gall_loader, ngall)
            # fc feature
            distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
            cmc, mAP, mINP = eval_regdb(-distmat, query_label, gall_label)

            print('Test Trial: {}'.format(trial))
            print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                    cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))

            is_best = (mAP > best_mAP)
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))
############################
        lr_scheduler.step()
    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))

def main_worker_stage2(args,log_s1_name,log_s2_name):
    # logs_dir_root = osp.join('logs/'+log_s2_name)
    trial = args.trial
    stage1_logs_dir = osp.join(args.logs_dir+'/'+log_s1_name, str(trial))
    # global start_epoch, best_mAP
    start_epoch =0
    best_mAP =0
    stage2_logs_dir = osp.join(args.logs_dir+'/'+log_s2_name, str(trial))
    # args.logs_dir = osp.join(logs_dir_root,str(trial))
    start_time = time.monotonic()

    # cudnn.benchmark = True

    sys.stdout = Logger(osp.join(stage2_logs_dir, str(trial)+'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset_ir = get_data('regdb_ir', args.data_dir, trial=trial)
    dataset_rgb = get_data('regdb_rgb', args.data_dir, trial=trial)

    # Create model
    model = create_model(args)
    checkpoint = load_checkpoint(osp.join(stage1_logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    print("==> Load last checkpoint successfully! ", osp.join(stage1_logs_dir, 'model_best.pth.tar'))
    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    # Trainer
    trainer = ClusterContrastTrainer(model)

    for epoch in range(args.epochs):
        with torch.no_grad():
            if epoch == 0:
                # DBSCAN cluster
                ir_eps = 0.3
                print('IR Clustering criterion: eps: {:.3f}'.format(ir_eps))
                cluster_ir = DBSCAN(eps=ir_eps, min_samples=4, metric='precomputed', n_jobs=-1)
                rgb_eps = 0.3
                print('RGB Clustering criterion: eps: {:.3f}'.format(rgb_eps))
                cluster_rgb = DBSCAN(eps=rgb_eps, min_samples=4, metric='precomputed', n_jobs=-1)

            print('==> Create pseudo labels for unlabeled RGB data')

            cluster_loader_rgb = get_test_loader(dataset_rgb, args.height, args.width,
                                             args.batch_size, args.workers, 
                                             testset=sorted(dataset_rgb.train))
            features_rgb, _ = extract_features(model, cluster_loader_rgb, print_freq=50,mode=1)
            del cluster_loader_rgb,
            features_rgb = torch.cat([features_rgb[f].unsqueeze(0) for f, _, _ in sorted(dataset_rgb.train)], 0)

            
            print('==> Create pseudo labels for unlabeled IR data')
            cluster_loader_ir = get_test_loader(dataset_ir, args.height, args.width,
                                             args.batch_size, args.workers, 
                                             testset=sorted(dataset_ir.train))
            features_ir, _ = extract_features(model, cluster_loader_ir, print_freq=50,mode=2)
            del cluster_loader_ir
            features_ir = torch.cat([features_ir[f].unsqueeze(0) for f, _, _ in sorted(dataset_ir.train)], 0)

            rerank_dist_ir = compute_jaccard_distance(features_ir, k1=args.k1, k2=args.k2, search_option=3)#rerank_dist_all_jacard[features_rgb.size(0):,features_rgb.size(0):]#
            pseudo_labels_ir = cluster_ir.fit_predict(rerank_dist_ir)
            del rerank_dist_ir
            
        # generate new dataset and calculate cluster centers
        @torch.no_grad()
        def generate_cluster_features(labels, features):
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])

            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]

            centers = torch.stack(centers, dim=0)
            return centers

        cluster_features_ir = generate_cluster_features(pseudo_labels_ir, features_ir)
        rerank_dist_rgb = compute_jaccard_distance(features_rgb, k1=args.k1, k2=args.k2,search_option=3)#rerank_dist_all_jacard[:features_rgb.size(0),:features_rgb.size(0)]#
        pseudo_labels_rgb = cluster_rgb.fit_predict(rerank_dist_rgb)
        del rerank_dist_rgb
        num_cluster_ir = len(set(pseudo_labels_ir)) - (1 if -1 in pseudo_labels_ir else 0)
        num_cluster_rgb = len(set(pseudo_labels_rgb)) - (1 if -1 in pseudo_labels_rgb else 0)
        cluster_features_rgb = generate_cluster_features(pseudo_labels_rgb, features_rgb)

        memory_ir = ClusterMemory(model.module.num_features, num_cluster_ir, temp=args.temp,
                               momentum=args.momentum, use_hard=args.use_hard).cuda()
        memory_rgb = ClusterMemory(model.module.num_features, num_cluster_rgb, temp=args.temp,
                               momentum=args.momentum, use_hard=args.use_hard).cuda()
        memory_ir.features = F.normalize(cluster_features_ir, dim=1).cuda()
        memory_rgb.features = F.normalize(cluster_features_rgb, dim=1).cuda()

        trainer.memory_ir = memory_ir
        trainer.memory_rgb = memory_rgb

        pseudo_labeled_dataset_ir = []
        ir_label=[]
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_ir.train), pseudo_labels_ir)):
            if label != -1:
                pseudo_labeled_dataset_ir.append((fname, label.item(), cid))
                ir_label.append(label.item())
        print('==> Statistics for IR epoch {}: {} clusters'.format(epoch, num_cluster_ir))

        pseudo_labeled_dataset_rgb = []
        rgb_label=[]
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_rgb.train), pseudo_labels_rgb)):
            if label != -1:
                pseudo_labeled_dataset_rgb.append((fname, label.item(), cid))
                rgb_label.append(label.item())

        print('==> Statistics for RGB epoch {}: {} clusters'.format(epoch, num_cluster_rgb))

        if epoch>=0:
            ## PGM
            print("Progressive Graph Matching")
            i2r = {}
            r2i = {}
            R = []
            bgm = False
            if num_cluster_rgb >= num_cluster_ir:
                # clusternorm
                cluster_features_rgb = F.normalize(cluster_features_rgb, dim=1)
                cluster_features_ir = F.normalize(cluster_features_ir, dim=1)
                # [-1, 1] torch.mm(cluster_features_rgb, cluster_features_ir.T) #CostMatrix
                similarity = ((torch.mm(cluster_features_rgb, cluster_features_ir.T))/1).exp().cpu() #.exp().cpu()
                dis_similarity = (1 / (similarity))
                # dis_similarity = - similarity # binus
                cost = dis_similarity / 1
                tmp = torch.zeros(dis_similarity.shape[0], dis_similarity.shape[0] - dis_similarity.shape[1])
                cost = (torch.cat((cost, tmp), 1))
                unmatched_row = []
                row_ind, col_ind = linear_sum_assignment(cost)
                for idx, item in enumerate(row_ind):
                    if col_ind[idx] < similarity.shape[1]:
                        R.append((row_ind[idx], col_ind[idx]))
                        r2i[row_ind[idx]] = col_ind[idx]
                        i2r[col_ind[idx]] = row_ind[idx]
                    else:
                        unmatched_row.append(row_ind[idx])
                if bgm is False:
                    unmatched_cost = cost[unmatched_row][:,:dis_similarity.shape[1]]
                    unmatched_row_ind, unmatched_col_ind = linear_sum_assignment(unmatched_cost)
                    for idx, item in enumerate(unmatched_row_ind):
                        R.append((unmatched_row[idx], unmatched_col_ind[idx]))
                        r2i[unmatched_row[idx]] = unmatched_col_ind[idx]
                del cluster_features_ir, cluster_features_rgb
            else:
                cluster_features_rgb = F.normalize(cluster_features_rgb, dim=1)
                cluster_features_ir = F.normalize(cluster_features_ir, dim=1)
                similarity = ((torch.mm(cluster_features_ir, cluster_features_rgb.T))/1).exp().cpu() #.exp().cpu()
                dis_similarity = (1 / (similarity))
                cost = dis_similarity / 1
                tmp = torch.zeros(dis_similarity.shape[0], dis_similarity.shape[0] - dis_similarity.shape[1])
                cost = (torch.cat((cost, tmp), 1))
                unmatched_row = []
                row_ind, col_ind = linear_sum_assignment(cost)
                for idx, item in enumerate(row_ind):
                    if col_ind[idx] < similarity.shape[1]:
                        R.append((row_ind[idx], col_ind[idx]))
                        i2r[row_ind[idx]] = col_ind[idx]
                        r2i[col_ind[idx]] = row_ind[idx]
                    else:
                        unmatched_row.append(row_ind[idx])
                if bgm is False:
                    unmatched_cost = cost[unmatched_row][:,:dis_similarity.shape[1]]
                    unmatched_row_ind, unmatched_col_ind = linear_sum_assignment(unmatched_cost)
                    for idx, item in enumerate(unmatched_row_ind):
                        R.append((unmatched_row[idx], unmatched_col_ind[idx]))
                        i2r[unmatched_row[idx]] = unmatched_col_ind[idx]
                del cluster_features_ir, cluster_features_rgb
            
            print("Progressive Graph Matching Done")

        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        height=args.height
        width=args.width
        train_transformer_rgb = T.Compose([
            T.Resize((height, width), interpolation=3),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            normalizer,
            ChannelRandomErasing(probability = 0.5)
        ])
        
        train_transformer_rgb1 = T.Compose([
            T.Resize((height, width), interpolation=3),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.5),
            T.ToTensor(),
            normalizer,
            ChannelRandomErasing(probability = 0.5),
            ChannelExchange(gray = 2)
        ])

        transform_thermal = T.Compose( [
            T.Resize((height, width), interpolation=3),
            T.Pad(10),
            T.RandomCrop((288, 144)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalizer,
            ChannelRandomErasing(probability = 0.5),
            ChannelAdapGray(probability =0.5)])

        train_loader_ir = get_train_loader_ir(args, dataset_ir, args.height, args.width,
                                        args.batch_size, args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset_ir, no_cam=args.no_cam,train_transformer=transform_thermal)

        train_loader_rgb = get_train_loader_color(args, dataset_rgb, args.height, args.width,
                                        128, args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset_rgb, no_cam=args.no_cam,train_transformer=train_transformer_rgb,train_transformer1=train_transformer_rgb1)

        train_loader_ir.new_epoch()
        train_loader_rgb.new_epoch()

        trainer.train(epoch, train_loader_ir,train_loader_rgb, optimizer,
                      print_freq=args.print_freq, train_iters=len(train_loader_ir), i2r=i2r, r2i=r2i)

        if epoch>=1 and ( (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1)):
##############################
            args.test_batch=64
            args.img_w=args.width
            args.img_h=args.height
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            transform_test = T.Compose([
                T.ToPILImage(),
                T.Resize((args.img_h,args.img_w)),
                T.ToTensor(),
                normalize,
            ])

            data_path='/home/liyongxiang/code/ReID/dataset/RegDB/'
            query_img, query_label = process_test_regdb(data_path, trial=trial, modal='visible')
            gall_img, gall_label = process_test_regdb(data_path, trial=trial, modal='thermal')

            gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
            gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
            nquery = len(query_label)
            ngall = len(gall_label)
            queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
            query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
            query_feat_fc = extract_query_feat(model,query_loader,nquery)
            # for trial in range(1):
            ngall = len(gall_label)
            gall_feat_fc = extract_gall_feat(model,gall_loader,ngall)
            # fc feature
            distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
            cmc, mAP, mINP = eval_regdb(-distmat, query_label, gall_label)

            print('Test Trial: {}'.format(trial))
            print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                    cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))

            is_best = (mAP > best_mAP)
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(stage2_logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:2d}:  model mAP: {:5.2%}  best: {:5.2%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))
############################
        lr_scheduler.step()
    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))

def main_worker_stage1_robust_loss(args, log_s1_name):
    logs_dir_root = osp.join(args.logs_dir+'/'+log_s1_name)
    trial = args.trial
    # global start_epoch, best_mAP
    start_epoch =0
    best_mAP =0
    args.logs_dir = osp.join(logs_dir_root,str(trial))
    start_time = time.monotonic()

    sys.stdout = Logger(osp.join(args.logs_dir, str(trial)+'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset_ir = get_data('regdb_ir', args.data_dir,trial=trial)
    dataset_rgb = get_data('regdb_rgb', args.data_dir,trial=trial)

    test_loader_ir = get_test_loader(dataset_ir, args.height, args.width, args.batch_size, args.workers)
    test_loader_rgb = get_test_loader(dataset_rgb, args.height, args.width, args.batch_size, args.workers)
    # Create model
    model = create_model(args)

    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    # Trainer
    trainer = ClusterContrastTrainer(model)

    for epoch in range(args.epochs):
        with torch.no_grad():
            if epoch == 0:
                # KMeans cluster
                ir_num_clusters = args.ir_num_cluster + args.over_cluster
                print('IR Clustering criterion: num_clusters: {:d}'.format(ir_num_clusters))
                rgb_num_clusters = args.rgb_num_cluster + args.over_cluster
                print('RGB Clustering criterion: num_clusters: {:d}'.format(rgb_num_clusters))

                hierarchical_clustering_ir = AgglomerativeClustering(n_clusters=ir_num_clusters)
                hierarchical_clustering_rgb = AgglomerativeClustering(n_clusters=rgb_num_clusters)

            print('==> Create pseudo labels for unlabeled RGB data')
            cluster_loader_rgb = get_test_loader(dataset_rgb, args.height, args.width,
                                             args.batch_size, args.workers, 
                                             testset=sorted(dataset_rgb.train))
            features_rgb, _ = extract_features(model, cluster_loader_rgb, print_freq=50, mode=1)
            del cluster_loader_rgb
            features_rgb = torch.cat([features_rgb[f].unsqueeze(0) for f, _, _ in sorted(dataset_rgb.train)], 0)

            print('==> Create pseudo labels for unlabeled IR data')
            cluster_loader_ir = get_test_loader(dataset_ir, args.height, args.width,
                                             args.batch_size, args.workers, 
                                             testset=sorted(dataset_ir.train))
            features_ir, _ = extract_features(model, cluster_loader_ir, print_freq=50, mode=2)
            del cluster_loader_ir
            features_ir = torch.cat([features_ir[f].unsqueeze(0) for f, _, _ in sorted(dataset_ir.train)], 0)            

            ## administrative cluster for ir
            rerank_dist_ir = compute_jaccard_distance(features_ir, k1=args.k1, k2=args.k2, search_option=3)
            hierarchical_labels_ir = hierarchical_clustering_ir.fit_predict(rerank_dist_ir)
            initial_centers_ir = []
            for label in np.unique(hierarchical_labels_ir):
                cluster_points = rerank_dist_ir[hierarchical_labels_ir == label]
                cluster_center = np.mean(cluster_points, axis=0)
                initial_centers_ir.append(cluster_center)
            initial_centers_ir = np.array(initial_centers_ir)
            ## administrative cluster for rgb
            rerank_dist_rgb = compute_jaccard_distance(features_rgb, k1=args.k1, k2=args.k2, search_option=3)
            hierarchical_labels_rgb = hierarchical_clustering_rgb.fit_predict(rerank_dist_rgb)
            initial_centers_rgb = []
            for label in np.unique(hierarchical_labels_rgb):
                cluster_points = rerank_dist_rgb[hierarchical_labels_rgb == label]
                cluster_center = np.mean(cluster_points, axis=0)
                initial_centers_rgb.append(cluster_center)
            initial_centers_rgb = np.array(initial_centers_rgb)

            ## k-means cluster for ir
            cluster_ir = KMeans(n_clusters=ir_num_clusters, init=initial_centers_ir, n_init=5)
            cluster_ir.fit(rerank_dist_ir)
            pseudo_labels_ir, cluster_centers_ir = cluster_ir.labels_, cluster_ir.cluster_centers_
            ## k-means cluster for rgb
            cluster_rgb = KMeans(n_clusters=ir_num_clusters, init=initial_centers_rgb, n_init=5)
            cluster_rgb.fit(rerank_dist_rgb)
            pseudo_labels_rgb, cluster_centers_rgb = cluster_rgb.labels_, cluster_rgb.cluster_centers_

            num_cluster_ir, num_cluster_rgb = len(set(pseudo_labels_ir)), len(set(pseudo_labels_rgb))
            print("num_cluster_ir: ", num_cluster_ir, "  num_cluster_rgb: ", num_cluster_rgb)
            num_cluster_ir, num_cluster_rgb = ir_num_clusters, rgb_num_clusters

            del rerank_dist_ir
            del rerank_dist_rgb

        # generate new dataset and calculate cluster centers
        @torch.no_grad()
        def generate_cluster_features(labels, features):
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])

            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]

            centers = torch.stack(centers, dim=0)
            return centers

        cluster_features_ir = generate_cluster_features(pseudo_labels_ir, features_ir) 
        cluster_features_rgb = generate_cluster_features(pseudo_labels_rgb, features_rgb)
        memory_ir = ClusterMemory(model.module.num_features, num_cluster_ir, temp=args.temp,
                               momentum=args.momentum, use_hard=args.use_hard).cuda()
        memory_rgb = ClusterMemory(model.module.num_features, num_cluster_rgb, temp=args.temp,
                               momentum=args.momentum, use_hard=args.use_hard).cuda()
        memory_ir.features = F.normalize(cluster_features_ir, dim=1).cuda()
        memory_rgb.features = F.normalize(cluster_features_rgb, dim=1).cuda()

        trainer.memory_ir = memory_ir
        trainer.memory_rgb = memory_rgb

        pseudo_labeled_dataset_ir = []
        ir_label=[]
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_ir.train), pseudo_labels_ir)):
            if label != -1:
                pseudo_labeled_dataset_ir.append((fname, label.item(), cid))
                ir_label.append(label.item())
        print('==> Statistics for IR epoch {}: {} clusters'.format(epoch, num_cluster_ir))

        pseudo_labeled_dataset_rgb = []
        rgb_label=[]
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_rgb.train), pseudo_labels_rgb)):
            if label != -1:
                pseudo_labeled_dataset_rgb.append((fname, label.item(), cid))
                rgb_label.append(label.item())
        print('==> Statistics for RGB epoch {}: {} clusters'.format(epoch, num_cluster_rgb))

        # ######################## 
        # i2r, r2i, R = {}, {}, []
        # bgm = False
        
        # # clusternorm
        # cluster_features_rgb, cluster_features_ir = \
        #     F.normalize(cluster_features_rgb, dim=1), F.normalize(cluster_features_ir, dim=1)
        # print("cluster_features_rgb: ", cluster_features_rgb.shape)
        # print("cluster_features_ir: ", cluster_features_ir.shape)

        # # [-1, 1] torch.mm(cluster_features_rgb, cluster_features_ir.T) #CostMatrix
        # similarity = 1 - (torch.mm(cluster_features_rgb, cluster_features_ir.T))/1 #.exp().cpu()
        # dis_similarity = similarity.exp().cpu()
        # # dis_similarity = (1 / (similarity))
        # ## dis_similarity = - similarity # binus
        # cost = dis_similarity / 1
        # tmp = torch.zeros(dis_similarity.shape[0], dis_similarity.shape[0] - dis_similarity.shape[1])
        # cost = (torch.cat((cost, tmp), 1))
        # unmatched_row = []
        # row_ind, col_ind = linear_sum_assignment(cost)
        # for idx, item in enumerate(row_ind):
        #     if col_ind[idx] < similarity.shape[1]:
        #         R.append((row_ind[idx], col_ind[idx]))
        #         r2i[row_ind[idx]] = col_ind[idx]
        #         i2r[col_ind[idx]] = row_ind[idx]
        #     else:
        #         unmatched_row.append(row_ind[idx])
        # if bgm is False:
        #     unmatched_cost = cost[unmatched_row][:,:dis_similarity.shape[1]]
        #     unmatched_row_ind, unmatched_col_ind = linear_sum_assignment(unmatched_cost)
        #     for idx, item in enumerate(unmatched_row_ind):
        #         R.append((unmatched_row[idx], unmatched_col_ind[idx]))
        #         r2i[unmatched_row[idx]] = unmatched_col_ind[idx]
        # del cluster_features_ir, cluster_features_rgb

        # ########################
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        height = args.height
        width = args.width
        train_transformer_rgb = T.Compose([
            T.Resize((height, width), interpolation=3),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            normalizer,
            ChannelRandomErasing(probability = 0.5)
        ])
        
        train_transformer_rgb1 = T.Compose([
            T.Resize((height, width), interpolation=3),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            T.ToTensor(),
            normalizer,
            ChannelRandomErasing(probability = 0.5),
            ChannelExchange(gray = 2)
        ])

        transform_thermal = T.Compose([
            T.Resize((height, width), interpolation=3),
            T.Pad(10),
            T.RandomCrop((height, width)),  # (288, 144)
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalizer,
            ChannelRandomErasing(probability = 0.5),
            ChannelAdapGray(probability =0.5)])

        train_loader_ir = get_train_loader_ir(args, dataset_ir, args.height, args.width,
                                        args.batch_size, args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset_ir, 
                                        no_cam=args.no_cam,
                                        train_transformer=transform_thermal)

        train_loader_rgb = get_train_loader_color(args, dataset_rgb, args.height, args.width,
                                        128, args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset_rgb, 
                                        no_cam=args.no_cam,
                                        train_transformer=train_transformer_rgb, train_transformer1=train_transformer_rgb1)

        train_loader_ir.new_epoch()
        train_loader_rgb.new_epoch()

        trainer.train(epoch, train_loader_ir, train_loader_rgb, optimizer,
                      print_freq=args.print_freq, train_iters=len(train_loader_ir))
        
        if epoch>=1 and ( (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1)):
##############################
            args.test_batch=64
            args.img_w=args.width
            args.img_h=args.height
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            transform_test = T.Compose([
                T.ToPILImage(),
                T.Resize((args.img_h,args.img_w)),
                T.ToTensor(),
                normalize,
            ])

            ## ############## visible2thermal
            mode='visible2thermal'
            data_path='/home/liyongxiang/code/ReID/dataset/RegDB/'

            if mode == 'visible2thermal':
                query_img, query_label = process_test_regdb(data_path, trial=trial, modal='visible')
                gall_img, gall_label = process_test_regdb(data_path, trial=trial, modal='thermal')
            elif mode == 'thermal2visible':
                query_img, query_label = process_test_regdb(data_path, trial=trial, modal='thermal')
                gall_img, gall_label = process_test_regdb(data_path, trial=trial, modal='visible')

            gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
            gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
            nquery = len(query_label)
            ngall = len(gall_label)
            queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
            query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
            query_feat_fc = extract_query_feat(model,query_loader,nquery)
            # for trial in range(1):
            ngall = len(gall_label)
            gall_feat_fc = extract_gall_feat(model, gall_loader, ngall)
            # fc feature
            distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
            v2t_cmc, v2t_mAP, v2t_mINP = eval_regdb(-distmat, query_label, gall_label)

            print('Test {} Trial: {}'.format(mode, trial))
            print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                    v2t_cmc[0], v2t_cmc[4], v2t_cmc[9], v2t_cmc[19], v2t_mAP, v2t_mINP))
            
            ## ############## thermal2visible
            mode='thermal2visible'
            data_path='/home/liyongxiang/code/ReID/dataset/RegDB/'

            if mode == 'visible2thermal':
                query_img, query_label = process_test_regdb(data_path, trial=trial, modal='visible')
                gall_img, gall_label = process_test_regdb(data_path, trial=trial, modal='thermal')
            elif mode == 'thermal2visible':
                query_img, query_label = process_test_regdb(data_path, trial=trial, modal='thermal')
                gall_img, gall_label = process_test_regdb(data_path, trial=trial, modal='visible')

            gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
            gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
            nquery = len(query_label)
            ngall = len(gall_label)
            queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
            query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
            query_feat_fc = extract_query_feat(model,query_loader,nquery)
            # for trial in range(1):
            ngall = len(gall_label)
            gall_feat_fc = extract_gall_feat(model, gall_loader, ngall)
            # fc feature
            distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
            t2v_cmc, t2v_mAP, t2v_mINP = eval_regdb(-distmat, query_label, gall_label)

            print('Test {} Trial: {}'.format(mode, trial))
            print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                    t2v_cmc[0], t2v_cmc[4], t2v_cmc[9], t2v_cmc[19], t2v_mAP, t2v_mINP))

            is_best = ((v2t_mAP+t2v_mAP)/2 > best_mAP)
            best_mAP = max((v2t_mAP+t2v_mAP)/2, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d}  v2t model mAP: {:5.1%}  t2v model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, v2t_mAP, t2v_mAP, best_mAP, ' *' if is_best else ''))
############################
        lr_scheduler.step()
    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))

def main_worker_stage2_robust_loss(args, log_s1_name, log_s2_name):
    # logs_dir_root = osp.join('logs/'+log_s2_name)
    trial = args.trial
    stage1_logs_dir = osp.join(args.logs_dir+'/'+log_s1_name, str(trial))
    # global start_epoch, best_mAP
    start_epoch =0
    best_mAP =0
    stage2_logs_dir = osp.join(args.logs_dir+'/'+log_s2_name, str(trial))
    # args.logs_dir = osp.join(logs_dir_root,str(trial))
    start_time = time.monotonic()

    # cudnn.benchmark = True

    sys.stdout = Logger(osp.join(stage2_logs_dir, str(trial)+'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset_ir = get_data('regdb_ir', args.data_dir, trial=trial)
    dataset_rgb = get_data('regdb_rgb', args.data_dir, trial=trial)

    # Create model
    model = create_model(args)
    checkpoint = load_checkpoint(osp.join(stage1_logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    print("==> Load last checkpoint successfully! ", osp.join(stage1_logs_dir, 'model_best.pth.tar'))
    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    # Trainer
    trainer = ClusterContrastTrainer(model)

    for epoch in range(args.epochs):
        with torch.no_grad():
            if epoch == 0:
                # KMeans cluster
                ir_num_clusters = args.ir_num_cluster + args.over_cluster
                print('IR Clustering criterion: num_clusters: {:d}'.format(ir_num_clusters))
                rgb_num_clusters = args.rgb_num_cluster + args.over_cluster
                print('RGB Clustering criterion: num_clusters: {:d}'.format(rgb_num_clusters))

                hierarchical_clustering_ir = AgglomerativeClustering(n_clusters=ir_num_clusters)
                hierarchical_clustering_rgb = AgglomerativeClustering(n_clusters=rgb_num_clusters)

            print('==> Create pseudo labels for unlabeled RGB data')
            cluster_loader_rgb = get_test_loader(dataset_rgb, args.height, args.width,
                                             args.batch_size, args.workers, 
                                             testset=sorted(dataset_rgb.train))
            features_rgb, _ = extract_features(model, cluster_loader_rgb, print_freq=50, mode=1)
            del cluster_loader_rgb
            features_rgb = torch.cat([features_rgb[f].unsqueeze(0) for f, _, _ in sorted(dataset_rgb.train)], 0)

            print('==> Create pseudo labels for unlabeled IR data')
            cluster_loader_ir = get_test_loader(dataset_ir, args.height, args.width,
                                             args.batch_size, args.workers, 
                                             testset=sorted(dataset_ir.train))
            features_ir, _ = extract_features(model, cluster_loader_ir, print_freq=50, mode=2)
            del cluster_loader_ir
            features_ir = torch.cat([features_ir[f].unsqueeze(0) for f, _, _ in sorted(dataset_ir.train)], 0)            

            ## administrative cluster for ir
            rerank_dist_ir = compute_jaccard_distance(features_ir, k1=args.k1, k2=args.k2, search_option=3)
            hierarchical_labels_ir = hierarchical_clustering_ir.fit_predict(rerank_dist_ir)
            initial_centers_ir = []
            for label in np.unique(hierarchical_labels_ir):
                cluster_points = rerank_dist_ir[hierarchical_labels_ir == label]
                cluster_center = np.mean(cluster_points, axis=0)
                initial_centers_ir.append(cluster_center)
            initial_centers_ir = np.array(initial_centers_ir)
            ## administrative cluster for rgb
            rerank_dist_rgb = compute_jaccard_distance(features_rgb, k1=args.k1, k2=args.k2, search_option=3)
            hierarchical_labels_rgb = hierarchical_clustering_rgb.fit_predict(rerank_dist_rgb)
            initial_centers_rgb = []
            for label in np.unique(hierarchical_labels_rgb):
                cluster_points = rerank_dist_rgb[hierarchical_labels_rgb == label]
                cluster_center = np.mean(cluster_points, axis=0)
                initial_centers_rgb.append(cluster_center)
            initial_centers_rgb = np.array(initial_centers_rgb)

            ## k-means cluster for ir
            cluster_ir = KMeans(n_clusters=ir_num_clusters, init=initial_centers_ir, n_init=5)
            cluster_ir.fit(rerank_dist_ir)
            pseudo_labels_ir, cluster_centers_ir = cluster_ir.labels_, cluster_ir.cluster_centers_
            ## k-means cluster for rgb
            cluster_rgb = KMeans(n_clusters=ir_num_clusters, init=initial_centers_rgb, n_init=5)
            cluster_rgb.fit(rerank_dist_rgb)
            pseudo_labels_rgb, cluster_centers_rgb = cluster_rgb.labels_, cluster_rgb.cluster_centers_

            num_cluster_ir, num_cluster_rgb = len(set(pseudo_labels_ir)), len(set(pseudo_labels_rgb))
            print("num_cluster_ir: ", num_cluster_ir, "  num_cluster_rgb: ", num_cluster_rgb)
            num_cluster_ir, num_cluster_rgb = ir_num_clusters, rgb_num_clusters

            del rerank_dist_ir
            del rerank_dist_rgb

        # generate new dataset and calculate cluster centers
        @torch.no_grad()
        def generate_cluster_features(labels, features):
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])

            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]

            centers = torch.stack(centers, dim=0)
            return centers

        cluster_features_ir = generate_cluster_features(pseudo_labels_ir, features_ir) 
        cluster_features_rgb = generate_cluster_features(pseudo_labels_rgb, features_rgb)
        memory_ir = ClusterMemory(model.module.num_features, num_cluster_ir, temp=args.temp,
                               momentum=args.momentum, use_hard=args.use_hard).cuda()
        memory_rgb = ClusterMemory(model.module.num_features, num_cluster_rgb, temp=args.temp,
                               momentum=args.momentum, use_hard=args.use_hard).cuda()
        memory_ir.features = F.normalize(cluster_features_ir, dim=1).cuda()
        memory_rgb.features = F.normalize(cluster_features_rgb, dim=1).cuda()


        trainer.memory_ir = memory_ir
        trainer.memory_rgb = memory_rgb

        pseudo_labeled_dataset_ir = []
        ir_label=[]
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_ir.train), pseudo_labels_ir)):
            if label != -1:
                pseudo_labeled_dataset_ir.append((fname, label.item(), cid))
                ir_label.append(label.item())
        print('==> Statistics for IR epoch {}: {} clusters'.format(epoch, num_cluster_ir))

        pseudo_labeled_dataset_rgb = []
        rgb_label=[]
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_rgb.train), pseudo_labels_rgb)):
            if label != -1:
                pseudo_labeled_dataset_rgb.append((fname, label.item(), cid))
                rgb_label.append(label.item())

        print('==> Statistics for RGB epoch {}: {} clusters'.format(epoch, num_cluster_rgb))

        ## ######################## PGM
        print("Progressive Graph Matching")
        i2r, r2i, R = {}, {}, []
        bgm = False
        
        # clusternorm
        cluster_features_rgb, cluster_features_ir = \
            F.normalize(cluster_features_rgb, dim=1), F.normalize(cluster_features_ir, dim=1)
        print("cluster_features_rgb: ", cluster_features_rgb.shape)
        print("cluster_features_ir: ", cluster_features_ir.shape)

        # [-1, 1] torch.mm(cluster_features_rgb, cluster_features_ir.T) #CostMatrix
        similarity = 1 - (torch.mm(cluster_features_rgb, cluster_features_ir.T))/1 #.exp().cpu()
        dis_similarity = similarity.exp().cpu()
        cost = dis_similarity / 1
        tmp = torch.zeros(dis_similarity.shape[0], dis_similarity.shape[0] - dis_similarity.shape[1])
        cost = (torch.cat((cost, tmp), 1))
        unmatched_row = []
        row_ind, col_ind = linear_sum_assignment(cost)
        for idx, item in enumerate(row_ind):
            if col_ind[idx] < similarity.shape[1]:
                R.append((row_ind[idx], col_ind[idx]))
                r2i[row_ind[idx]] = col_ind[idx]
                i2r[col_ind[idx]] = row_ind[idx]
            else:
                unmatched_row.append(row_ind[idx])
        if bgm is False:
            unmatched_cost = cost[unmatched_row][:,:dis_similarity.shape[1]]
            unmatched_row_ind, unmatched_col_ind = linear_sum_assignment(unmatched_cost)
            for idx, item in enumerate(unmatched_row_ind):
                R.append((unmatched_row[idx], unmatched_col_ind[idx]))
                r2i[unmatched_row[idx]] = unmatched_col_ind[idx]
        del cluster_features_ir, cluster_features_rgb
        print("Progressive Graph Matching Done")


        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        height=args.height
        width=args.width
        train_transformer_rgb = T.Compose([
            T.Resize((height, width), interpolation=3),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            normalizer,
            ChannelRandomErasing(probability = 0.5)
        ])
        
        train_transformer_rgb1 = T.Compose([
            T.Resize((height, width), interpolation=3),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.5),
            T.ToTensor(),
            normalizer,
            ChannelRandomErasing(probability = 0.5),
            ChannelExchange(gray = 2)
        ])

        transform_thermal = T.Compose( [
            T.Resize((height, width), interpolation=3),
            T.Pad(10),
            T.RandomCrop((288, 144)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalizer,
            ChannelRandomErasing(probability = 0.5),
            ChannelAdapGray(probability =0.5)])

        train_loader_ir = get_train_loader_ir(args, dataset_ir, args.height, args.width,
                                        args.batch_size, args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset_ir, no_cam=args.no_cam,train_transformer=transform_thermal)

        train_loader_rgb = get_train_loader_color(args, dataset_rgb, args.height, args.width,
                                        128, args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset_rgb, no_cam=args.no_cam,train_transformer=train_transformer_rgb,train_transformer1=train_transformer_rgb1)

        train_loader_ir.new_epoch()
        train_loader_rgb.new_epoch()

        trainer.train(epoch, train_loader_ir,train_loader_rgb, optimizer,
                      print_freq=args.print_freq, train_iters=len(train_loader_ir), i2r=i2r, r2i=r2i)

        if epoch>=1 and ( (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1)):
##############################
            args.test_batch=64
            args.img_w=args.width
            args.img_h=args.height
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            transform_test = T.Compose([
                T.ToPILImage(),
                T.Resize((args.img_h,args.img_w)),
                T.ToTensor(),
                normalize,
            ])

            ## ############## visible2thermal
            mode='visible2thermal'
            data_path='/home/liyongxiang/code/ReID/dataset/RegDB/'

            if mode == 'visible2thermal':
                query_img, query_label = process_test_regdb(data_path, trial=trial, modal='visible')
                gall_img, gall_label = process_test_regdb(data_path, trial=trial, modal='thermal')
            elif mode == 'thermal2visible':
                query_img, query_label = process_test_regdb(data_path, trial=trial, modal='thermal')
                gall_img, gall_label = process_test_regdb(data_path, trial=trial, modal='visible')

            gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
            gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
            nquery = len(query_label)
            ngall = len(gall_label)
            queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
            query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
            query_feat_fc = extract_query_feat(model,query_loader,nquery)
            # for trial in range(1):
            ngall = len(gall_label)
            gall_feat_fc = extract_gall_feat(model, gall_loader, ngall)
            # fc feature
            distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
            v2t_cmc, v2t_mAP, v2t_mINP = eval_regdb(-distmat, query_label, gall_label)

            print('Test {} Trial: {}'.format(mode, trial))
            print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                    v2t_cmc[0], v2t_cmc[4], v2t_cmc[9], v2t_cmc[19], v2t_mAP, v2t_mINP))
            
            ## ############## thermal2visible
            mode='thermal2visible'
            data_path='/home/liyongxiang/code/ReID/dataset/RegDB/'

            if mode == 'visible2thermal':
                query_img, query_label = process_test_regdb(data_path, trial=trial, modal='visible')
                gall_img, gall_label = process_test_regdb(data_path, trial=trial, modal='thermal')
            elif mode == 'thermal2visible':
                query_img, query_label = process_test_regdb(data_path, trial=trial, modal='thermal')
                gall_img, gall_label = process_test_regdb(data_path, trial=trial, modal='visible')

            gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
            gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
            nquery = len(query_label)
            ngall = len(gall_label)
            queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
            query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
            query_feat_fc = extract_query_feat(model,query_loader,nquery)
            # for trial in range(1):
            ngall = len(gall_label)
            gall_feat_fc = extract_gall_feat(model, gall_loader, ngall)
            # fc feature
            distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
            t2v_cmc, t2v_mAP, t2v_mINP = eval_regdb(-distmat, query_label, gall_label)

            print('Test {} Trial: {}'.format(mode, trial))
            print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                    t2v_cmc[0], t2v_cmc[4], t2v_cmc[9], t2v_cmc[19], t2v_mAP, t2v_mINP))

            is_best = ((v2t_mAP+t2v_mAP)/2 > best_mAP)
            best_mAP = max((v2t_mAP+t2v_mAP)/2, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d}  v2t model mAP: {:5.1%}  t2v model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, v2t_mAP, t2v_mAP, best_mAP, ' *' if is_best else ''))
############################
        lr_scheduler.step()
    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))

def main_worker_stage1_colearning(args, log_s1_name):
    logs_dir_root = osp.join(args.logs_dir+'/'+log_s1_name)
    trial = args.trial
    # global start_epoch, best_mAP
    start_epoch =0
    best_mAP =0
    args.logs_dir = osp.join(logs_dir_root,str(trial))
    start_time = time.monotonic()

    sys.stdout = Logger(osp.join(args.logs_dir, str(trial)+'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset_ir = get_data('regdb_ir', args.data_dir,trial=trial)
    dataset_rgb = get_data('regdb_rgb', args.data_dir,trial=trial)

    test_loader_ir = get_test_loader(dataset_ir, args.height, args.width, args.batch_size, args.workers)
    test_loader_rgb = get_test_loader(dataset_rgb, args.height, args.width, args.batch_size, args.workers)

    # Create model
    model_list = []
    model_list.append(create_model(args))
    model_list.append(create_model(args))

    # Optimizer
    params_list = []
    for ii in range(len(model_list)):
        params_list.append([{"params": [value]} for _, value in model_list[ii].named_parameters() if value.requires_grad])
    
    optimizer_list = []
    for ii in range(len(model_list)):
        optimizer_list.append(torch.optim.Adam(params_list[ii], lr=args.lr_list[ii], weight_decay=args.weight_decay))

    lr_scheduler_list = []
    for ii in range(len(model_list)):
        lr_scheduler_list.append(torch.optim.lr_scheduler.StepLR(optimizer_list[ii], step_size=args.step_size, gamma=0.1))

    # Trainer
    trainer_list = []
    for ii in range(len(model_list)):
        trainer_list.append(ClusterContrastTrainer(model_list[ii]))

    # features and pseudo-labels
    cluster_features_ir_list, cluster_features_rgb_list = [], []
    pseudo_labeled_dataset_ir_list, pseudo_labeled_dataset_rgb_list = [], []
    for ii in range(len(model_list)):
        cluster_features_ir_list.append(0)
        cluster_features_rgb_list.append(0)
        pseudo_labeled_dataset_ir_list.append(0)
        pseudo_labeled_dataset_rgb_list.append(0)

    is_best, best_mAP = [], []
    for ii in range(len(model_list)):
        is_best.append(0.0)
        best_mAP.append(0.0)

    for epoch in range(args.epochs):
        # 在这里 测试每个样本的概率


        for ii in range(len(model_list)):
            with torch.no_grad():
                if epoch == 0:
                    # KMeans cluster
                    ir_num_clusters = args.ir_num_cluster + args.over_cluster
                    print('IR Clustering criterion: num_clusters: {:d}'.format(ir_num_clusters))
                    rgb_num_clusters = args.rgb_num_cluster + args.over_cluster
                    print('RGB Clustering criterion: num_clusters: {:d}'.format(rgb_num_clusters))

                    hierarchical_clustering_ir = AgglomerativeClustering(n_clusters=ir_num_clusters)
                    hierarchical_clustering_rgb = AgglomerativeClustering(n_clusters=rgb_num_clusters)

                print('==> Create pseudo labels for unlabeled RGB data')
                cluster_loader_rgb = get_test_loader(dataset_rgb, args.height, args.width,
                                                args.batch_size, args.workers, 
                                                testset=sorted(dataset_rgb.train))
                features_rgb, _ = extract_features(model_list[ii], cluster_loader_rgb, print_freq=50, mode=1)
                del cluster_loader_rgb
                features_rgb = torch.cat([features_rgb[f].unsqueeze(0) for f, _, _ in sorted(dataset_rgb.train)], 0)

                print('==> Create pseudo labels for unlabeled IR data')
                cluster_loader_ir = get_test_loader(dataset_ir, args.height, args.width,
                                                args.batch_size, args.workers, 
                                                testset=sorted(dataset_ir.train))
                features_ir, _ = extract_features(model_list[ii], cluster_loader_ir, print_freq=50, mode=2)
                del cluster_loader_ir
                features_ir = torch.cat([features_ir[f].unsqueeze(0) for f, _, _ in sorted(dataset_ir.train)], 0)            

                ## administrative cluster for ir
                rerank_dist_ir = compute_jaccard_distance(features_ir, k1=args.k1, k2=args.k2, search_option=3)
                hierarchical_labels_ir = hierarchical_clustering_ir.fit_predict(rerank_dist_ir)
                initial_centers_ir = []
                for label in np.unique(hierarchical_labels_ir):
                    cluster_points = rerank_dist_ir[hierarchical_labels_ir == label]
                    cluster_center = np.mean(cluster_points, axis=0)
                    initial_centers_ir.append(cluster_center)
                initial_centers_ir = np.array(initial_centers_ir)
                ## administrative cluster for rgb
                rerank_dist_rgb = compute_jaccard_distance(features_rgb, k1=args.k1, k2=args.k2, search_option=3)
                hierarchical_labels_rgb = hierarchical_clustering_rgb.fit_predict(rerank_dist_rgb)
                initial_centers_rgb = []
                for label in np.unique(hierarchical_labels_rgb):
                    cluster_points = rerank_dist_rgb[hierarchical_labels_rgb == label]
                    cluster_center = np.mean(cluster_points, axis=0)
                    initial_centers_rgb.append(cluster_center)
                initial_centers_rgb = np.array(initial_centers_rgb)

                ## k-means cluster for ir
                cluster_ir = KMeans(n_clusters=ir_num_clusters, init=initial_centers_ir, n_init=5)
                cluster_ir.fit(rerank_dist_ir)
                pseudo_labels_ir, cluster_centers_ir = cluster_ir.labels_, cluster_ir.cluster_centers_
                ## k-means cluster for rgb
                cluster_rgb = KMeans(n_clusters=ir_num_clusters, init=initial_centers_rgb, n_init=5)
                cluster_rgb.fit(rerank_dist_rgb)
                pseudo_labels_rgb, cluster_centers_rgb = cluster_rgb.labels_, cluster_rgb.cluster_centers_

                num_cluster_ir, num_cluster_rgb = len(set(pseudo_labels_ir)), len(set(pseudo_labels_rgb))
                print("num_cluster_ir: ", num_cluster_ir, "  num_cluster_rgb: ", num_cluster_rgb)
                num_cluster_ir, num_cluster_rgb = ir_num_clusters, rgb_num_clusters

                del rerank_dist_ir
                del rerank_dist_rgb

            # generate new dataset and calculate cluster centers
            @torch.no_grad()
            def generate_cluster_features(labels, features):
                centers = collections.defaultdict(list)
                for i, label in enumerate(labels):
                    if label == -1:
                        continue
                    centers[labels[i]].append(features[i])

                centers = [
                    torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
                ]

                centers = torch.stack(centers, dim=0)
                return centers

            cluster_features_ir_list[ii] = generate_cluster_features(pseudo_labels_ir, features_ir)
            cluster_features_rgb_list[ii] = generate_cluster_features(pseudo_labels_rgb, features_rgb)
            memory_ir = ClusterMemory(model_list[ii].module.num_features, num_cluster_ir, temp=args.temp,
                                momentum=args.momentum, use_hard=args.use_hard).cuda()
            memory_rgb = ClusterMemory(model_list[ii].module.num_features, num_cluster_rgb, temp=args.temp,
                                momentum=args.momentum, use_hard=args.use_hard).cuda()
            memory_ir.features = F.normalize(cluster_features_ir_list[ii], dim=1).cuda()
            memory_rgb.features = F.normalize(cluster_features_rgb_list[ii], dim=1).cuda()

            trainer_list[ii].memory_ir = memory_ir
            trainer_list[ii].memory_rgb = memory_rgb

            pseudo_labeled_dataset_ir_list[ii] = []
            ir_label=[]
            for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_ir.train), pseudo_labels_ir)):
                if label != -1:
                    pseudo_labeled_dataset_ir_list[ii].append((fname, label.item(), cid))
                    ir_label.append(label.item())
            print('==> Statistics for IR epoch {}: {} clusters'.format(epoch, num_cluster_ir))

            pseudo_labeled_dataset_rgb_list[ii] = []
            rgb_label=[]
            for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_rgb.train), pseudo_labels_rgb)):
                if label != -1:
                    pseudo_labeled_dataset_rgb_list[ii].append((fname, label.item(), cid))
                    rgb_label.append(label.item())
            print('==> Statistics for RGB epoch {}: {} clusters'.format(epoch, num_cluster_rgb))

            normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            height = args.height
            width = args.width
            train_transformer_rgb = T.Compose([
                T.Resize((height, width), interpolation=3),
                T.Pad(10),
                T.RandomCrop((height, width)),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                normalizer,
                ChannelRandomErasing(probability = 0.5)
            ])
            
            train_transformer_rgb1 = T.Compose([
                T.Resize((height, width), interpolation=3),
                T.Pad(10),
                T.RandomCrop((height, width)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                T.ToTensor(),
                normalizer,
                ChannelRandomErasing(probability = 0.5),
                ChannelExchange(gray = 2)
            ])

            transform_thermal = T.Compose([
                T.Resize((height, width), interpolation=3),
                T.Pad(10),
                T.RandomCrop((height, width)),  # (288, 144)
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalizer,
                ChannelRandomErasing(probability = 0.5),
                ChannelAdapGray(probability =0.5)])

            train_loader_ir = get_train_loader_ir(args, dataset_ir, args.height, args.width,
                                            args.batch_size, args.workers, args.num_instances, iters,
                                            trainset=pseudo_labeled_dataset_ir_list[ii], 
                                            no_cam=args.no_cam,
                                            train_transformer=transform_thermal)

            train_loader_rgb = get_train_loader_color(args, dataset_rgb, args.height, args.width,
                                            128, args.workers, args.num_instances, iters,
                                            trainset=pseudo_labeled_dataset_rgb_list[ii], 
                                            no_cam=args.no_cam,
                                            train_transformer=train_transformer_rgb, train_transformer1=train_transformer_rgb1)

            train_loader_ir.new_epoch()
            train_loader_rgb.new_epoch()

            trainer_list[ii].train(epoch, train_loader_ir, train_loader_rgb, optimizer_list[ii],
                        print_freq=args.print_freq, train_iters=len(train_loader_ir))
            
            if epoch>=0 and ( (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1)):
##############################
                args.test_batch=64
                args.img_w=args.width
                args.img_h=args.height
                normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                transform_test = T.Compose([
                    T.ToPILImage(),
                    T.Resize((args.img_h,args.img_w)),
                    T.ToTensor(),
                    normalize,
                ])

                ## ############## visible2thermal
                mode='visible2thermal'
                data_path='/home/liyongxiang/code/ReID/dataset/RegDB/'

                if mode == 'visible2thermal':
                    query_img, query_label = process_test_regdb(data_path, trial=trial, modal='visible')
                    gall_img, gall_label = process_test_regdb(data_path, trial=trial, modal='thermal')
                elif mode == 'thermal2visible':
                    query_img, query_label = process_test_regdb(data_path, trial=trial, modal='thermal')
                    gall_img, gall_label = process_test_regdb(data_path, trial=trial, modal='visible')

                gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
                gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
                nquery = len(query_label)
                ngall = len(gall_label)
                queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
                query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
                query_feat_fc = extract_query_feat(model_list[ii],query_loader,nquery)
                # for trial in range(1):
                ngall = len(gall_label)
                gall_feat_fc = extract_gall_feat(model_list[ii], gall_loader, ngall)
                # fc feature
                distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
                v2t_cmc, v2t_mAP, v2t_mINP = eval_regdb(-distmat, query_label, gall_label)

                print('Model {} Test {} Trial: {}'.format(str(ii), mode, trial))
                print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                        v2t_cmc[0], v2t_cmc[4], v2t_cmc[9], v2t_cmc[19], v2t_mAP, v2t_mINP))
                
                ## ############## thermal2visible
                mode='thermal2visible'
                data_path='/home/liyongxiang/code/ReID/dataset/RegDB/'

                if mode == 'visible2thermal':
                    query_img, query_label = process_test_regdb(data_path, trial=trial, modal='visible')
                    gall_img, gall_label = process_test_regdb(data_path, trial=trial, modal='thermal')
                elif mode == 'thermal2visible':
                    query_img, query_label = process_test_regdb(data_path, trial=trial, modal='thermal')
                    gall_img, gall_label = process_test_regdb(data_path, trial=trial, modal='visible')

                gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
                gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
                nquery = len(query_label)
                ngall = len(gall_label)
                queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
                query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
                query_feat_fc = extract_query_feat(model_list[ii], query_loader,nquery)
                # for trial in range(1):
                ngall = len(gall_label)
                gall_feat_fc = extract_gall_feat(model_list[ii], gall_loader, ngall)
                # fc feature
                distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
                t2v_cmc, t2v_mAP, t2v_mINP = eval_regdb(-distmat, query_label, gall_label)

                print('Model {} Test {} Trial: {}'.format(str(ii), mode, trial))
                print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                        t2v_cmc[0], t2v_cmc[4], t2v_cmc[9], t2v_cmc[19], t2v_mAP, t2v_mINP))


                is_best[ii] = ((v2t_mAP+t2v_mAP)/2 > best_mAP[ii])
                best_mAP[ii] = max((v2t_mAP+t2v_mAP)/2, best_mAP[ii])

                print("checkpoint: ", osp.join(args.logs_dir, 'checkpoint'+str(ii)+'.pth.tar'))
                print("model_best: ", osp.join(args.logs_dir, 'model_best_'+str(ii)+'.pth.tar'))
                
                save_checkpoint({
                    'state_dict': model_list[ii].state_dict(),
                    'epoch': epoch + 1,
                    'best_mAP': best_mAP[ii],
                }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint'+str(ii)+'.pth.tar'), model_best_path=osp.join(args.logs_dir, 'model_best_'+str(ii)+'.pth.tar'))

                print('\n * Finished epoch {:3d}  v2t model mAP: {:5.1%}  t2v model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                    format(epoch, v2t_mAP, t2v_mAP, best_mAP[ii], ' *' if is_best[ii] else ''))
    ############################
            lr_scheduler_list[ii].step()
    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))

def main_worker_stage2_colearning(args, log_s1_name, log_s2_name):
    # logs_dir_root = osp.join('logs/'+log_s2_name)
    trial = args.trial
    stage1_logs_dir = osp.join(args.logs_dir+'/'+log_s1_name, str(trial))
    # global start_epoch, best_mAP
    start_epoch =0
    best_mAP =0
    stage2_logs_dir = osp.join(args.logs_dir+'/'+log_s2_name, str(trial))
    # args.logs_dir = osp.join(logs_dir_root,str(trial))
    start_time = time.monotonic()

    # cudnn.benchmark = True
    sys.stdout = Logger(osp.join(stage2_logs_dir, str(trial)+'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset_ir = get_data('regdb_ir', args.data_dir, trial=trial)
    dataset_rgb = get_data('regdb_rgb', args.data_dir, trial=trial)

    # Create model
    model_list = []
    model_list.append(create_model(args))
    model_list.append(create_model(args))

    for ii in range(len(model_list)):
        checkpoint = load_checkpoint(osp.join(stage1_logs_dir, 'model_best_'+str(ii)+'.pth.tar'))
        model_list[ii].load_state_dict(checkpoint['state_dict'])
        print("==> Load last checkpoint successfully! ", osp.join(stage1_logs_dir, 'model_best_'+str(ii)+'.pth.tar'))
    
    prob_list, pred_list = [], []
    for ii in range(len(model_list)):
        prob_list.append(0)
        pred_list.append(0)

    # Optimizer
    params_list = []
    for ii in range(len(model_list)):
        params_list.append([{"params": [value]} for _, value in model_list[ii].named_parameters() if value.requires_grad])
    optimizer_list = []
    for ii in range(len(model_list)):
        optimizer_list.append(torch.optim.Adam(params_list[ii], lr=args.lr_list[ii], weight_decay=args.weight_decay))
    lr_scheduler_list = []
    for ii in range(len(model_list)):
        lr_scheduler_list.append(torch.optim.lr_scheduler.StepLR(optimizer_list[ii], step_size=args.step_size, gamma=0.1))

    # Trainer
    trainer_list = []
    for ii in range(len(model_list)):
        trainer_list.append(ClusterContrastTrainer(model_list[ii]))

    is_best, best_mAP = [], []
    for ii in range(len(model_list)):
        is_best.append(0)
        best_mAP.append(0)
    
    # features and pseudo-labels
    cluster_centers_ir_list, cluster_centers_rgb_list = [], []
    cluster_features_ir_list, cluster_features_rgb_list = [], []
    pseudo_labels_ir_list, pseudo_labels_rgb_list = [], []
    pseudo_labeled_dataset_ir_list, pseudo_labeled_dataset_rgb_list = [], []
    features_ir_list, features_rgb_list = [], []
    for ii in range(len(model_list)):
        cluster_centers_ir_list.append(0)
        cluster_centers_rgb_list.append(0)
        cluster_features_ir_list.append(0)
        cluster_features_rgb_list.append(0)
        pseudo_labels_ir_list.append(0)
        pseudo_labels_rgb_list.append(0)
        pseudo_labeled_dataset_ir_list.append(0)
        pseudo_labeled_dataset_rgb_list.append(0)
        features_ir_list.append(0)
        features_rgb_list.append(0)

    # relationships
    i2r_list, r2i_list = [{}, {}], [{}, {}]

    # gmm
    prob_ir_list, prob_rgb_list = [], []
    pred_ir_list, pred_rgb_list = [], []
    pseudo_labeled_easy_dataset_ir_list, pseudo_labeled_easy_dataset_rgb_list = [], []
    pseudo_labeled_diff_dataset_ir_list, pseudo_labeled_diff_dataset_rgb_list = [], []
    p_threshold_ir_list, p_threshold_rgb_list = [], []
    for ii in range(len(model_list)):
        prob_ir_list.append(0)
        prob_rgb_list.append(0)
        pred_ir_list.append(0)
        pred_rgb_list.append(0)
        pseudo_labeled_easy_dataset_ir_list.append(0)
        pseudo_labeled_easy_dataset_rgb_list.append(0)
        pseudo_labeled_diff_dataset_ir_list.append(0)
        pseudo_labeled_diff_dataset_rgb_list.append(0)
        p_threshold_ir_list.append(0)
        p_threshold_rgb_list.append(0)

    for epoch in range(args.epochs):
        # cluster ################################################
        cluster_start_time = time.time()
        for ii in range(len(model_list)):
            with torch.no_grad():
                if epoch == 0:
                    ir_num_clusters = args.ir_num_cluster + args.over_cluster
                    print('IR Clustering criterion: num_clusters: {:d}'.format(ir_num_clusters))
                    rgb_num_clusters = args.rgb_num_cluster + args.over_cluster
                    print('RGB Clustering criterion: num_clusters: {:d}'.format(rgb_num_clusters))

                    hierarchical_clustering_ir = cuAgglomerativeClustering(n_clusters=ir_num_clusters)
                    hierarchical_clustering_rgb = cuAgglomerativeClustering(n_clusters=rgb_num_clusters)

                print('==> Create pseudo labels for unlabeled RGB data')
                cluster_loader_rgb = get_test_loader(dataset_rgb, args.height, args.width,
                                                args.batch_size, args.workers, testset=sorted(dataset_rgb.train))
                features_rgb, _ = extract_features(model_list[ii], cluster_loader_rgb, print_freq=50, mode=1)
                del cluster_loader_rgb
                features_rgb_list[ii] = torch.cat([features_rgb[f].unsqueeze(0) for f, _, _ in sorted(dataset_rgb.train)], 0)

                print('==> Create pseudo labels for unlabeled IR data')
                cluster_loader_ir = get_test_loader(dataset_ir, args.height, args.width,
                                                args.batch_size, args.workers, testset=sorted(dataset_ir.train))
                features_ir, _ = extract_features(model_list[ii], cluster_loader_ir, print_freq=50, mode=2)
                del cluster_loader_ir
                features_ir_list[ii] = torch.cat([features_ir[f].unsqueeze(0) for f, _, _ in sorted(dataset_ir.train)], 0)          

                ## administrative cluster for ir
                rerank_dist_ir = compute_jaccard_distance(features_ir_list[ii], k1=args.k1, k2=args.k2, search_option=3)
                hierarchical_labels_ir = cp.asnumpy(hierarchical_clustering_ir.fit_predict(rerank_dist_ir))
                initial_centers_ir = []
                for label in np.unique(hierarchical_labels_ir):
                    cluster_points = rerank_dist_ir[hierarchical_labels_ir == label]
                    cluster_center = np.mean(cluster_points, axis=0)
                    initial_centers_ir.append(cluster_center)
                initial_centers_ir = np.array(initial_centers_ir)
                ## administrative cluster for rgb
                rerank_dist_rgb = compute_jaccard_distance(features_rgb_list[ii], k1=args.k1, k2=args.k2, search_option=3)
                hierarchical_labels_rgb = cp.asnumpy(hierarchical_clustering_rgb.fit_predict(rerank_dist_rgb))
                initial_centers_rgb = []
                for label in np.unique(hierarchical_labels_rgb):
                    cluster_points = rerank_dist_rgb[hierarchical_labels_rgb == label]
                    cluster_center = np.mean(cluster_points, axis=0)
                    initial_centers_rgb.append(cluster_center)
                initial_centers_rgb = np.array(initial_centers_rgb)

                ## k-means cluster for ir
                cluster_ir = cuKMeans(n_clusters=ir_num_clusters, init=initial_centers_ir, n_init=5)
                cluster_ir.fit(rerank_dist_ir)
                pseudo_labels_ir_list[ii], cluster_centers_ir_list[ii] = cp.asnumpy(cluster_ir.labels_), cp.asnumpy(cluster_ir.cluster_centers_)
                ## k-means cluster for rgb
                cluster_rgb = cuKMeans(n_clusters=ir_num_clusters, init=initial_centers_rgb, n_init=5)
                cluster_rgb.fit(rerank_dist_rgb)
                pseudo_labels_rgb_list[ii], cluster_centers_rgb_list[ii] = cp.asnumpy(cluster_rgb.labels_), cp.asnumpy(cluster_rgb.cluster_centers_)

                num_cluster_ir, num_cluster_rgb = len(set(pseudo_labels_ir_list[ii])), len(set(pseudo_labels_rgb_list[ii]))
                print("num_cluster_ir: ", num_cluster_ir, "  num_cluster_rgb: ", num_cluster_rgb)
                num_cluster_ir, num_cluster_rgb = ir_num_clusters, rgb_num_clusters

                del rerank_dist_ir
                del rerank_dist_rgb
        # ################################################
        cluster_end_time = time.time()
        print("cluster time: ", cluster_end_time-cluster_start_time)

        # pseudo-label pairing ################################################
        mapping_labels_ir = {}
        iii = 0
        cluster_centers_ir_temp = cluster_centers_ir_list[0]
        for centroid in cluster_centers_ir_temp:
            distances = [np.linalg.norm(centroid - c) for c in cluster_centers_ir_list[1]]
            min_index = 0
            while True:
                closest_label = np.argsort(distances)[min_index]
                if closest_label in set(mapping_labels_ir.values()):
                    min_index += 1
                else:
                    mapping_labels_ir[iii] = closest_label
                    break   
            iii += 1
        pseudo_labels_ir_list[0] = np.array([mapping_labels_ir[label] for label in pseudo_labels_ir_list[0]])

        mapping_labels_rgb = {}
        iii = 0
        for centroid in cluster_centers_rgb_list[0]:
            distances = [np.linalg.norm(centroid - c) for c in cluster_centers_rgb_list[1]]
            min_index = 0
            while True:
                closest_label = np.argsort(distances)[min_index]
                if closest_label in set(mapping_labels_rgb.values()):
                    min_index += 1
                else:
                    mapping_labels_rgb[iii] = closest_label
                    break   
            iii += 1
        pseudo_labels_rgb_list[0] = np.array([mapping_labels_rgb[label] for label in pseudo_labels_rgb_list[0]])

        if len(set(pseudo_labels_ir_list[0])) > len(set(pseudo_labels_rgb_list[0])):
            for iii in range(len(pseudo_labels_ir_list[0])):
                if pseudo_labels_ir_list[0][iii] not in set(pseudo_labels_rgb_list[0]):
                    pseudo_labels_ir_list[0][iii] = -1
            for iii in range(len(pseudo_labels_ir_list[0])):
                if pseudo_labels_ir_list[1][iii] not in set(pseudo_labels_rgb_list[0]):
                    pseudo_labels_ir_list[1][iii] = -1
            for iii in range(len(pseudo_labels_ir_list[0])):
                if pseudo_labels_rgb_list[1][iii] not in set(pseudo_labels_rgb_list[0]):
                    pseudo_labels_rgb_list[1][iii] = -1
        elif len(set(pseudo_labels_ir_list[0])) < len(set(pseudo_labels_rgb_list[0])):
            for iii in range(len(pseudo_labels_rgb_list[0])):
                if pseudo_labels_rgb_list[0][iii] not in set(pseudo_labels_ir_list[0]):
                    pseudo_labels_rgb_list[0][iii] = -1
            for iii in range(len(pseudo_labels_ir_list[0])):
                if pseudo_labels_rgb_list[1][iii] not in set(pseudo_labels_ir_list[0]):
                    pseudo_labels_rgb_list[1][iii] = -1
            for iii in range(len(pseudo_labels_ir_list[0])):
                if pseudo_labels_ir_list[1][iii] not in set(pseudo_labels_ir_list[0]):
                    pseudo_labels_ir_list[1][iii] = -1
        # ################################################

        # memory banks ################################################ 
        for ii in range(len(model_list)):
            # generate new dataset and calculate cluster centers
            @torch.no_grad()
            def generate_cluster_features(labels, features):
                centers = collections.defaultdict(list)
                for i, label in enumerate(labels):
                    if label == -1:
                        continue
                    centers[labels[i]].append(features[i])

                centers = [
                    torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
                ]

                centers = torch.stack(centers, dim=0)
                return centers

            cluster_features_ir_list[ii] = generate_cluster_features(pseudo_labels_ir_list[ii], features_ir_list[ii]) 
            cluster_features_rgb_list[ii] = generate_cluster_features(pseudo_labels_rgb_list[ii], features_rgb_list[ii])
            memory_ir = ClusterMemory(model_list[ii].module.num_features, num_cluster_ir, temp=args.temp,
                                momentum=args.momentum, use_hard=args.use_hard).cuda()
            memory_rgb = ClusterMemory(model_list[ii].module.num_features, num_cluster_rgb, temp=args.temp,
                                momentum=args.momentum, use_hard=args.use_hard).cuda()
            memory_ir.features = F.normalize(cluster_features_ir_list[ii], dim=1).cuda()
            memory_rgb.features = F.normalize(cluster_features_rgb_list[ii], dim=1).cuda()

            trainer_list[ii].memory_ir = memory_ir
            trainer_list[ii].memory_rgb = memory_rgb
            # ################################################

            # CMM ################################################ 
            print("Cross-modal Matching")
            R = []
            bgm = False
            
            # clusternorm
            cluster_features_rgb, cluster_features_ir = \
                F.normalize(cluster_features_rgb_list[ii], dim=1), F.normalize(cluster_features_ir_list[ii], dim=1)
            print("cluster_features_rgb: ", cluster_features_rgb.shape)
            print("cluster_features_ir: ", cluster_features_ir.shape)

            # [-1, 1] torch.mm(cluster_features_rgb, cluster_features_ir.T) #CostMatrix
            similarity = 1 - (torch.mm(cluster_features_rgb, cluster_features_ir.T))/1 #.exp().cpu()
            dis_similarity = similarity.exp().cpu()
            cost = dis_similarity / 1
            tmp = torch.zeros(dis_similarity.shape[0], dis_similarity.shape[0] - dis_similarity.shape[1])
            cost = (torch.cat((cost, tmp), 1))
            unmatched_row = []
            row_ind, col_ind = linear_sum_assignment(cost)
            for idx, item in enumerate(row_ind):
                if col_ind[idx] < similarity.shape[1]:
                    R.append((row_ind[idx], col_ind[idx]))
                    r2i_list[ii][row_ind[idx]] = col_ind[idx]
                    i2r_list[ii][col_ind[idx]] = row_ind[idx]
                else:
                    unmatched_row.append(row_ind[idx])
            if bgm is False:
                unmatched_cost = cost[unmatched_row][:,:dis_similarity.shape[1]]
                unmatched_row_ind, unmatched_col_ind = linear_sum_assignment(unmatched_cost)
                for idx, item in enumerate(unmatched_row_ind):
                    R.append((unmatched_row[idx], unmatched_col_ind[idx]))
                    r2i_list[ii][unmatched_row[idx]] = unmatched_col_ind[idx]
            del cluster_features_ir, cluster_features_rgb
            print("Cross-modal Matching Done")
        # ################################################

        # GMM(dataset_name) ################################################
        for ii in range(len(model_list)):
            print("Model "+str(ii)+": for RGB. Done\n")
            gmm_loader_rgb = get_test_loader(dataset_rgb, args.height, args.width,
                                            args.batch_size, args.workers, testset=sorted(dataset_rgb.train))
            prob_rgb_list[ii], p_threshold_rgb_list[ii] = trainer_list[ii].eval_train(gmm_loader_rgb, mode=1, dataset_name='RegDB')

            print("Model "+str(ii)+": for IR. Done\n")
            gmm_loader_ir = get_test_loader(dataset_ir, args.height, args.width,
                                            args.batch_size, args.workers, testset=sorted(dataset_ir.train))
            prob_ir_list[ii], p_threshold_ir_list[ii] = trainer_list[ii].eval_train(gmm_loader_ir, mode=2, dataset_name='RegDB')

            del gmm_loader_rgb
            del gmm_loader_ir

            pred_ir_list[ii] = (prob_ir_list[ii] > p_threshold_ir_list[ii])
            pred_rgb_list[ii] = (prob_rgb_list[ii] > p_threshold_rgb_list[ii])

            # setting GMM lower boundary
            if np.sum(pred_ir_list[ii]) < len(pred_ir_list[ii]) * args.gmm_lower_boundary:
                p_threshold_temp = np.sort(prob_ir_list[ii])[int(len(prob_ir_list[ii]) * (1 - args.gmm_lower_boundary))]
                pred_ir_list[ii] = (prob_ir_list[ii] > p_threshold_temp)
            if np.sum(pred_rgb_list[ii]) < len(pred_rgb_list[ii]) * args.gmm_lower_boundary:
                p_threshold_temp = np.sort(prob_rgb_list[ii])[int(len(prob_rgb_list[ii]) * (1 - args.gmm_lower_boundary))]
                pred_rgb_list[ii] = (prob_rgb_list[ii] > p_threshold_temp)
           
            print("IR diff:", np.sum(pred_ir_list[ii]), "    IR easy:", len(pred_ir_list[ii]) - np.sum(pred_ir_list[ii]))
            print("RGB diff:", np.sum(pred_rgb_list[ii]), "    RGB easy:", len(pred_rgb_list[ii]) - np.sum(pred_rgb_list[ii]))

            pseudo_labeled_dataset_ir_list[ii] = []
            pseudo_labeled_easy_dataset_ir_list[ii] = []
            pseudo_labeled_diff_dataset_ir_list[ii] = []
            ir_label=[]
            for index, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_ir.train), pseudo_labels_ir_list[ii])):
                if label != -1:
                    if not pred_ir_list[ii][index]:     # easy samples
                        pseudo_labeled_easy_dataset_ir_list[ii].append((fname, label.item(), cid))
                    elif pred_ir_list[ii][index]:       # difficult samples
                        pseudo_labeled_diff_dataset_ir_list[ii].append((fname, label.item(), cid))

                    pseudo_labeled_dataset_ir_list[ii].append((fname, label.item(), cid))
                    ir_label.append(label.item())
            print('==> Statistics for IR epoch {}: {} clusters'.format(epoch, num_cluster_ir))

            pseudo_labeled_dataset_rgb_list[ii] = []
            pseudo_labeled_easy_dataset_rgb_list[ii] = []
            pseudo_labeled_diff_dataset_rgb_list[ii] = []
            rgb_label=[]
            for index, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_rgb.train), pseudo_labels_rgb_list[ii])):
                if label != -1:
                    if not pred_rgb_list[ii][index]:    # easy samples
                        pseudo_labeled_easy_dataset_rgb_list[ii].append((fname, label.item(), cid))
                    elif pred_rgb_list[ii][index]:      # difficult samples
                        pseudo_labeled_diff_dataset_rgb_list[ii].append((fname, label.item(), cid))

                    pseudo_labeled_dataset_rgb_list[ii].append((fname, label.item(), cid))
                    rgb_label.append(label.item())
            print('==> Statistics for RGB epoch {}: {} clusters'.format(epoch, num_cluster_rgb))
        # ################################################

        # co-learning ################################################ 
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        height, width = args.height, args.width
        for ii in range(len(model_list)):
            train_transformer_rgb = T.Compose([
                T.Resize((height, width), interpolation=3),
                T.Pad(10),
                T.RandomCrop((height, width)),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                normalizer,
                ChannelRandomErasing(probability = 0.5)
            ])
            
            train_transformer_rgb1 = T.Compose([
                T.Resize((height, width), interpolation=3),
                T.Pad(10),
                T.RandomCrop((height, width)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                T.ToTensor(),
                normalizer,
                ChannelRandomErasing(probability = 0.5),
                ChannelExchange(gray = 2)
            ])

            transform_thermal = T.Compose( [
                T.Resize((height, width), interpolation=3),
                T.Pad(10),
                T.RandomCrop((288, 144)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalizer,
                ChannelRandomErasing(probability = 0.5),
                ChannelAdapGray(probability =0.5)])
            
            # self-training ####################################
            ## self-training (for easy samples) ################################# 
            train_loader_ir_easy = get_train_loader_ir(args, dataset_ir, args.height, args.width,
                                            args.batch_size, args.workers, args.num_instances, iters,
                                            trainset=pseudo_labeled_easy_dataset_ir_list[ii], no_cam=args.no_cam, train_transformer=transform_thermal)
            train_loader_rgb_easy = get_train_loader_color(args, dataset_rgb, args.height, args.width,
                                            128, args.workers, args.num_instances, iters,
                                            trainset=pseudo_labeled_easy_dataset_rgb_list[ii], no_cam=args.no_cam, train_transformer=train_transformer_rgb, train_transformer1=train_transformer_rgb1)
            train_loader_ir_easy.new_epoch()
            train_loader_rgb_easy.new_epoch()
            print("Self-Training: model[", ii, "] for EASY samples")
            trainer_list[ii].train_gmm(epoch, train_loader_ir_easy, train_loader_rgb_easy, optimizer_list[ii],
                        print_freq=args.print_freq, train_iters=len(train_loader_ir_easy), i2r=i2r_list[ii], r2i=r2i_list[ii], gmm_mode="easy",
                        gmm_p_threshold_ir=p_threshold_ir_list[ii], gmm_p_threshold_rgb=p_threshold_rgb_list[ii],
                        sharpen_easy_tau=args.sharpen_easy_tau, sharpen_diff_tau=args.sharpen_diff_tau)
            ## self-training (for diff samples) #################################
            train_loader_ir_diff = get_train_loader_ir(args, dataset_ir, args.height, args.width,
                                            args.batch_size, args.workers, args.num_instances, iters,
                                            trainset=pseudo_labeled_diff_dataset_ir_list[ii], no_cam=args.no_cam, train_transformer=transform_thermal)
            train_loader_rgb_diff = get_train_loader_color(args, dataset_rgb, args.height, args.width,
                                            128, args.workers, args.num_instances, iters,
                                            trainset=pseudo_labeled_diff_dataset_rgb_list[ii], no_cam=args.no_cam, train_transformer=train_transformer_rgb, train_transformer1=train_transformer_rgb1)
            train_loader_ir_diff.new_epoch()
            train_loader_rgb_diff.new_epoch()
            print("Self-Training: model[", ii, "] for DIFFICULT samples")
            trainer_list[ii].train_gmm(epoch, train_loader_ir_diff, train_loader_rgb_diff, optimizer_list[ii],
                        print_freq=args.print_freq, train_iters=len(train_loader_ir_diff), i2r=i2r_list[ii], r2i=r2i_list[ii], gmm_mode="diff",
                        gmm_p_threshold_ir=p_threshold_ir_list[ii], gmm_p_threshold_rgb=p_threshold_rgb_list[ii],
                        sharpen_easy_tau=args.sharpen_easy_tau, sharpen_diff_tau=args.sharpen_diff_tau)

            # cross-training ####################################
            ## cross-training (for easy samples) ################################# 
            train_loader_ir_easy = get_train_loader_ir(args, dataset_ir, args.height, args.width,
                                            args.batch_size, args.workers, args.num_instances, iters,
                                            trainset=pseudo_labeled_easy_dataset_ir_list[(ii+1) % len(model_list)], no_cam=args.no_cam,train_transformer=transform_thermal)
            train_loader_rgb_easy = get_train_loader_color(args, dataset_rgb, args.height, args.width,
                                            128, args.workers, args.num_instances, iters,
                                            trainset=pseudo_labeled_easy_dataset_rgb_list[(ii+1) % len(model_list)], no_cam=args.no_cam,train_transformer=train_transformer_rgb,train_transformer1=train_transformer_rgb1)
            train_loader_ir_easy.new_epoch()
            train_loader_rgb_easy.new_epoch()
            print("Cross-Training: model[", ii, "] for EASY samples")
            trainer_list[ii].train_gmm(epoch, train_loader_ir_easy, train_loader_rgb_easy, optimizer_list[ii],
                        print_freq=args.print_freq, train_iters=len(train_loader_ir_easy), i2r=i2r_list[(ii+1) % len(model_list)], r2i=r2i_list[(ii+1) % len(model_list)],
                        gmm_p_threshold_ir=p_threshold_ir_list[ii], gmm_p_threshold_rgb=p_threshold_rgb_list[ii],
                        sharpen_easy_tau=args.sharpen_easy_tau, sharpen_diff_tau=args.sharpen_diff_tau)
            ## cross-training (for diff samples) ################################# 
            train_loader_ir_diff = get_train_loader_ir(args, dataset_ir, args.height, args.width,
                                            args.batch_size, args.workers, args.num_instances, iters,
                                            trainset=pseudo_labeled_diff_dataset_ir_list[(ii+1) % len(model_list)], no_cam=args.no_cam,train_transformer=transform_thermal)
            train_loader_rgb_diff = get_train_loader_color(args, dataset_rgb, args.height, args.width,
                                            128, args.workers, args.num_instances, iters,
                                            trainset=pseudo_labeled_diff_dataset_rgb_list[(ii+1) % len(model_list)], no_cam=args.no_cam,train_transformer=train_transformer_rgb,train_transformer1=train_transformer_rgb1)
            train_loader_ir_diff.new_epoch()
            train_loader_rgb_diff.new_epoch()
            print("Cross-Training: model[", ii, "] for DIFFICULT samples")
            trainer_list[ii].train_gmm(epoch, train_loader_ir_diff, train_loader_rgb_diff, optimizer_list[ii],
                        print_freq=args.print_freq, train_iters=len(train_loader_ir_diff), i2r=i2r_list[(ii+1) % len(model_list)], r2i=r2i_list[(ii+1) % len(model_list)],
                        gmm_p_threshold_ir=p_threshold_ir_list[ii], gmm_p_threshold_rgb=p_threshold_rgb_list[ii],
                        sharpen_easy_tau=args.sharpen_easy_tau, sharpen_diff_tau=args.sharpen_diff_tau)

            # evaluation #############################
            if epoch>=0 and ( (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1)):
                args.test_batch=64
                args.img_w=args.width
                args.img_h=args.height
                normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                transform_test = T.Compose([
                    T.ToPILImage(),
                    T.Resize((args.img_h,args.img_w)),
                    T.ToTensor(),
                    normalize,
                ])

                ## ############## visible2thermal
                mode='visible2thermal'
                data_path='/home/liyongxiang/code/ReID/dataset/RegDB/'

                if mode == 'visible2thermal':
                    query_img, query_label = process_test_regdb(data_path, trial=trial, modal='visible')
                    gall_img, gall_label = process_test_regdb(data_path, trial=trial, modal='thermal')
                elif mode == 'thermal2visible':
                    query_img, query_label = process_test_regdb(data_path, trial=trial, modal='thermal')
                    gall_img, gall_label = process_test_regdb(data_path, trial=trial, modal='visible')

                gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
                gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
                nquery = len(query_label)
                ngall = len(gall_label)
                queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
                query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
                query_feat_fc = extract_query_feat(model_list[ii],query_loader,nquery)
                # for trial in range(1):
                ngall = len(gall_label)
                gall_feat_fc = extract_gall_feat(model_list[ii], gall_loader, ngall)
                # fc feature
                distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
                v2t_cmc, v2t_mAP, v2t_mINP = eval_regdb(-distmat, query_label, gall_label)

                print('Test {} Trial: {}'.format(mode, trial))
                print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                        v2t_cmc[0], v2t_cmc[4], v2t_cmc[9], v2t_cmc[19], v2t_mAP, v2t_mINP))
                
                ## ############## thermal2visible
                mode='thermal2visible'
                data_path='/home/liyongxiang/code/ReID/dataset/RegDB/'

                if mode == 'visible2thermal':
                    query_img, query_label = process_test_regdb(data_path, trial=trial, modal='visible')
                    gall_img, gall_label = process_test_regdb(data_path, trial=trial, modal='thermal')
                elif mode == 'thermal2visible':
                    query_img, query_label = process_test_regdb(data_path, trial=trial, modal='thermal')
                    gall_img, gall_label = process_test_regdb(data_path, trial=trial, modal='visible')

                gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
                gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
                nquery = len(query_label)
                ngall = len(gall_label)
                queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
                query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
                query_feat_fc = extract_query_feat(model_list[ii],query_loader,nquery)
                # for trial in range(1):
                ngall = len(gall_label)
                gall_feat_fc = extract_gall_feat(model_list[ii], gall_loader, ngall)
                # fc feature
                distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
                t2v_cmc, t2v_mAP, t2v_mINP = eval_regdb(-distmat, query_label, gall_label)

                print('Model:[{}]  Test:{}  Trial:{}'.format(str(ii), mode, trial))
                print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                        t2v_cmc[0], t2v_cmc[4], t2v_cmc[9], t2v_cmc[19], t2v_mAP, t2v_mINP))

                is_best[ii] = ((v2t_mAP+t2v_mAP)/2 > best_mAP[ii])
                best_mAP[ii] = max((v2t_mAP+t2v_mAP)/2, best_mAP[ii])

                print("checkpoint: ", osp.join(args.logs_dir, 'checkpoint'+str(ii)+'.pth.tar'))
                print("model_best: ", osp.join(args.logs_dir, 'model_best_'+str(ii)+'.pth.tar'))
                
                save_checkpoint({
                    'state_dict': model_list[ii].state_dict(),
                    'epoch': epoch + 1,
                    'best_mAP': best_mAP[ii],
                }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint'+str(ii)+'.pth.tar'), model_best_path=osp.join(args.logs_dir, 'model_best_'+str(ii)+'.pth.tar'))

                print('\n * Finished epoch {:3d}  v2t model mAP: {:5.1%}  t2v model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                    format(epoch, v2t_mAP, t2v_mAP, best_mAP[ii], ' *' if is_best[ii] else ''))
    ############################
            lr_scheduler_list[ii].step()
    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))

def main_worker_stage2_colearning_backup(args, log_s1_name, log_s2_name):
    # logs_dir_root = osp.join('logs/'+log_s2_name)
    trial = args.trial
    stage1_logs_dir = osp.join(args.logs_dir+'/'+log_s1_name, str(trial))
    # global start_epoch, best_mAP
    start_epoch =0
    best_mAP =0
    stage2_logs_dir = osp.join(args.logs_dir+'/'+log_s2_name, str(trial))
    # args.logs_dir = osp.join(logs_dir_root,str(trial))
    start_time = time.monotonic()

    # cudnn.benchmark = True
    sys.stdout = Logger(osp.join(stage2_logs_dir, str(trial)+'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset_ir = get_data('regdb_ir', args.data_dir, trial=trial)
    dataset_rgb = get_data('regdb_rgb', args.data_dir, trial=trial)

    # Create model
    model_list = []
    model_list.append(create_model(args))
    model_list.append(create_model(args))

    for ii in range(len(model_list)):
        checkpoint = load_checkpoint(osp.join(stage1_logs_dir, 'model_best_'+str(ii)+'.pth.tar'))
        model_list[ii].load_state_dict(checkpoint['state_dict'])
        print("==> Load last checkpoint successfully! ", osp.join(stage1_logs_dir, 'model_best_'+str(ii)+'.pth.tar'))
    
    prob_list = []
    for ii in range(len(model_list)):
        prob_list.append(0)

    # Optimizer
    params_list = []
    for ii in range(len(model_list)):
        params_list.append([{"params": [value]} for _, value in model_list[ii].named_parameters() if value.requires_grad])
    optimizer_list = []
    for ii in range(len(model_list)):
        optimizer_list.append(torch.optim.Adam(params_list[ii], lr=args.lr_list[ii], weight_decay=args.weight_decay))
    lr_scheduler_list = []
    for ii in range(len(model_list)):
        lr_scheduler_list.append(torch.optim.lr_scheduler.StepLR(optimizer_list[ii], step_size=args.step_size, gamma=0.1))

    # Trainer
    trainer_list = []
    for ii in range(len(model_list)):
        trainer_list.append(ClusterContrastTrainer(model_list[ii]))

    is_best, best_mAP = [], []
    for ii in range(len(model_list)):
        is_best.append(0)
        best_mAP.append(0)
    
    # features and pseudo-labels
    cluster_centers_ir_list, cluster_centers_rgb_list = [], []
    cluster_features_ir_list, cluster_features_rgb_list = [], []
    pseudo_labels_ir_list, pseudo_labels_rgb_list = [], []
    pseudo_labeled_dataset_ir_list, pseudo_labeled_dataset_rgb_list = [], []
    features_ir_list, features_rgb_list = [], []
    for ii in range(len(model_list)):
        cluster_centers_ir_list.append(0)
        cluster_centers_rgb_list.append(0)
        cluster_features_ir_list.append(0)
        cluster_features_rgb_list.append(0)
        pseudo_labels_ir_list.append(0)
        pseudo_labels_rgb_list.append(0)
        pseudo_labeled_dataset_ir_list.append(0)
        pseudo_labeled_dataset_rgb_list.append(0)
        features_ir_list.append(0)
        features_rgb_list.append(0)
    
    i2r_list, r2i_list = [{}, {}], [{}, {}]

    for epoch in range(args.epochs):
        for ii in range(len(model_list)):
            with torch.no_grad():
                if epoch == 0:
                    # MiniBatchKMeans cluster
                    ir_num_clusters = args.ir_num_cluster + args.over_cluster
                    print('IR Clustering criterion: num_clusters: {:d}'.format(ir_num_clusters))
                    rgb_num_clusters = args.rgb_num_cluster + args.over_cluster
                    print('RGB Clustering criterion: num_clusters: {:d}'.format(rgb_num_clusters))

                print('==> Create pseudo labels for unlabeled RGB data')
                cluster_loader_rgb = get_test_loader(dataset_rgb, args.height, args.width,
                                                args.batch_size, args.workers, testset=sorted(dataset_rgb.train))
                
                features_rgb, _ = extract_features(model_list[ii], cluster_loader_rgb, print_freq=50, mode=1)
                del cluster_loader_rgb
                features_rgb_list[ii] = torch.cat([features_rgb[f].unsqueeze(0) for f, _, _ in sorted(dataset_rgb.train)], 0)

                print('==> Create pseudo labels for unlabeled IR data')
                cluster_loader_ir = get_test_loader(dataset_ir, args.height, args.width,
                                                args.batch_size, args.workers, testset=sorted(dataset_ir.train))
                features_ir, _ = extract_features(model_list[ii], cluster_loader_ir, print_freq=50, mode=2)
                del cluster_loader_ir
                features_ir_list[ii] = torch.cat([features_ir[f].unsqueeze(0) for f, _, _ in sorted(dataset_ir.train)], 0)            

                cluster_start_time = time.time()
                rerank_dist_ir = compute_jaccard_distance(features_ir_list[ii], k1=args.k1, k2=args.k2, search_option=3)
                rerank_dist_rgb = compute_jaccard_distance(features_rgb_list[ii], k1=args.k1, k2=args.k2, search_option=3)
                print("rerank_dist_ir: ", rerank_dist_ir.shape)
                print("rerank_dist_rgb: ", rerank_dist_rgb.shape)

                ## mini k-means cluster for ir
                cluster_ir = KMeans(n_clusters=ir_num_clusters, batch_size=rerank_dist_ir.shape[0], init='k-means++', n_init=5)
                cluster_ir.fit(rerank_dist_ir)
                pseudo_labels_ir_list[ii], cluster_centers_ir_list[ii] = cluster_ir.labels_, cluster_ir.cluster_centers_
                ## mini k-means cluster for rgb
                cluster_rgb = KMeans(n_clusters=ir_num_clusters, batch_size=rerank_dist_rgb.shape[0], init='k-means++', n_init=5)
                cluster_rgb.fit(rerank_dist_rgb)
                pseudo_labels_rgb_list[ii], cluster_centers_rgb_list[ii] = cluster_rgb.labels_, cluster_rgb.cluster_centers_

                num_cluster_ir, num_cluster_rgb = len(set(pseudo_labels_ir_list[ii])), len(set(pseudo_labels_rgb_list[ii]))
                print("num_cluster_ir: ", num_cluster_ir, "  num_cluster_rgb: ", num_cluster_rgb)
                num_cluster_ir, num_cluster_rgb = ir_num_clusters, rgb_num_clusters

                cluster_end_time = time.time()
                print("cluster time: ", cluster_end_time - cluster_start_time)

                del rerank_dist_ir
                del rerank_dist_rgb

        print("num_cluster_ir0: ", len(set(pseudo_labels_ir_list[0])), "  num_cluster_ir1: ", len(set(pseudo_labels_ir_list[1])))
        print("num_cluster_rgb0: ", len(set(pseudo_labels_rgb_list[0])), "  num_cluster_rgb1: ", len(set(pseudo_labels_rgb_list[1])))

        # pseudo-label pairing ################################################
        mapping_labels_ir = {}
        iii = 0
        cluster_centers_ir_temp = cluster_centers_ir_list[0]
        for centroid in cluster_centers_ir_temp:
            # distances = [np.linalg.norm(centroid - c) for c in cluster_centers_ir_list[1]]
            distances = [np.exp(1 - np.dot(centroid, c) / (np.linalg.norm(centroid) * np.linalg.norm(c))) for c in cluster_centers_ir_list[1]]
            min_index = 0
            while True:
                closest_label = np.argsort(distances)[min_index]
                if closest_label in set(mapping_labels_ir.values()):
                    min_index += 1
                else:
                    mapping_labels_ir[iii] = closest_label
                    break   
            iii += 1
        pseudo_labels_ir_list[0] = np.array([mapping_labels_ir[label] for label in pseudo_labels_ir_list[0]])

        mapping_labels_rgb = {}
        iii = 0
        for centroid in cluster_centers_rgb_list[0]:
            distances = [np.exp(1 - np.dot(centroid, c) / (np.linalg.norm(centroid) * np.linalg.norm(c))) for c in cluster_centers_rgb_list[1]]
            min_index = 0
            while True:
                closest_label = np.argsort(distances)[min_index]
                if closest_label in set(mapping_labels_rgb.values()):
                    min_index += 1
                else:
                    mapping_labels_rgb[iii] = closest_label
                    break   
            iii += 1
        pseudo_labels_rgb_list[0] = np.array([mapping_labels_rgb[label] for label in pseudo_labels_rgb_list[0]])

        if len(set(pseudo_labels_ir_list[0])) > len(set(pseudo_labels_rgb_list[0])):
            for iii in range(len(pseudo_labels_ir_list[0])):
                if pseudo_labels_ir_list[0][iii] not in set(pseudo_labels_rgb_list[0]):
                    pseudo_labels_ir_list[0][iii] = -1
            for iii in range(len(pseudo_labels_ir_list[0])):
                if pseudo_labels_ir_list[1][iii] not in set(pseudo_labels_rgb_list[0]):
                    pseudo_labels_ir_list[1][iii] = -1
            for iii in range(len(pseudo_labels_ir_list[0])):
                if pseudo_labels_rgb_list[1][iii] not in set(pseudo_labels_rgb_list[0]):
                    pseudo_labels_rgb_list[1][iii] = -1
        elif len(set(pseudo_labels_ir_list[0])) < len(set(pseudo_labels_rgb_list[0])):
            for iii in range(len(pseudo_labels_rgb_list[0])):
                if pseudo_labels_rgb_list[0][iii] not in set(pseudo_labels_ir_list[0]):
                    pseudo_labels_rgb_list[0][iii] = -1
            for iii in range(len(pseudo_labels_ir_list[0])):
                if pseudo_labels_rgb_list[1][iii] not in set(pseudo_labels_ir_list[0]):
                    pseudo_labels_rgb_list[1][iii] = -1
            for iii in range(len(pseudo_labels_ir_list[0])):
                if pseudo_labels_ir_list[1][iii] not in set(pseudo_labels_ir_list[0]):
                    pseudo_labels_ir_list[1][iii] = -1
        # ################################################

        for ii in range(len(model_list)):
            # generate new dataset and calculate cluster centers
            @torch.no_grad()
            def generate_cluster_features(labels, features):
                centers = collections.defaultdict(list)
                for i, label in enumerate(labels):
                    if label == -1:
                        continue
                    centers[labels[i]].append(features[i])

                centers = [
                    torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
                ]

                centers = torch.stack(centers, dim=0)
                return centers

            cluster_features_ir_list[ii] = generate_cluster_features(pseudo_labels_ir_list[ii], features_ir_list[ii]) 
            cluster_features_rgb_list[ii] = generate_cluster_features(pseudo_labels_rgb_list[ii], features_rgb_list[ii])
            memory_ir = ClusterMemory(model_list[ii].module.num_features, num_cluster_ir, temp=args.temp,
                                momentum=args.momentum, use_hard=args.use_hard).cuda()
            memory_rgb = ClusterMemory(model_list[ii].module.num_features, num_cluster_rgb, temp=args.temp,
                                momentum=args.momentum, use_hard=args.use_hard).cuda()
            memory_ir.features = F.normalize(cluster_features_ir_list[ii], dim=1).cuda()
            memory_rgb.features = F.normalize(cluster_features_rgb_list[ii], dim=1).cuda()

            trainer_list[ii].memory_ir = memory_ir
            trainer_list[ii].memory_rgb = memory_rgb

            pseudo_labeled_dataset_ir_list[ii] = []
            ir_label=[]
            for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_ir.train), pseudo_labels_ir_list[ii])):
                if label != -1:
                    pseudo_labeled_dataset_ir_list[ii].append((fname, label.item(), cid))
                    ir_label.append(label.item())
            print('==> Statistics for IR epoch {}: {} clusters'.format(epoch, num_cluster_ir))

            pseudo_labeled_dataset_rgb_list[ii] = []
            rgb_label=[]
            for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_rgb.train), pseudo_labels_rgb_list[ii])):
                if label != -1:
                    pseudo_labeled_dataset_rgb_list[ii].append((fname, label.item(), cid))
                    rgb_label.append(label.item())
            print('==> Statistics for RGB epoch {}: {} clusters'.format(epoch, num_cluster_rgb))
        
            ## ######################## PGM
            print("Progressive Graph Matching")
            R = []
            bgm = False
            
            # clusternorm
            cluster_features_rgb, cluster_features_ir = \
                F.normalize(cluster_features_rgb_list[ii], dim=1), F.normalize(cluster_features_ir_list[ii], dim=1)

            # [-1, 1] torch.mm(cluster_features_rgb, cluster_features_ir.T) #CostMatrix
            similarity = 1 - (torch.mm(cluster_features_rgb, cluster_features_ir.T)) / 1 #.exp().cpu()
            dis_similarity = similarity.exp().cpu()
            cost = dis_similarity / 1
            tmp = torch.zeros(dis_similarity.shape[0], dis_similarity.shape[0] - dis_similarity.shape[1])
            cost = (torch.cat((cost, tmp), 1))
            unmatched_row = []
            row_ind, col_ind = linear_sum_assignment(cost)
            for idx, item in enumerate(row_ind):
                if col_ind[idx] < similarity.shape[1]:
                    R.append((row_ind[idx], col_ind[idx]))
                    r2i_list[ii][row_ind[idx]] = col_ind[idx]
                    i2r_list[ii][col_ind[idx]] = row_ind[idx]
                else:
                    unmatched_row.append(row_ind[idx])
            if bgm is False:
                unmatched_cost = cost[unmatched_row][:,:dis_similarity.shape[1]]
                unmatched_row_ind, unmatched_col_ind = linear_sum_assignment(unmatched_cost)
                for idx, item in enumerate(unmatched_row_ind):
                    R.append((unmatched_row[idx], unmatched_col_ind[idx]))
                    r2i_list[ii][unmatched_row[idx]] = unmatched_col_ind[idx]
            del cluster_features_ir, cluster_features_rgb
            print("Progressive Graph Matching Done")

        for ii in range(len(model_list)):
            # training #############################
            normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
            height=args.height
            width=args.width
            train_transformer_rgb = T.Compose([
                T.Resize((height, width), interpolation=3),
                T.Pad(10),
                T.RandomCrop((height, width)),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                normalizer,
                ChannelRandomErasing(probability = 0.5)
            ])
            
            train_transformer_rgb1 = T.Compose([
                T.Resize((height, width), interpolation=3),
                T.Pad(10),
                T.RandomCrop((height, width)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                T.ToTensor(),
                normalizer,
                ChannelRandomErasing(probability = 0.5),
                ChannelExchange(gray = 2)
            ])

            transform_thermal = T.Compose( [
                T.Resize((height, width), interpolation=3),
                T.Pad(10),
                T.RandomCrop((288, 144)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalizer,
                ChannelRandomErasing(probability = 0.5),
                ChannelAdapGray(probability =0.5)])

            # self-training #############################
            train_loader_ir = get_train_loader_ir(args, dataset_ir, args.height, args.width,
                                            args.batch_size, args.workers, args.num_instances, iters,
                                            trainset=pseudo_labeled_dataset_ir_list[ii], no_cam=args.no_cam, train_transformer=transform_thermal)
            train_loader_rgb = get_train_loader_color(args, dataset_rgb, args.height, args.width,
                                            128, args.workers, args.num_instances, iters,
                                            trainset=pseudo_labeled_dataset_rgb_list[ii], no_cam=args.no_cam, train_transformer=train_transformer_rgb, train_transformer1=train_transformer_rgb1)
            train_loader_ir.new_epoch()
            train_loader_rgb.new_epoch()
            trainer_list[ii].train(epoch, train_loader_ir,train_loader_rgb, optimizer_list[ii],
                        print_freq=args.print_freq, train_iters=len(train_loader_ir), i2r=i2r_list[ii], r2i=r2i_list[ii])

            # cross-training #############################
            train_loader_ir = get_train_loader_ir(args, dataset_ir, args.height, args.width,
                                            args.batch_size, args.workers, args.num_instances, iters,
                                            trainset=pseudo_labeled_dataset_ir_list[(ii+1) % len(model_list)], no_cam=args.no_cam,train_transformer=transform_thermal)
            train_loader_rgb = get_train_loader_color(args, dataset_rgb, args.height, args.width,
                                            128, args.workers, args.num_instances, iters,
                                            trainset=pseudo_labeled_dataset_rgb_list[(ii+1) % len(model_list)], no_cam=args.no_cam,train_transformer=train_transformer_rgb,train_transformer1=train_transformer_rgb1)
            train_loader_ir.new_epoch()
            train_loader_rgb.new_epoch()
            trainer_list[ii].train(epoch, train_loader_ir, train_loader_rgb, optimizer_list[ii],
                        print_freq=args.print_freq, train_iters=len(train_loader_ir), i2r=i2r_list[(ii+1) % len(model_list)], r2i=r2i_list[(ii+1) % len(model_list)])

            # evaluation #############################
            if epoch>=0 and ( (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1)):
                args.test_batch=64
                args.img_w=args.width
                args.img_h=args.height
                normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                transform_test = T.Compose([
                    T.ToPILImage(),
                    T.Resize((args.img_h,args.img_w)),
                    T.ToTensor(),
                    normalize,
                ])

                ## ############## visible2thermal
                mode='visible2thermal'
                data_path='/home/liyongxiang/code/ReID/dataset/RegDB/'

                if mode == 'visible2thermal':
                    query_img, query_label = process_test_regdb(data_path, trial=trial, modal='visible')
                    gall_img, gall_label = process_test_regdb(data_path, trial=trial, modal='thermal')
                elif mode == 'thermal2visible':
                    query_img, query_label = process_test_regdb(data_path, trial=trial, modal='thermal')
                    gall_img, gall_label = process_test_regdb(data_path, trial=trial, modal='visible')

                gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
                gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
                nquery = len(query_label)
                ngall = len(gall_label)
                queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
                query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
                query_feat_fc = extract_query_feat(model_list[ii],query_loader,nquery)
                # for trial in range(1):
                ngall = len(gall_label)
                gall_feat_fc = extract_gall_feat(model_list[ii], gall_loader, ngall)
                # fc feature
                distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
                v2t_cmc, v2t_mAP, v2t_mINP = eval_regdb(-distmat, query_label, gall_label)

                print('Test {} Trial: {}'.format(mode, trial))
                print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                        v2t_cmc[0], v2t_cmc[4], v2t_cmc[9], v2t_cmc[19], v2t_mAP, v2t_mINP))
                
                ## ############## thermal2visible
                mode='thermal2visible'
                data_path='/home/liyongxiang/code/ReID/dataset/RegDB/'

                if mode == 'visible2thermal':
                    query_img, query_label = process_test_regdb(data_path, trial=trial, modal='visible')
                    gall_img, gall_label = process_test_regdb(data_path, trial=trial, modal='thermal')
                elif mode == 'thermal2visible':
                    query_img, query_label = process_test_regdb(data_path, trial=trial, modal='thermal')
                    gall_img, gall_label = process_test_regdb(data_path, trial=trial, modal='visible')

                gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
                gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
                nquery = len(query_label)
                ngall = len(gall_label)
                queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
                query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
                query_feat_fc = extract_query_feat(model_list[ii],query_loader,nquery)
                # for trial in range(1):
                ngall = len(gall_label)
                gall_feat_fc = extract_gall_feat(model_list[ii], gall_loader, ngall)
                # fc feature
                distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
                t2v_cmc, t2v_mAP, t2v_mINP = eval_regdb(-distmat, query_label, gall_label)

                print('Model:[{}]  Test:{}  Trial:{}'.format(str(ii), mode, trial))
                print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                        t2v_cmc[0], t2v_cmc[4], t2v_cmc[9], t2v_cmc[19], t2v_mAP, t2v_mINP))

                is_best[ii] = ((v2t_mAP+t2v_mAP)/2 > best_mAP[ii])
                best_mAP[ii] = max((v2t_mAP+t2v_mAP)/2, best_mAP[ii])

                print("checkpoint: ", osp.join(args.logs_dir, 'checkpoint'+str(ii)+'.pth.tar'))
                print("model_best: ", osp.join(args.logs_dir, 'model_best_'+str(ii)+'.pth.tar'))
                
                save_checkpoint({
                    'state_dict': model_list[ii].state_dict(),
                    'epoch': epoch + 1,
                    'best_mAP': best_mAP[ii],
                }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint'+str(ii)+'.pth.tar'), model_best_path=osp.join(args.logs_dir, 'model_best_'+str(ii)+'.pth.tar'))

                print('\n * Finished epoch {:3d}  v2t model mAP: {:5.1%}  t2v model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                    format(epoch, v2t_mAP, t2v_mAP, best_mAP[ii], ' *' if is_best[ii] else ''))
    ############################
            lr_scheduler_list[ii].step()
    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-paced contrastive learning on unsupervised re-ID")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='dukemtmcreid',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=2)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=288, help="input height")
    parser.add_argument('--width', type=int, default=144, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    parser.add_argument('--num-instances-eval-train', type=int, default=0,
                        help="each minibatch consist of "
                             "(batch_size // num_instances_eval_train) identities, and "
                             "each identity has num_instances_eval_train instances, "
                             "default: 0 (NOT USE)")
    # cluster
    parser.add_argument('--eps', type=float, default=0.6,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--ir_num_cluster', type=int, default=206,
                        help="IR image cluster number")
    parser.add_argument('--rgb_num_cluster', type=int, default=206,
                        help="Visible image cluster number")
    parser.add_argument('--over_cluster', type=int, default=20,
                        help="over cluster number")
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")

    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.2,
                        help="update momentum for the hybrid memory")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035, # 0.00035
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--step-size', type=int, default=10)
    # gmm
    parser.add_argument('--p_threshold', type=float, default=0.4,
                        help='clean probability threshold')
    parser.add_argument('--gmm_lower_boundary', type=float, default=0.05,
                        help='gmm lower boundary ratio (0-1)')
    parser.add_argument('--sharpen_easy_tau', type=float, default=6)
    parser.add_argument('--sharpen_diff_tau', type=float, default=0.6)
    # training configs
    parser.add_argument('--seed', type=int, default=None)  # 313
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=1)
    parser.add_argument('--trial', type=int, default=1)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--pooling-type', type=str, default='gem')
    parser.add_argument('--use-hard', action="store_true")
    parser.add_argument('--no-cam',  action="store_true")

    args = parser.parse_args()
    args.lr_list = [args.lr-0.00005, args.lr+0.00005] # 0.00035
    args.lr_list = [args.lr, args.lr]
    main()
