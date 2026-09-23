from __future__ import print_function, absolute_import
from audioop import cross
import time
from .utils.meters import AverageMeter

import math
import numpy as np
from sklearn.mixture import GaussianMixture

import torch
import torch.nn as nn
from torch.nn import functional as F

def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx 

def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6 # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

class ClusterContrastTrainer(object):
    def __init__(self, encoder, memory=None):
        super(ClusterContrastTrainer, self).__init__()
        self.encoder = encoder
        self.memory_ir = memory
        self.memory_rgb = memory

    def train(self, epoch, data_loader_ir, data_loader_rgb, optimizer, print_freq=10, train_iters=400, i2r=None, r2i=None):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs_ir, labels_ir, indexes_ir = self._parse_data_ir(inputs_ir)
            inputs_rgb, inputs_rgb1, labels_rgb, indexes_rgb = self._parse_data_rgb(inputs_rgb)

            # forward
            inputs_rgb = torch.cat((inputs_rgb, inputs_rgb1), 0)
            labels_rgb = torch.cat((labels_rgb, labels_rgb), -1)
            _, f_out_rgb, f_out_ir, labels_rgb, labels_ir, pool_rgb, pool_ir = \
                self._forward(inputs_rgb, inputs_ir, label_1=labels_rgb, label_2=labels_ir, modal=0)

            # intra-modality nce loss
            loss_ir = self.memory_ir(f_out_ir, labels_ir) 
            loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)

            # cross contrastive learning
            if r2i:
                rgb2ir_labels = torch.tensor([r2i[key.item()] for key in labels_rgb]).cuda()
                ir2rgb_labels = torch.tensor([i2r[key.item()] for key in labels_ir]).cuda()

                alternate = True
                if alternate:
                    # accl
                    if epoch % 2 == 1:
                        cross_loss = 1 * self.memory_rgb(f_out_ir, ir2rgb_labels.long())
                    else:
                        cross_loss = 1 * self.memory_ir(f_out_rgb, rgb2ir_labels.long())
                else:
                    cross_loss = self.memory_rgb(f_out_ir, ir2rgb_labels.long()) + self.memory_ir(f_out_rgb, rgb2ir_labels.long())
            else:
                cross_loss = torch.tensor(0.0)
            
            loss = loss_ir + loss_rgb + 0.25*cross_loss # total loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss ir {:.3f}\t'
                      'Loss rgb {:.3f}\t'
                      'Loss cross {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              losses.val, losses.avg, 
                              loss_ir, 
                              loss_rgb, 
                              cross_loss
                            ))

    def train_gmm(self, epoch, epoch_count, data_loader_ir, data_loader_rgb, optimizer, print_freq=10, train_iters=400, i2r=None, r2i=None, gmm_mode="easy", 
                  gmm_p_threshold_ir=0.5, gmm_p_threshold_rgb=0.5, sharpen_easy_tau=6, sharpen_diff_tau=0.6):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs_ir, labels_ir, indexes_ir = self._parse_data_ir(inputs_ir)
            inputs_rgb, inputs_rgb1, labels_rgb, indexes_rgb = self._parse_data_rgb(inputs_rgb)

            # forward
            inputs_rgb = torch.cat((inputs_rgb, inputs_rgb1), 0)
            labels_rgb = torch.cat((labels_rgb, labels_rgb), -1)
            _, f_out_rgb, f_out_ir, labels_rgb, labels_ir, pool_rgb, pool_ir = \
                self._forward(inputs_rgb, inputs_ir, label_1=labels_rgb, label_2=labels_ir, modal=0)

            # intra-modality nce loss (for easy and diff respectively)
            if gmm_mode=="easy":
                loss_ir = self.memory_ir(f_out_ir, labels_ir, loss='sace', param=self.sharpen(gmm_p_threshold_ir, sharpen_easy_tau)) 
                loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb, loss='sace', param=self.sharpen(gmm_p_threshold_rgb, sharpen_easy_tau))
            elif gmm_mode=="diff":
                loss_ir = self.memory_ir(f_out_ir, labels_ir, loss='sace', param=self.sharpen(gmm_p_threshold_ir, sharpen_diff_tau))
                loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb, loss='sace', param=self.sharpen(gmm_p_threshold_rgb, sharpen_diff_tau))
            else:
                raise ValueError("gmm_mode input error!")
            
            # cross contrastive learning
            if r2i:
                rgb2ir_labels = torch.tensor([r2i[key.item()] for key in labels_rgb]).cuda()
                ir2rgb_labels = torch.tensor([i2r[key.item()] for key in labels_ir]).cuda()
                alternate = False
                if alternate:
                    if epoch_count % 2 == 1:
                        cross_loss = 1 * self.memory_rgb(f_out_ir, ir2rgb_labels.long())
                    else:
                        cross_loss = 1 * self.memory_ir(f_out_rgb, rgb2ir_labels.long())
                else:
                    cross_loss = self.memory_rgb(f_out_ir, ir2rgb_labels.long()) + self.memory_ir(f_out_rgb, rgb2ir_labels.long())
            else:
                cross_loss = torch.tensor(0.0)
            
            loss = loss_ir + loss_rgb + 0.5*cross_loss # total loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.update(loss.item())

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss ir {:.3f}\t'
                      'Loss rgb {:.3f}\t'
                      'Loss cross {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              losses.val, losses.avg,
                              loss_ir,
                              loss_rgb,
                              cross_loss
                            ))

    def eval_train(self, data_loader, mode=0, dataset_name="RegDB"):
        self.encoder.train()
        if dataset_name=="RegDB" and mode == 1:
            losses = torch.zeros(2060)
        elif dataset_name=="RegDB" and mode == 2:
            losses = torch.zeros(2060)
        elif dataset_name=="SYSU-MM01" and mode == 1:
            losses = torch.zeros(22258)
        elif dataset_name=="SYSU-MM01" and mode == 2:
            losses = torch.zeros(11909)

        with torch.no_grad():
            for i, (imgs, fnames, pids, _, index) in enumerate(data_loader):
                inputs = imgs.cuda()
                pids = pids.cuda()

                _, f_out_rgb, f_out_ir, labels_rgb, labels_ir, pool_rgb, pool_ir = \
                    self._forward(inputs, inputs, label_1=pids, label_2=pids, modal=mode)

                if mode == 1:
                    loss = self.memory_rgb(f_out_rgb, labels_rgb, reduction='none')
                elif mode == 2:
                    loss = self.memory_ir(f_out_ir, labels_ir, reduction='none')
                else:
                    raise ValueError("Error mode!")

                for b in range(inputs.size(0)):
                    losses[index[b]] = loss[b]

        losses = (losses-losses.min())/(losses.max()-losses.min())
        input_loss = losses.reshape(-1, 1).cpu()

        # fit a two-component GMM to the loss
        gmm = GaussianMixture(n_components=2, max_iter=100, tol=1e-3, reg_covar=5e-4)
        gmm.fit(input_loss)
        prob = gmm.predict_proba(input_loss)
        prob = prob[:, gmm.means_.argmin()]
        
        p_threshold = np.sort(prob)[int(len(prob)*gmm.weights_[gmm.weights_.argmin()])]
        
        return prob, p_threshold
    
    def sharpen(self, x, tau):
        return np.log((x**0.25)/tau+1)

    def _parse_data_rgb(self, inputs):
        imgs, imgs1, _, pids, _, indexes = inputs
        return imgs.cuda(), imgs1.cuda(), pids.cuda(), indexes.cuda()

    def _parse_data_ir(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, x1, x2, label_1=None, label_2=None, modal=0):
        return self.encoder(x1, x2, label_1=label_1, label_2=label_2, modal=modal)


class OriTripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
    Args:
    - margin (float): margin for triplet.
    """
    
    def __init__(self, batch_size, margin=0.3):
        super(OriTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct