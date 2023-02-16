import argparse
import os
import sys
import mindspore
import numpy as np
from mindspore import Tensor, context
from mindspore.ops import count_nonzero

import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as cv

from model.head import ClsCntRegHead
from model.fpn_neck import FPN
from model.backbone.resnet import resnet50,resnet101,resnet152
import mindspore.nn as nn
from model.loss import GenTargets, LOSS, coords_fmap2orig, GradNetWrtX
from model.config import DefaultConfig
import mindspore.ops as ops
from labeled_matching_layer_queue import LabeledMatching,LabeledMatchingLayerQueue
from unlabeled_matching_layer import UnlabeledMatchingLayer 
##TRAIN
from mindspore import context, Tensor
from mindspore.context import ParallelMode
from mindspore.nn import TrainOneStepCell
from mindspore.train.callback import TimeMonitor, CheckpointConfig, ModelCheckpoint, LossMonitor,SummaryCollector

from dataset import COCO_dataset
from model.config import DefaultConfig
from dataset.COCO_dataset import COCODataset
from dataset.augment import Transforms
import os
import argparse
import mindspore
from mindspore import Model
import mindspore.nn as nn
from mindspore.common import set_seed
import numpy as np
from model.loss import LOSS
from person_search_roi_head_2input1 import PersonSearchRoIHead2Input1
from fcos_reid_head_focal_sub_triqueue3 import FCOSReidHeadFocalSubTriQueue3
import traceback

class AlignPS(nn.Cell):

    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = DefaultConfig
        self.backbone = resnet50(pretrained=config.pretrained)
        self.fpn = FPN(config.fpn_out_channels, use_p5=config.use_p5)
        self.fpn = FPN(config.fpn_out_channels, use_p5=config.use_p5)
        self.head = ClsCntRegHead(config.fpn_out_channels, config.class_num,
                                  config.use_GN_head, config.cnt_on_reg, config.prior)
        self.config = config
        self.roi_head = PersonSearchRoIHead2Input1()

        self.bbox_head = FCOSReidHeadFocalSubTriQueue3()

        self.init_weights(pretrained=config.pretrained)

        self.loss_oim = OIMLoss()
        num_person = 5532
        queue_size = 5000
        self.labeled_matching_layer = LabeledMatchingLayerQueue(num_persons=num_person, feat_len=256) # for mot17half
        self.unlabeled_matching_layer = UnlabeledMatchingLayer(queue_size=queue_size, feat_len=256)
        self.loss_tri = TripletLossFilter()

        # self.mi_estimator = CLUBSample(256, 256, 512)
        # self.mi_estimator = CLUB()
        self.mi_estimator = MINE(256, 256, 256)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        xb = self.backbone(img)
        if self.with_neck:
            xn = self.neck(xb)
        #for xx in xb:
        #    print(xx.shape)
        #    print(xb[2].shape)
        return [xb[2]], xn

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_ids,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):

        xb, xn = self.extract_feat(img)
        #print("here", xb.shape)

        losses = dict()

        # RPN forward and loss
        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(xb)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_metas)
            rpn_losses = self.rpn_head.loss(*rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get("rpn_proposal", self.test_cfg.rpn)
            proposal_list = self.rpn_head.get_bboxes(*rpn_outs, img_metas, cfg=proposal_cfg)
        else:
            proposal_list = proposals

        roi_losses, feats_pids_roi = self.roi_head.forward_train(xb, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels, gt_ids,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)
        # oim_loss = dict()
        # oim_loss["loss_oim_roi"] = self.loss_oim(feats_pids_roi["bbox_feats"], feats_pids_roi["gt_pids"])
        # losses.update(oim_loss)


        single_losses, feats_pids = self.bbox_head.forward_train(xn, img_metas, gt_bboxes,
                                              gt_labels, gt_ids, gt_bboxes_ignore)

        ####### calculate mutual information ############
        feats_roi = feats_pids_roi["bbox_feats"]
        pids_roi = feats_pids_roi["gt_pids"]
        feats_fcos = feats_pids["pos_reid"]
        pids_fcos = feats_pids["pos_reid_ids"]
        dic1 = dict(list)
        dic2 = dict(list)

        for i in range(len(pids_roi)):
            if pids_roi[i] < 0:
                continue
            else:
                targets1_value = pids_roi[i].cpu().numpy().item()
                dic1[targets1_value].append(feats_roi[i])
        
        for i in range(len(pids_fcos)):
            if pids_fcos[i] < 0:
                continue
            else:
                targets2_value = pids_fcos[i].cpu().numpy().item()
                dic2[targets2_value].append(feats_fcos[i])
        
        all_feats1 = []
        all_feats2 = []
        for key, val in dic1.items():
            if key in dic2:
                val2 = dic2[key]
                feat1 = sum(val)/len(val)
                # print(feat1.shape)
                mean1 = mindspore.ops.normalize(feat1.unsqueeze(0))
                # mean1 = feat1.unsqueeze(0)
                feat2 = sum(val2)/len(val2)
                mean2 = mindspore.ops.normalize(feat2.unsqueeze(0))
                # mean2 = feat2.unsqueeze(0)
                all_feats1.append(mean1)
                all_feats2.append(mean2)

        if len(all_feats1) > 0 and len(all_feats2) >0:
            all_feats1 = mindspore.cat(all_feats1)
            all_feats2 = mindspore.cat(all_feats2)
            # print(all_feats1.shape, all_feats2.shape)

            all_feats1_d = all_feats1.detach()
            all_feats2_d = all_feats2.detach()
            mi_loss = dict()
            if mindspore.randint(1, 100, (1,)) % 3:
                self.mi_estimator.train()
                mi_loss["loss_mi"] = 0.2 * self.mi_estimator.learning_loss(all_feats1_d, all_feats2_d)
            else:
                self.mi_estimator.eval()
                # mi_loss["loss_mi_bound"] = self.mi_estimator(all_feats1, all_feats2)
                mi_loss["loss_mi_bound"] = 0.2 * self.mi_estimator.learning_loss(all_feats1, all_feats2)

            losses.update(mi_loss)

        # losses.update(single_losses)
        for key, val in single_losses.items():
            if key in losses:
                #print("losses", key, losses[key], losses[key].shape)
                #print("val", val, val.shape)
                losses[key] += val
            else:
                losses[key] = val

        return losses

class TripletLossFilter(nn.Cell):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3):
        super(TripletLossFilter, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Does not calculate noise inputs with label -1
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        #print(inputs.shape, targets.shape)
        inputs_new = []
        targets_new = []
        targets_value = []
        for i in range(len(targets)):
            if targets[i] == -1:
                continue
            else:
                inputs_new.append(inputs[i])
                targets_new.append(targets[i])
                targets_value.append(targets[i].cpu().numpy().item())
        if len(set(targets_value)) < 2:
            tmp_loss = mindspore.ops.zeros(1)
            tmp_loss = tmp_loss[0]
            tmp_loss = tmp_loss.to(targets.device)
            return tmp_loss
        #print(targets_value)
        inputs_new = mindspore.ops.stack(inputs_new)
        targets_new = mindspore.ops.stack(targets_new)
        #print(inputs_new.shape, targets_new.shape)
        n = inputs_new.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = mindspore.ops.pow(inputs_new, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs_new, inputs_new.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        #print("Triplet ", dist)
        # For each anchor, find the hardest positive and negative
        mask = targets_new.expand(n, n).eq(targets_new.expand(n, n).t())
        #print(mask)
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max())
            dist_an.append(dist[i][mask[i] == 0].min())
        dist_ap = mindspore.ops.stack(dist_ap)
        dist_an = mindspore.ops.stack(dist_an)
        # Compute ranking hinge loss
        y = mindspore.ops.ones_like(dist_an)
        #y = dist_an.data.new()
        #y.resize_as_(dist_an.data)
        #y.fill_(1)
        #y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss

class OIM():
    def forward(ctx, inputs, targets, lut, cq, header, momentum):
        ctx.save_for_backward(inputs, targets, lut, cq, header, momentum)
        outputs_labeled = inputs.mm(lut.t())
        outputs_unlabeled = inputs.mm(cq.t())
        return mindspore.ops.cat([outputs_labeled, outputs_unlabeled], dim=1)

    def backward(ctx, grad_outputs):
        inputs, targets, lut, cq, header, momentum = ctx.saved_tensors

        # inputs, targets = tensor_gather((inputs, targets))

        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(mindspore.ops.cat([lut, cq], dim=0))
            if grad_inputs.dtype == mindspore.dtype.float16:
                grad_inputs = grad_inputs.to(mindspore.dtype.float32)

        for x, y in zip(inputs, targets):
            if y >= 0:
                lut[y] = momentum * lut[y] + (1.0 - momentum) * x
                lut[y] /= lut[y].norm()
            else:
                cq[header] = x
                header = (header + 1) % cq.size(0)
        return grad_inputs, None, None, None, None, None


def oim(inputs, targets, lut, cq, header, momentum=0.5):
    return OIM.apply(inputs, targets, lut, cq, mindspore.tensor(header), mindspore.tensor(momentum))


class OIMLoss(nn.Cell):
    def __init__(self, num_features=256, num_pids=5532, num_cq_size=5000, oim_momentum=0.5, oim_scalar=30):
        super(OIMLoss, self).__init__()
        self.num_features = num_features
        self.num_pids = num_pids
        self.num_unlabeled = num_cq_size
        self.momentum = oim_momentum
        self.oim_scalar = oim_scalar

        self.register_buffer("lut", mindspore.ops.zeros(self.num_pids, self.num_features))
        self.register_buffer("cq", mindspore.ops.zeros(self.num_unlabeled, self.num_features))

        self.header_cq = 0

    def forward(self, inputs, roi_label):


        inds = roi_label >= -1
        label = roi_label[inds]
        inputs = inputs[inds.unsqueeze(1).expand_as(inputs)].view(-1, self.num_features)

        projected = oim(inputs, label, self.lut, self.cq, self.header_cq, momentum=self.momentum)
        projected *= self.oim_scalar

        self.header_cq = (
            self.header_cq + (label == -1).long().sum().item()
        ) % self.num_unlabeled
        loss_oim = mindspore.ops.cross_entropy(projected, label, ignore_index=-1)
        return loss_oim

class MINE(nn.Cell):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(MINE, self).__init__()
        self.T_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1))
    
    def forward(self, x_samples, y_samples):  # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]
        random_index = mindspore.randint(sample_size, (sample_size,)).long()

        y_shuffle = y_samples[random_index]

        T0 = self.T_func(mindspore.ops.cat([x_samples,y_samples], dim = -1))
        T1 = self.T_func(mindspore.ops.cat([x_samples,y_shuffle], dim = -1))

        lower_bound = T0.mean() - mindspore.log(T1.exp().mean())

        # compute the negative loss (maximise loss == minimise -loss)
        return lower_bound
    
    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples)