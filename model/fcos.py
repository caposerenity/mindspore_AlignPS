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
import logging
import traceback




class FCOS(nn.Cell):

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

    def train(self, mode=True):
        """
        set module training mode, and frozen bn
        """
        super().train(mode=True)

        def freeze_bn(module):
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
            classname = module.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in module.parameters(): p.requires_grad = False

        if self.config.freeze_bn:
            self.apply(freeze_bn)
            # print("INFO===>success frozen BN")
        if self.config.freeze_stage_1:
            self.backbone.freeze_stages(1)
            # print("INFO===>success frozen backbone stage1")

    def construct(self, x):
        """
        Returns
        list [cls_logits,cnt_logits,reg_preds]
        cls_logits  list contains five [batch_size,class_num,h,w]
        cnt_logits  list contains five [batch_size,1,h,w]
        reg_preds   list contains five [batch_size,4,h,w]
        """
        # print("test in fcos body,input type(x),shape(x):",type(x),x.shape)
        C3, C4, C5 = self.backbone(x)
        # print("test in body,C3,C4,C5 out type and shape",type(C3),type(C4),type(C5),C3.shape,C4.shape,C5.shape)
        # print("************************************")
        all_P = self.fpn((C3, C4, C5))
        # print("test in body,ALL_P out type and len and p3 p4 p5 p6 p7",type(all_P),len(all_P),all_P[0].shape,all_P[1].shape,all_P[2].shape,all_P[3].shape,all_P[4].shape)
        # print("____________________________________")
        cls_logits, cnt_logits, reg_preds = self.head(all_P)
        # print("test in body,cls type and shape",type(cls_logits),type(cnt_logits),type(reg_preds))
        # print("test in body,cls_logits[0] type and shape",type(cls_logits[0]),cls_logits[0].shape)
        # print(". . . . . . . . . . . . . . . . . . .")
        return (cls_logits, cnt_logits, reg_preds)


class DetectHead(nn.Cell):
    def __init__(self, score_threshold, nms_iou_threshold, max_detection_boxes_num, strides, config=None):
        super().__init__()
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detection_boxes_num = max_detection_boxes_num
        self.strides = strides
        if config is None:
            self.config = DefaultConfig
        else:
            self.config = config

    def construct(self, inputs):
        '''
        inputs  list [cls_logits,cnt_logits,reg_preds]
        cls_logits  list contains five [batch_size,class_num,h,w]
        cnt_logits  list contains five [batch_size,1,h,w]
        reg_preds   list contains five [batch_size,4,h,w]
        '''

        
        cast = ops.Cast()
        cls_logits, coords = self._reshape_cat_out(inputs[0], self.strides)  # [batch_size,sum(_h*_w),class_num]
        cnt_logits, _ = self._reshape_cat_out(inputs[1], self.strides)  # [batch_size,sum(_h*_w),1]
        reg_preds, _ = self._reshape_cat_out(inputs[2], self.strides)  # [batch_size,sum(_h*_w),4]\
        
        sigmoid = ops.Sigmoid()

        cls_preds = sigmoid(cls_logits)
        cnt_preds = sigmoid(cnt_logits)

        cls_classes, cls_scores = ops.ArgMaxWithValue(axis=-1)(cls_preds)  # [batch_size,sum(_h*_w)]
        
           
        cnt_preds = ops.Squeeze(axis=-1)(cnt_preds)
        cls_scores = ops.Sqrt()(cls_scores * cnt_preds)
        cls_classes = cls_classes + 1  # [batch_size,sum(_h*_w)]

        boxes = self._coords2boxes(coords, reg_preds)  # [batch_size,sum(_h*_w),4]
        if self.max_detection_boxes_num > cls_scores.shape[-1]:
            max_num = cls_scores.shape[-1]
        else:
            max_num = self.max_detection_boxes_num
        topk = ops.TopK(sorted=True)
        topk_ind = topk(cls_scores, max_num)[1]  # [batch_size,max_num]

        _cls_scores = ()
        _cls_classes = ()
        _boxes = ()
        
        for batch in range(cls_scores.shape[0]):
            topk_index = cast(topk_ind, mindspore.int32)
            _cls_scores=   _cls_scores+  ( cls_scores[batch][topk_index] ,) # [max_num]
            _cls_classes=  _cls_classes+  (cls_classes[batch][topk_index],)  # [max_num]
            _boxes=_boxes+(boxes[batch][topk_index],)  # [max_num,4]
        
        return _cls_scores,_cls_classes,_boxes
        #return self._post_process([cls_scores_topk, cls_classes_topk, boxes_topk])

    def _coords2boxes(self, coords, offsets):
        '''
        Args
        coords [sum(_h*_w),2]
        offsets [batch_size,sum(_h*_w),4] ltrb
        '''
        x1y1 = coords[None, :, :] - offsets[..., :2]
        x2y2 = coords[None, :, :] + offsets[..., 2:]  # [batch_size,sum(_h*_w),2]
        concat = ops.Concat(axis=-1)
        boxes = concat((x1y1, x2y2))  # [batch_size,sum(_h*_w),4]
        return boxes
   
    def _reshape_cat_out(self, inputs, strides):
        '''
        Args
        inputs: list contains five [batch_size,c,_h,_w]
        Returns
        out [batch_size,sum(_h*_w),c]
        coords [sum(_h*_w),2]
        '''
        batch_size = inputs[0].shape[0]
        c = inputs[0].shape[1]
        out =()
        coords = ()
        reshape = ops.Reshape()
        transpose = ops.Transpose()
        for pred, stride in zip(inputs, strides):
            input_perm = (0, 2, 3,1)
            pred = transpose(pred,input_perm)
            coord = coords_fmap2orig(pred, stride)
            pred = reshape(pred, (batch_size, -1, c))
            out=out+(pred,)
            coords=coords+(coord,)
        return ops.Concat(axis=1)(out), ops.Concat(axis=0)(coords)




class FCOSDetector(nn.Cell):
    def __init__(self, mode, config=None):
        super().__init__()

        config = DefaultConfig
        self.mode = mode
        self.fcos_body = FCOS(config=config)
        if mode == "training":
            self.target_layer = GenTargets(strides=config.strides, limit_range=config.limit_range)
            self.loss_layer = LOSS()
        elif mode == "inference":
            self.detection_head = DetectHead(config.score_threshold, config.nms_iou_threshold,
                                             config.max_detection_boxes_num, config.strides, config)


    def construct(self, input_imgs, input_boxes, input_classes):
        '''
        inputs 
        [training] list  batch_imgs,batch_boxes,batch_classes
        [inference] img
        '''
        if self.mode == "training":
          batch_imgs = input_imgs
          batch_boxes = input_boxes
          batch_classes = input_classes
          #print("test in fcos construct:(shape(batch_imgs),shape(batch_boxes),shape(batch_classes))",batch_imgs.shape,batch_boxes.shape,batch_classes.shape)
          # batch_imgs = np.transpose(batch_imgs, (0, 3, 1, 2))
       
          #s = time.time()
          out = self.fcos_body(batch_imgs)
          #e = time.time()
          #print("test in fcos construct of out:",e-s)
          
          #s = time.time()
          targets = self.target_layer((out, batch_boxes, batch_classes))
          #e = time.time()
          #print("test in fcos construct of targets:",e-s)
          
          #s = time.time()
          losses = self.loss_layer((out, targets))
          #e = time.time()
          #print("test in loss:(loss)",losses[-1],e-s)
          #print("out:",out)
          #print("targets:",targets)
          return losses[-1]
        else:
          batch_imgs = input_imgs 
            
          out = self.fcos_body(batch_imgs)
          
          scores,classes,boxes = self.detection_head(out)
          
         # boxes = self.clip_boxes(batch_imgs, boxes)
          return  scores, classes, boxes,batch_imgs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", type=str, default='CPU', help="run platform")
    opt = parser.parse_args()
    if opt.platform == 'CPU':
        device_num = int(os.environ.get("DEVICE_NUM", 1))
        context.set_context(mode=context.PYNATIVE_MODE,device_target='CPU',device_id=0)
    input_img = Tensor(np.ones((3, 832, 1216, 3)), dtype=mindspore.float32)

    x = np.transpose(input_img, (0, 3, 1, 2))
    input_box = Tensor(np.ones((3, 8, 4)), dtype=mindspore.float32)
    input_cls = Tensor(np.ones((3, 8)), dtype=mindspore.float32)

    print('layer1:resnet50...')
    layer1 = resnet50(pretrained=True)
    x1 = layer1(x)

    print('layer2:FPN...')
    layer2 = FPN()
    x2 = layer2(x1)

    print('layer3:head...')
    layer3 = ClsCntRegHead(256, 80, True, True, 0.01)
    x3 = layer3(x2)

    print('layer4:generate target box...')
    layer4 = GenTargets(strides=[8,16,32,64,128], limit_range=[[-1,64],[64,128],[128,256],[256,512],[512,999999]])
    x4 = layer4([x3,input_box,input_cls])
    print('layer5:calculate loss...')
    layer5 = LOSS()
    x = layer5([x3, x4])
    print(x)




# PATH_DATASET = r'E:\dataset\coco\val2017'
# PATH_ANNO = r'E:\dataset\coco\annotations'
#
#
# set_seed(1)
#
# parser = argparse.ArgumentParser()
# parser.add_argument("--epochs", type=int, default=1, help="number of epochs")
# parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")
# parser.add_argument("--platform", type=str, default='CPU', help="run platform")
# parser.add_argument("--device_num", type=int, default='1', help="device_number to run")
# parser.add_argument("--device_id", type=int, default='0', help="DEVICE_ID to run ")
# opt = parser.parse_args()
#
# #####Trick#########
#
# ###################
#
#
# if __name__ == '__main__':
#     if opt.platform == 'CPU':
#         # device_num = int(os.environ.get("DEVICE_NUM", 1))
#         # device_id = int(os.getenv('DEVICE_ID'))
#         device_num = opt.device_num
#         device_id = opt.device_id
#
#     context.set_context(mode=context.PYNATIVE_MODE, device_target='CPU', device_id=device_id)
#     if device_num > 1:
#         context.reset_auto_parallel_context()
#         context.set_auto_parallel_context(device_num=device_num,
#                                           parallel_mode=ParallelMode.DATA_PARALLEL,
#                                           gradients_mean=True)
#
#     dataset_dir = PATH_DATASET
#     annotation_file = PATH_ANNO + r'/instances_val2017.json'
#     BATCH_SIZE = opt.batch_size
#     EPOCHS = opt.epochs
#
#     tr = Transforms()
#     train_dataset, dataset_size = COCO_dataset.create_coco_dataset(dataset_dir, annotation_file, BATCH_SIZE,
#                                                                    shuffle=True, transform=tr)
#     print("the size of the dataset is %d" % train_dataset.get_dataset_size())
#
#     steps_per_epoch = dataset_size // BATCH_SIZE
#     TOTAL_STEPS = steps_per_epoch * EPOCHS
#     # WARMUP_STEPS = 10
#     WARMUP_STEPS = 500
#     WARMUP_FACTOR = 1.0 / 3.0
#     GLOBAL_STEPS = 0
#     LR_INIT = 0.01
#     lr_schedule = [120000, 160000]
#
#
#     ########Trick##############
#     def lr_func(LR_INIT, WARMUP_STEPS, WARMUP_FACTOR, TOTAL_STEPS, lr_schedule):
#         lr_res = []
#         step = 0
#
#         for i in range(0, TOTAL_STEPS):
#
#             lr = LR_INIT
#
#             if step < WARMUP_STEPS:
#                 alpha = float(step) / WARMUP_STEPS
#                 warmup_factor = WARMUP_FACTOR * (1.0 - alpha) + alpha
#                 lr = lr * warmup_factor
#                 lr_res.append(lr)
#
#
#             else:
#                 for i in range(len(lr_schedule)):
#                     if step < lr_schedule[i]:
#                         lr_res.append(lr)
#                         break
#                     lr *= 0.1
#                 if step >= 160000:
#                     lr_res.append(lr)
#
#             step += 1
#
#         return np.array(lr_res, dtype=np.float32)
#
#
#     ##############################
#
#     print("start set model...")
#     fcos = FCOSDetector(mode="training")
#     # momentum = nn.Momentum(fcos.trainable_params(), learning_rate=LR_INIT, momentum=0.9, weight_decay=1e-4)
#
#     time_cb = TimeMonitor(data_size=dataset_size)
#     loss_cb = LossMonitor()
#     summary_collector = SummaryCollector(summary_dir='./wang_summary_dir', collect_freq=1)
#     cb = [summary_collector, time_cb, loss_cb]
#     if DefaultConfig.save_checkpoint:
#         ckptconfig = CheckpointConfig(save_checkpoint_steps=50, keep_checkpoint_max=15000)
#         save_checkpoint_path = os.path.join(DefaultConfig.save_checkpoint_path, "ckpt_0" + "/")
#         ckpt_cb = ModelCheckpoint(prefix='wongs_new', directory=save_checkpoint_path, config=ckptconfig)
#         cb += [ckpt_cb]
#
#     lr = Tensor(lr_func(LR_INIT, WARMUP_STEPS, WARMUP_FACTOR, TOTAL_STEPS, lr_schedule))
#
#     print("test in train coco.py: lr.shape and total_steps and lr[0],lr[15999],lr[total_steps-1]", lr.shape,
#           TOTAL_STEPS, lr[0])
#     sgd_optimizer = nn.SGD(fcos.trainable_params(), learning_rate=lr, momentum=0.9, weight_decay=0.0001)
#
#     model = Model(network=fcos, optimizer=sgd_optimizer)
#
#     print("successfully build model, and now train the model...")
#
#     model.train(EPOCHS, train_dataset=train_dataset, callbacks=cb)
#


