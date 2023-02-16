import os
import sys
import datetime
import stat
import mindspore
from collections import defaultdict
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from mindspore.train.callback import Callback
from mindspore import log as logger, ops, Tensor
from mindspore import save_checkpoint


def post_process(preds_topk,score_threshold,nms_iou_threshold):
    '''
    cls_scores_topk [batch_size,max_num]
    cls_classes_topk [batch_size,max_num]
    boxes_topk [batch_size,max_num,4]
    '''
    _cls_scores_post = []
    _cls_classes_post = []
    _boxes_post = []
    cls_scores_topk, cls_classes_topk, boxes_topk = preds_topk

    for batch in range(cls_classes_topk.shape[0]):
        mask = cls_scores_topk[batch] >= score_threshold
        mask_select = ops.MaskedSelect()
        _cls_scores_b = mask_select(cls_scores_topk[batch], mask)  # [?]
        _cls_scores_b = np.squeeze(_cls_scores_b)
        _cls_classes_b = mask_select(cls_classes_topk[batch], mask)  # [?]
        _cls_classes_b = np.squeeze(_cls_classes_b)
        # print("这里是cls.shape")
        # print(_cls_classes_b.shape)
        expand_dims = ops.ExpandDims()
        mask = expand_dims(mask, 2)
        op = ops.Concat(2)
        mask = op((mask, mask, mask, mask))
        _boxes_b = mask_select(boxes_topk[batch], mask)  # [?,4]
        _boxes_b = _boxes_b.reshape(-1, 4)
        nms_ind = batched_nms(_boxes_b, _cls_scores_b, _cls_classes_b,nms_iou_threshold)
        _cls_scores_post.append(_cls_scores_b[nms_ind])
        _cls_classes_post.append(_cls_classes_b[nms_ind])
        _boxes_post.append(_boxes_b[nms_ind])
    stack = ops.Stack(axis=0)
    scores, classes, boxes = stack(_cls_scores_post), stack(_cls_classes_post), stack(_boxes_post)
    return scores, classes, boxes
def batched_nms(boxes, scores, idxs, iou_threshold):
        if ops.Size()(boxes) == 0:
            return Tensor(0,mindspore.int32)
        # strategy: in order to perform NMS independently per class.
        # we add an offset to all the boxes. The offset is dependent
        # only on the class idx, and is large enough so that boxes
        # from different classes do not overlap
        max = ops.ArgMaxWithValue()
        reshape = ops.Reshape()
        squeeze = ops.Squeeze()
        boxes2=reshape(boxes,(-1,1))
        index,max_coordinate = max(boxes2)
        max_coordinate=squeeze(max_coordinate)
        offsets = idxs * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]
        keep = box_nms(boxes_for_nms, scores, iou_threshold)
        return keep


def box_nms(boxes, scores, thr):
    '''
    boxes: [?,4]
    scores: [?]
    '''
    if boxes.shape[0] == 0:
        return ops.Zeros(0, mindspore.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    sort = ops.Sort(0, descending=True)
    index, order = sort(scores)
    keep = []
    cast = ops.Cast()
    while ops.Size()(order) > 1:
        i = order[0].asnumpy().item(0)
        keep.append(i)
        order_1 = cast(order[1:], mindspore.int32)
        min_value = 0
        max_value = 100000
        xmin = Tensor(x1[order_1]).clip(x1[i], max_value)
        ymin = Tensor(y1[order_1]).clip(y1[i], max_value)
        xmax = Tensor(x2[order_1]).clip(min_value, x2[i])
        ymax = Tensor(y2[order_1]).clip(min_value, y2[i])
        inter = Tensor((xmax - xmin)).clip(0, max_value) * Tensor((ymax - ymin)).clip(0, max_value)
        iou = inter / (areas[i] + areas[order_1] - inter)
        idx = iou <= thr
        idx2 = mindspore.numpy.arange(ops.Size()(idx))
        idx = ops.MaskedSelect()(idx2, idx) + 1
        order = order[idx]
    if ops.Size()(order) == 1:
        i = order[0].asnumpy().item(0)
        keep.append(i)
    return Tensor(keep,mindspore.int32)
def ClipBoxes( batch_imgs, batch_boxes):
    batch_boxes = ops.clip_by_value(batch_boxes, Tensor(0, mindspore.float32),
                                    Tensor(9999999, mindspore.float32))
    h, w = batch_imgs.shape[2:]
    batch_boxes[..., [0, 2]] = ops.clip_by_value(batch_boxes[..., [0, 2]], Tensor(0, mindspore.float32), w - 1)
    batch_boxes[..., [1, 3]] = ops.clip_by_value(batch_boxes[..., [1, 3]], Tensor(0, mindspore.float32), h - 1)
    return batch_boxes