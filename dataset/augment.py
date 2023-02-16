import math, random
import sys

import mindspore
import numpy as np
from PIL import Image
import random
import mindspore.dataset.vision.py_transforms as transforms
import mindspore.ops as ops
from mindspore import Tensor


class Transforms(object):
    def __init__(self):
        pass

    # def __call__(self, img, boxes):
    #     if random.random() < 0.3:
    #         img, boxes = colorJitter(img, boxes)
    #     if random.random() < 0.5:
    #         img, boxes = random_rotation(img, boxes)
    #     if random.random() < 0.5:
    #         img, boxes = random_crop_resize(img, boxes)
    #     return img, boxes

    def __call__(self, img, boxes):
        if random.random() < 0.3:
            img, boxes = colorJitter(img, boxes)
        if random.random() < 0.5:
            img, boxes = random_rotation(img, boxes)
        if random.random() < 0.5:
            img, boxes = random_crop_resize(img, boxes)
        # return img,boxes
        return img, np.array(boxes)




def colorJitter(img, boxes, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1):
    img = transforms.RandomColorAdjust(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)(img)
    return img, boxes


def random_rotation(img, boxes, degree=10):
    d = random.uniform(-degree, degree)
    w, h = img.size
    rx0, ry0 = w / 2.0, h / 2.0
    img = img.rotate(d)
    a = -d / 180.0 * math.pi
   # boxes = Tensor(boxes)
    new_boxes = np.zeros_like(boxes)
    new_boxes[:, 0] = boxes[:, 1]
    new_boxes[:, 1] = boxes[:, 0]
    new_boxes[:, 2] = boxes[:, 3]
    new_boxes[:, 3] = boxes[:, 2]
    for i in range(boxes.shape[0]):
        ymin, xmin, ymax, xmax = new_boxes[i, :]
        xmin, ymin, xmax, ymax = float(xmin), float(ymin), float(xmax), float(ymax)
        # xmin, ymin, xmax, ymax = float(xmin.asnumpy()), float(ymin.asnumpy()), float(xmax.asnumpy()), float(ymax.asnumpy())
        x0, y0 = xmin, ymin
        x1, y1 = xmin, ymax
        x2, y2 = xmax, ymin
        x3, y3 = xmax, ymax
      #  z = Tensor([[y0, x0], [y1, x1], [y2, x2], [y3, x3]], mindspore.float32)
        z = np.array([[y0, x0], [y1, x1], [y2, x2], [y3, x3]], dtype=np.float32)
        tp = np.zeros_like(z)
        tp[:, 1] = (z[:, 1] - rx0) * math.cos(a) - (z[:, 0] - ry0) * math.sin(a) + rx0
        tp[:, 0] = (z[:, 1] - rx0) * math.sin(a) + (z[:, 0] - ry0) * math.cos(a) + ry0

        ##########################################################1
        # ymax, xmax = ops.ArgMaxWithValue(axis=0)(tp)[1]
        # ymin, xmin = ops.ArgMinWithValue(axis=0)(tp)[1]
        ymax, xmax = np.max(tp,axis=0)
        ymin, xmin = np.min(tp,axis=0)
        #########################################################

        # new_boxes[i] = ops.Stack()([ymin, xmin, ymax, xmax])
        new_boxes[i] = np.stack([ymin, xmin, ymax, xmax])
    # new_boxes[:, 1::2] = ops.clip_by_value(new_boxes[:, 1::2], Tensor(0), Tensor(w - 1))
    # new_boxes[:, 0::2] = ops.clip_by_value(new_boxes[:, 0::2], Tensor(0), Tensor(h - 1))
    # new_boxes[:, 1::2] = ops.clip_by_value(new_boxes[:, 1::2], 0, w - 1)
    # new_boxes[:, 0::2] = ops.clip_by_value(new_boxes[:, 0::2], 0, h - 1)
    new_boxes[:, 1::2] = np.clip(new_boxes[:, 1::2], 0, w - 1)
    new_boxes[:, 0::2] = np.clip(new_boxes[:, 0::2], 0, h - 1)
    boxes[:, 0] = new_boxes[:, 1]
    boxes[:, 1] = new_boxes[:, 0]
    boxes[:, 2] = new_boxes[:, 3]
    boxes[:, 3] = new_boxes[:, 2]
    return img, boxes

##?????#####################2
def _box_inter(box1, box2):
    tl = np.maximum(box1[:, None, :2], box2[:, :2])  # [n,m,2]
    br = np.minimum(box1[:, None, 2:], box2[:, 2:])  # [n,m,2]
   # tl = np.maximum(box1.asnumpy()[:, None, :2], box2.asnumpy()[:, :2])  # [n,m,2]
   # br = np.minimum(box1.asnumpy()[:, None, 2:], box2.asnumpy()[:, 2:])  # [n,m,2]
    inter_tensor = np.array((br-tl),dtype=np.float32)
    hw = np.clip(inter_tensor, 0, sys.maxsize)  # [n,m,2]
    inter = hw[:, :, 0] * hw[:, :, 1]  # [n,m]
    return inter
############################2


def random_crop_resize(img, boxes, crop_scale_min=0.2, aspect_ratio=[3. / 4, 4. / 3], remain_min=0.7, attempt_max=10):
    success = False
   # boxes = Tensor(boxes)
    for attempt in range(attempt_max):
        # choose crop size
        area = img.size[0] * img.size[1]
        target_area = random.uniform(crop_scale_min, 1.0) * area
        aspect_ratio_ = random.uniform(aspect_ratio[0], aspect_ratio[1])
        w = int(round(math.sqrt(target_area * aspect_ratio_)))
        h = int(round(math.sqrt(target_area / aspect_ratio_)))
        if random.random() < 0.5:
            w, h = h, w
        # if size is right then random crop
        if w <= img.size[0] and h <= img.size[1]:
            x = random.randint(0, img.size[0] - w)
            y = random.randint(0, img.size[1] - h)
            # check
            crop_box = np.array([[x, y, x + w, y + h]], dtype=np.float32)
            inter = _box_inter(crop_box, boxes)  # [1,N] N can be zero
            box_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])  # [N]
            mask = inter > 0.0001  # [1,N] N can be zero
            # inter = inter[mask]  # [1,S] S can be zero
            # inter = Tensor(inter.asnumpy()[mask.asnumpy()], mindspore.float32).squeeze()  # [1,S] S can be zero
            inter = np.array(inter[mask], dtype=np.float32).squeeze()  # [1,S] S can be zero
            #######box_area = box_area[mask]  # [S]

            # box_area = box_area[mask.view(-1)]  # [S]
           # box_area = np.array(box_area[mask.view(-1)],dtype=np.float32)  # [S]
            box_area = np.array(box_area[mask.reshape(-1)], dtype=np.float32)  # [S]
            ##### box_remain = inter / box_area  # [S]

            #box_remain = inter.view(-1) / box_area    # [S]
            box_remain = inter.reshape(-1) / box_area  # [S]


            if box_remain.shape[0] != 0:
                t = box_remain>remain_min
                myflag = True
                for i in box_remain > remain_min:
                    if i == False:
                        myflag = False
                        break

                if bool(myflag):
                    success = True
                    break
            else:
                success = True
                break




    if success:
        img = img.crop((x, y, x + w, y + h))
        boxes -= np.array([x, y, x, y])
        # boxes[:, 1::2] = ops.clip_by_value(boxes[:, 1::2], Tensor(0), Tensor(h - 1))
        # boxes[:, 0::2] = ops.clip_by_value(boxes[:, 0::2], Tensor(0), Tensor(w - 1))
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, h - 1)
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, w - 1)

    return img, boxes
