import multiprocessing
import os

import mindspore
import numpy as np
import cv2
from PIL import Image
import random
import mindspore.common.dtype as mstype
import mindspore.dataset as de
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as cv
from mindspore.dataset import DistributedSampler
from pycocotools.coco import COCO
from mindspore.dataset.vision import Inter
from mindspore.dataset import py_transforms
import mindspore.dataset.vision.py_transforms as py_vision
from dataset.augment import Transforms
import mindspore.ops as ops
import numpy as np

PATH_DATASET = r'E:\dataset\coco\val2017'
PATH_ANNO = r'E:\dataset\coco\annotations'

H = []
W = []
B = []
def flip(img, boxes):
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    w = img.width
    if boxes.shape[0] != 0:
        xmin = w - boxes[:, 2]
        xmax = w - boxes[:, 0]
        boxes[:, 2] = xmax
        boxes[:, 0] = xmin
    return img, boxes


class COCODataset:
    CLASSES_NAME = (
        '__back_ground__', 'person', 'bicycle', 'car', 'motorcycle',
        'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
        'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle',
        'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli',
        'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet',
        'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush')

    def __init__(self, dataset_dir, annotation_file, resize_size=[800, 1333], is_train=True, transform=None):
        self.coco = COCO(annotation_file)
        self.root = dataset_dir
        ids = list(sorted(self.coco.imgs.keys()))
        print("INFO====>check annos, filtering invalid data......")
        new_ids = []
        for id in ids:
            ann_id = self.coco.getAnnIds(imgIds=id, iscrowd=None)
            ann = self.coco.loadAnns(ann_id)
            if self._has_valid_annotation(ann):
                new_ids.append(id)
        self.ids = new_ids

        self.category2id = {v: i + 1 for i, v in enumerate(self.coco.getCatIds())}
        self.id2category = {v: k for k, v in self.category2id.items()}

        self.transform = transform
        self.resize_size = resize_size

        self.train = is_train

    def getImg(self, index):
        img_id = self.ids[index]
        coco = self.coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        # if self.transform is not None:
        #     img, target = self.transform(img, target)

        return img, target

    def __getitem__(self, index):

        img, ann = self.getImg(index)

        ann = [o for o in ann if o['iscrowd'] == 0]
        boxes = [o['bbox'] for o in ann]
        boxes = np.array(boxes, dtype=np.float32)
        # xywh-->xyxy
        boxes[..., 2:] = boxes[..., 2:] + boxes[..., :2]

        if self.train:
            if random.random() < 0.5:
                img, boxes = flip(img, boxes)
            if self.transform is not None:
                img, boxes = self.transform(img, boxes)

        img = np.array(img)

        img, boxes = self.preprocess_img_boxes(img, boxes, self.resize_size)
        # img=draw_bboxes(img,boxes)

        classes = [o['category_id'] for o in ann]
        classes = [self.category2id[c] for c in classes]

        #####img = np.array(img)
        to_tensor = py_vision.ToTensor()
        img = to_tensor(img)
        #    img = Tensor(img,dtype=mstype.float32)

        # img= transforms.Normalize(self.mean, self.std,inplace=True)(img)
        ###### boxes = np.array(boxes, dtype=mstype.float32)
        ###### classes = np.array(classes, dtype=mstype.int64)

        #   boxes = Tensor(boxes,dtype=mstype.float32)
        #  classes = Tensor(classes, dtype=mstype.int64)

        # return img,boxes,classes
        max_h = 1344
        max_w = 1344
        max_num = 90
        img = np.pad(img, ((0, 0), (max(int(max_h - img.shape[1]), 0), 0), (0, max(int(max_w - img.shape[2]), 0))))
        boxes = np.pad(boxes,((0,max(max_num-boxes.shape[0],0)),(0,0)))
        classes = np.pad(classes, (0, max(max_num - len(classes), 0)))
      #  print("test in dataset",img.shape,boxes.shape,classes.shape)
        return img, boxes, classes

    def __len__(self):
        return len(self.ids)

    def preprocess_img_boxes(self, image, boxes, input_ksize):
        '''
        resize image and bboxes
        Returns
        image_paded: input_ksize
        bboxes: [None,4]
        '''
        min_side, max_side = input_ksize
        h, w, _ = image.shape

        smallest_side = min(w, h)
        largest_side = max(w, h)
        scale = min_side / smallest_side
        if largest_side * scale > max_side:
            scale = max_side / largest_side
        nw, nh = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))

        pad_w = 32 - nw % 32
        pad_h = 32 - nh % 32

        image_paded = np.zeros(shape=[nh + pad_h, nw + pad_w, 3], dtype=np.uint8)
        image_paded[:nh, :nw, :] = image_resized

        if boxes is None:
            return image_paded
        else:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale
            return image_paded, boxes

    def _has_only_empty_bbox(self, annot):
        return all(any(o <= 1 for o in obj['bbox'][2:]) for obj in annot)

    def _has_valid_annotation(self, annot):
        if len(annot) == 0:
            return False

        if self._has_only_empty_bbox(annot):
            return False

        return True


def collate_fn(imgs, boxes, classes, batchInfo):
    imgs_list = imgs
    boxes_list = boxes
    classes_list = classes
    assert len(imgs_list) == len(boxes_list) == len(classes_list)
    batch_size = len(imgs_list)
    pad_imgs_list = []
    pad_boxes_list = []
    pad_classes_list = []

    mean = [0.40789654, 0.44719302, 0.47026115]
    std = [0.28863828, 0.27408164, 0.27809835]
    # h_list = [int(s.shape[0]) for s in imgs_list]
    # w_list = [int(s.shape[1]) for s in imgs_list]

    #BCHW
    # h_list = [int(s.shape[1]) for s in imgs_list]
    # w_list = [int(s.shape[2]) for s in imgs_list]
    # max_h = np.array(h_list).max()
    # max_w = np.array(w_list).max()
    #
    # H.append(max_h)
    # W.append(max_w)

    max_h = 1344
    max_w = 1344
    for i in range(batch_size):
        img = imgs_list[i]

        pad_img = np.pad(img, ((0, 0), (max(int(max_h - img.shape[1]),0), 0), (0, max(int(max_w - img.shape[2]),0))))
        normalize_img = py_vision.Normalize(mean, std)(pad_img)
        pad_imgs_list.append(normalize_img)

    # max_num = 0
    # for i in range(batch_size):
    #     n = boxes_list[i].shape[0]
    #     if n > max_num: max_num = n
    max_num = 90
    for i in range(batch_size):
        pad_boxes_list.append(cv.Pad((0, 0, 0, max(max_num - boxes_list[i].shape[0],0)), 0)(boxes_list[i]))
        pad_classes_list.append(np.pad(classes_list[i], (0, max(max_num - classes_list[i].shape[0],0))))

    # return batch_imgs, batch_boxes, batch_classes
    batch_boxes = np.stack(pad_boxes_list)
    batch_classes = np.stack(pad_classes_list)
    batch_imgs = np.stack(pad_imgs_list)



    return batch_imgs, batch_boxes, batch_classes

    # return Tensor(batch_imgs), Tensor(batch_boxes), Tensor(batch_classes)


# def collate_fn(imgs, boxes, classes, batchInfo):
#     imgs_list = imgs
#     boxes_list = boxes
#     classes_list = classes
#
#     print("len:",len(imgs_list),"type of list:",type(imgs_list),"image:",imgs_list[0],"type of image:",type(imgs_list[0]),"shape:",imgs_list[0].shape)
#     return Tensor([[1],[1],[1],[1]]).asnumpy(), Tensor([[1],[1],[1],[1]]).asnumpy(), Tensor([[1],[1],[1],[1]]).asnumpy()


def create_coco_dataset(dataset_dir, annotation_file, batch_size, shuffle=True, transform=None):
    cv2.setNumThreads(0)

    dataset = COCODataset(dataset_dir, annotation_file, is_train=True, transform=transform)
    HWC2CHW = cv.HWC2CHW()
    dataset_column_names = ["img", "boxes", "class"]

    ds = de.GeneratorDataset(dataset, column_names=dataset_column_names, shuffle=shuffle)
    # ds = ds.map(operations=HWC2CHW, input_columns=["img"], num_parallel_workers=8)
    #ds = ds.batch(batch_size, per_batch_map=collate_fn, input_columns=dataset_column_names, drop_remainder=True)
    ds = ds.batch(batch_size, drop_remainder=True)

    return ds, len(dataset)


if __name__ == "__main__":

    dataset_dir = PATH_DATASET
    annotation_file = PATH_ANNO + r'/instances_val2017.json'
    tr = Transforms()
    dataset, size = create_coco_dataset(dataset_dir, annotation_file, 4, transform=None, shuffle=False)
    iterator1 = dataset.create_tuple_iterator(output_numpy=True)

    count = 0
    for i,data in enumerate(iterator1):
        # print("image:",item["img"],"image.shape:",item["img"].shape,"image.type:",type(item["img"]))
        # print("boxes:", item["boxes"], "boxes.shape:", item["boxes"].shape, "boxes.type:", type(item["boxes"]))
        # print("class:", item["class"], "class.shape:", item["class"].shape, "class.type:", type(item["class"]))
        print(data[0].shape)
        print(data[1].shape)
        print(data[2].shape)
        print(data[0])
        print(data[1])
        print(data[2])
        if i>4:
            break

#####################################################3

# dataset_dir = PATH_DATASET
# annotation_file = PATH_ANNO+ r'/instances_val2017.json'
# tr = Transforms()
# dataset, size = create_coco_dataset(dataset_dir, annotation_file, 1,transform=tr,shuffle=False)
# iterator1 = dataset.create_dict_iterator()
#
# imgs = []
# boxes = []
# classes = []
#
# count = 0
# for item in iterator1:
#     # print("image:",item["img"],"image.shape:",item["img"].shape,"image.type:",type(item["img"]))
#     # print("boxes:", item["boxes"], "boxes.shape:", item["boxes"].shape, "boxes.type:", type(item["boxes"]))
#     # print("class:", item["class"], "class.shape:", item["class"].shape, "class.type:", type(item["class"]))
#     imgs.append(item["img"].asnumpy().squeeze(axis=0))
#     boxes.append(item["boxes"].asnumpy().squeeze(axis=0))
#     classes.append(item["class"].asnumpy().squeeze(axis=0))
#     count +=1
#     if count>4:
#         break
#
# x,y,z = collate_fn(imgs,boxes,classes,batchInfo="")
# print(x[0][0].shape)
# np.savetxt("./result.txt", x[0][0], fmt='%f')
##########################################################################################################


#########################################################################################

#  img,boxes,classes=dataset[0]
#  print(img,boxes,classes,"\n",img.shape,boxes.shape,classes.shape,boxes.dtype,classes.dtype,img.dtype)
#  print("type:", type(img), type(boxes), type(classes))
# # cv2.imwrite("./123.jpg",img)
#  img, boxes, classes = dataset.collate_fn([dataset[0], dataset[1], dataset[2]])
#  print(img.shape, boxes.shape, classes.shape, boxes.dtype, classes.dtype, img.dtype)
#########################################################################################