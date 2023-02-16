from pycocotools.cocoeval import COCOeval
import json
from tqdm import tqdm
from model.fcos import FCOSDetector
import mindspore.dataset.vision.py_transforms as py_vision
import os
import mindspore.ops as ops
from mindspore import context
import mindspore
import numpy as np
import cv2
from mindspore.context import ParallelMode
from PIL import Image
import argparse
import mindspore.common.dtype as mstype
from mindspore import Tensor
from pycocotools.coco import COCO
from mindspore.ops import stop_gradient
from model.eval_utils import post_process
from model.eval_utils import ClipBoxes

parser = argparse.ArgumentParser()
parser.add_argument("--data_url",type=str)
parser.add_argument("--train_url",type=str)
parser.add_argument("--label_url",type=str)
parser.add_argument("--pretrained_ckpt",type=str)
#parser.add_argument("--eval_ckpt",type=str)
opt = parser.parse_args()

class COCOGenerator:
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
    def __init__(self,dataset_dir,annotation_file,resize_size=[800,1333]):
        self.coco = COCO(annotation_file)
        self.root = dataset_dir
        ids = list(sorted(self.coco.imgs.keys()))
        print("INFO====>check annos, filtering invalid data......")
        new_ids=[]
        for id in ids:
            ann_id=self.coco.getAnnIds(imgIds=id,iscrowd=None)
            ann=self.coco.loadAnns(ann_id)
            if self._has_valid_annotation(ann):
                new_ids.append(id)
        self.ids=new_ids
        self.category2id = {v: i + 1 for i, v in enumerate(self.coco.getCatIds())}
        self.id2category = {v: k for k, v in self.category2id.items()}
        self.mean = [0.40789654, 0.44719302, 0.47026115]
        self.std = [0.28863828, 0.27408164, 0.27809835]
        self.resize_size=resize_size


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

    def __getitem__(self,index):
        
        img,ann=self.getImg(index)
        ann = [o for o in ann if o['iscrowd'] == 0]
        boxes = [o['bbox'] for o in ann]
        boxes=np.array(boxes,dtype=np.float32)
        #xywh-->xyxy
        boxes[...,2:]=boxes[...,2:]+boxes[...,:2]
        img=np.array(img)
        
        img,boxes,scale=self.preprocess_img_boxes(img,boxes,self.resize_size)
        # img=draw_bboxes(img,boxes)
        classes = [o['category_id'] for o in ann]
        classes = [self.category2id[c] for c in classes]
        to_tensor=py_vision.ToTensor()
        img=to_tensor(img)
        img= Tensor(img, dtype=mstype.float32)
        #img= c_transforms.Normalize(self.mean, self.std)(img)

        boxes = Tensor(boxes, dtype=mstype.float32)
        classes = Tensor(classes, dtype=mstype.int64)

        return img,boxes,classes,scale

    def __len__(self):
        return len(self.ids)

    def preprocess_img_boxes(self,image,boxes,input_ksize):
        '''
        resize image and bboxes 
        Returns
        image_paded: input_ksize  
        bboxes: [None,4]
        '''
        min_side, max_side    = input_ksize
        h,  w, _  = image.shape

        smallest_side = min(w,h)
        largest_side=max(w,h)
        scale=min_side/smallest_side
        if largest_side*scale>max_side:
            scale=max_side/largest_side
        nw, nh  = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))

        pad_w=32-nw%32
        pad_h=32-nh%32

        image_paded = np.zeros(shape=[nh+pad_h, nw+pad_w, 3],dtype=np.uint8)
        image_paded[:nh, :nw, :] = image_resized

        if boxes is None:
            return image_paded
        else:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale 
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale 
            return image_paded, boxes,scale



    def _has_only_empty_bbox(self,annot):
        return all(any(o <= 1 for o in obj['bbox'][2:]) for obj in annot)


    def _has_valid_annotation(self,annot):
        if len(annot) == 0:
            return False

        if self._has_only_empty_bbox(annot):
            return False

        return True
    

def evaluate_coco(generator, model, threshold=0.05):
    """ Use the pycocotools to evaluate a COCO model on a dataset.

    Args
        oU NMSgenerator : The generator for g
        model     : The model to evaluate.
        threshold : The score threshold to use.
    """
    # start collecting results
    results = []
    image_ids = []
    for index in tqdm(range(len(generator))):
        img,gt_boxes,gt_labels,scale = generator[index]
        # run network
        img= Tensor(img, mindspore.float32)
        expand_dims = ops.ExpandDims()
        img=expand_dims(img,0)
        scores,labels,boxes,batch_imgs= model(img,0,0)
        stack = ops.Stack(axis=0)
        scores = stack(scores)  # [batch_size,max_num]
        labels = stack(labels)  # [batch_size,max_num]
        boxes = stack(boxes)  # [batch_size,max_num,4]
        scores, labels, boxes=post_process((scores,labels,boxes),0.05,0.6)
        boxes = ClipBoxes(batch_imgs, boxes)
        scores = stop_gradient(scores)
        labels = stop_gradient(labels)
        boxes = stop_gradient(boxes)
        boxes /= scale
        # 校正图像比例框
        # 更改为（x、y、w、h）(MS COCO 标准)
        print(scores.shape)
        print(labels.shape)
        print(boxes.shape)
        sub = ops.Sub()
        boxes[ :, 2] = sub(boxes[ :, 2],boxes[ :, 0])
        boxes[ :, 3] =  sub(boxes[ :, 3],boxes[ :, 1])
        scores = scores.asnumpy()
        labels = labels.asnumpy()
        boxes= boxes.asnumpy()
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            
            # 分数已排序，因此我们可以跳出
            if score < threshold:
                break

            # append detection for each positively labeled class
            image_result = {
                'image_id'    : generator.ids[index],
                'category_id' : generator.id2category[label],
                'score'       : float(score),
                'bbox'        : box.tolist(),
            }
            # append detection to results
            results.append(image_result)
        # append image to list of processed images
        image_ids.append(generator.ids[index])
    if not len(results):
        return

    # write output
    json.dump(results, open('coco_bbox_results.json', 'w'), indent=4)
    coco_true = generator.coco
    coco_pred = coco_true.loadRes('coco_bbox_results.json')
    # run COCO evaluation
    coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats

if __name__ == "__main__":
    context.set_context(mode=context.PYNATIVE_MODE, device_target='Ascend', device_id=5)
    context.reset_auto_parallel_context()
    parallel_mode = ParallelMode.STAND_ALONE
    context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=1)
    generator = COCOGenerator("/data/coco/val", "/data/coco/annotations/annotations/instances_val.json")
    model=FCOSDetector(mode="inference")
    model.set_train(False)
    
    mindspore.load_param_into_net(model, mindspore.load_checkpoint(r"/data2/wzj/ckpt_0/wongs-2_2584.ckpt"))
    evaluate_coco(generator,model)
