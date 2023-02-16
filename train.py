from mindspore import context, Tensor
from mindspore.context import ParallelMode
from mindspore.nn import TrainOneStepCell
from mindspore.train.callback import TimeMonitor, CheckpointConfig, ModelCheckpoint, LossMonitor,SummaryCollector

from dataset.cuhk import CUHKDataset
from model.config import DefaultConfig
from model.fcos import FCOSDetector
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

mindspore.dataset.config.set_seed(1)
mindspore.common.seed.set_seed(1)
set_seed(1)

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=24, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")
parser.add_argument("--platform", type=str, default='Ascend', help="run platform")
parser.add_argument("--device_num", type=int, default='1', help="device_number to run")
parser.add_argument("--device_id", type=int, default='0', help="DEVICE_ID to run ")
opt = parser.parse_args()



if __name__ == '__main__':
    if opt.platform == 'Ascend':
        # device_num = int(os.environ.get("DEVICE_NUM", 1))
        # device_id = int(os.getenv('DEVICE_ID'))
        device_num = opt.device_num
        device_id = opt.device_id

    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', device_id=device_id)
    if device_num > 1:
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)



    dataset_dir = '/data/dataset/CUHK'
    annotation_file = '/data/dataset/CUHK/annotations/train.json'
    BATCH_SIZE = opt.batch_size
    EPOCHS = opt.epochs
    
    tr = Transforms()
    train_dataset, dataset_size = CUHKDataset.create_dataset(dataset_dir, annotation_file, BATCH_SIZE,
                                                                   shuffle=True, transform=tr)
    print("the size of the dataset is %d" % train_dataset.get_dataset_size())

    steps_per_epoch = dataset_size//BATCH_SIZE
    TOTAL_STEPS = steps_per_epoch * EPOCHS
    #WARMUP_STEPS = 10
    WARMUP_STEPS=500
    WARMUP_FACTOR = 1.0 / 3.0
    GLOBAL_STEPS = 0
    LR_INIT = 0.01
    lr_schedule = [120000, 160000]

    def lr_func(LR_INIT,WARMUP_STEPS,WARMUP_FACTOR,TOTAL_STEPS,lr_schedule):
        lr_res = []
        step = 0

        for i in range(0,TOTAL_STEPS):

            lr = LR_INIT

            if step < WARMUP_STEPS:
                alpha = float(step) / WARMUP_STEPS
                warmup_factor = WARMUP_FACTOR * (1.0 - alpha) + alpha
                lr = lr * warmup_factor
                lr_res.append(lr)


            else:
                for i in range(len(lr_schedule)):
                    if step < lr_schedule[i]:
                        lr_res.append(lr)
                        break
                    lr *= 0.1
                if step >= 160000:
                    lr_res.append(lr)

            step+=1

        return np.array(lr_res,dtype=np.float32)

    print("start set model...")
    fcos = FCOSDetector(mode="training")

    time_cb = TimeMonitor(data_size=dataset_size)
    loss_cb = LossMonitor()
    summary_collector = SummaryCollector(summary_dir='./wang_summary_dir', collect_freq=1)
    cb = [summary_collector,time_cb, loss_cb]
    if DefaultConfig.save_checkpoint:
        ckptconfig = CheckpointConfig(save_checkpoint_steps=50, keep_checkpoint_max=15000)
        save_checkpoint_path = os.path.join(DefaultConfig.save_checkpoint_path, "ckpt_0" + "/")
        ckpt_cb = ModelCheckpoint(prefix='wongs_new',directory=save_checkpoint_path, config=ckptconfig)
        cb += [ckpt_cb]

    lr = Tensor(lr_func(LR_INIT,WARMUP_STEPS,WARMUP_FACTOR,TOTAL_STEPS,lr_schedule))
    
    sgd_optimizer = nn.SGD(fcos.trainable_params(), learning_rate=lr, momentum=0.9, weight_decay=0.0001)

    model = Model(network=fcos, optimizer=sgd_optimizer)

    print("successfully build model, and now train the model...")
    


    model.train(EPOCHS, train_dataset=train_dataset, dataset_sink_mode=True,callbacks=cb)
