# AlignPS描述

## 概述

AlignPS[论文地址参见](https://openaccess.thecvf.com/content/CVPR2021/papers/Yan_Anchor-Free_Person_Search_CVPR_2021_paper.pdf)是CVPR2021的中稿工作，论文全名为Anchor-Free Person Search。其模型没有anchor box，在检测分支，AlignPS通过在卷积阶段加入FPN、在计算损失阶段加入centerness这一分支，实现了更高的检测精度。最终模型可以达到CUHK-SYSU数据集上95.1%的mAP精度。该论文的[PyTorch实现可参考](https://github.com/daodaofr/AlignPS)。

如下为MindSpore使用CUHK-SYSU数据集对AlignPS进行训练的示例。

## 性能

|Dataset|Model|mAP|Rank1|
|-----|-----|------|-----|
|CUHK-SYSU|AlignPS| 93.1%|93.4%|
|CUHK-SYSU|AlignPS+|94.0%|94.5%|[cfg](https://github.com/daodaofr/AlignPS/blob/master/configs/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_4x4_1x_cuhk_reid_1500_stage1_fpncat_dcn_epoch24_multiscale_focal_x4_bg-2_lconv3dcn_sub_triqueue.py)| [model](https://drive.google.com/file/d/12AuG37IPkhyrpHG_kqpUzzoDEEkXlgne/view?usp=sharing)| 
|PRW|AlignPS| 45.9%|81.9%|[cfg](https://github.com/daodaofr/AlignPS/blob/master/configs/fcos/prw_base_focal_labelnorm_sub_ldcn_fg15_wd1-3.py)| [model](https://drive.google.com/file/d/1QQNoYQTiO3FIiEpu0AtigGFIDf3wG2u5/view?usp=sharing)| 
|PRW|AlignPS+|46.1%|82.1%|[cfg](https://github.com/daodaofr/AlignPS/blob/master/configs/fcos/prw_dcn_base_focal_labelnorm_sub_ldcn_fg15_wd7-4.py)| [model](https://drive.google.com/file/d/1O02EBrHglE1x-zk88QLLdXF-x6yebwBp/view?usp=sharing)| 

## 数据集

使用的数据集：[CUHK-SYSU](https://github.com/ShuangLI59/person_search)和[PRW](https://github.com/liangzheng06/PRW-baseline)

全部下载好后，我们提供了COCO格式的[标注文件](https://github.com/daodaofr/AlignPS/tree/master/demo/anno)

对于CUHK-SYSU数据集，可以在配置文件的第30，31行修改数据集和标注的地址

## 环境要求

  - 硬件
    - 准备Ascend处理器搭建硬件环境。
  - 框架
    - [MindSpore](https://www.mindspore.cn/install/en)，本模型编写时版本为r1.2，12.30更新由r1.5编写的版本。
  - 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

## 脚本及样例代码

```
└──AlignPS
    ├── README.md
    ├── checkpoint                               # 训练时的checkpoint存储 
    ├── dataset                                  # 数据集处理
      ├── augment.py                             # 对数据集进行随机扩充、平移等操作
      ├── cuhk.py                                # CUHK-SYSU数据集
    ├── model
      ├── backbone                               # 骨干网络
        ├── resnet.py                            # 骨干网络ResNet
      ├── config.py                              # 参数配置
      ├── eval_utils.py                          # 评估时需要用到的工具
      ├── alignps.py                             # AlignPS模型网络
      ├── fpn_neck.py                            # FPN
      ├── fcos_reid_head_focal_sub_triqueue3.py  # 模型的bbox head和reid head
      └── loss.py                                # 检测分支所使用的生成目标框和loss损失
    ├── test_images                              # 测试图像
    ├── analyze_fail_0.dat                       # 运行报错时提及的dat文件       
    └── train.py                                 # 训练网络
```

## 脚本参数

在config.py中可以同时配置训练参数和评估参数。

```
    #backbone
    pretrained=True
    freeze_stage_1=True
    freeze_bn=True

    #fpn
    fpn_out_channels=256
    use_p5=True
    
    #head
    class_num=80
    use_GN_head=True
    prior=0.01
    add_centerness=True
    cnt_on_reg=True

    #training
    strides=[8,16,32,64,128]
    limit_range=[[-1,64],[64,128],[128,256],[256,512],[512,999999]]

    #inference
    score_threshold=0.05
    nms_iou_threshold=0.6
    max_detection_boxes_num=1000
```

## 训练过程

### 运行流程   

数据集：CUHK-SYSU/PRW      
预训练模型：resnet50    
启动文件:train.py       

eval运行说明：       
运行操作过程：    
1.得到训练好的ckpt    
2.修改运行说明中的参数    
3.运行eval.py文件    

