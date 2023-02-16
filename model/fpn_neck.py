import mindspore.nn as nn
from mindspore import ops
import mindspore.common.initializer as weight_init
from mindspore.ops import ResizeNearestNeighbor
import mindspore.nn as nn
from mindspore import ops
import mindspore.common.initializer as weight_init
from mindspore.ops import ResizeNearestNeighbor
import mindspore
from mindspore.common.initializer import initializer, HeUniform
from mindspore.common import dtype as mstype

class FPN(nn.Cell):
    '''only for resnet50,101,152'''

    def __init__(self, features=256, use_p5=True):
        super(FPN, self).__init__()
      #  shape = (out_channels, in_channels, kernel_size, kernel_size)

        self.prj_5 = nn.Conv2d(2048, features, kernel_size=1,has_bias=True,pad_mode='valid',weight_init = initializer(HeUniform(1), shape=(features,2048,1,1), dtype=mstype.float32).init_data())
        self.prj_4 = nn.Conv2d(1024, features, kernel_size=1,has_bias=True,pad_mode='valid',weight_init = initializer(HeUniform(1), shape=(features,1024,1,1), dtype=mstype.float32).init_data())
        self.prj_3 = nn.Conv2d(512, features, kernel_size=1,has_bias=True,pad_mode='valid',weight_init = initializer(HeUniform(1), shape=(features,512,1,1), dtype=mstype.float32).init_data())
        self.conv_5 = nn.Conv2d(features, features, kernel_size=3, pad_mode='pad', padding=1,has_bias=True,weight_init = initializer(HeUniform(1), shape=(features,features,3,3), dtype=mstype.float32).init_data())
        self.conv_4 = nn.Conv2d(features, features, kernel_size=3, pad_mode='pad', padding=1,has_bias=True,weight_init = initializer(HeUniform(1), shape=(features,features,3,3), dtype=mstype.float32).init_data())
        self.conv_3 = nn.Conv2d(features, features, kernel_size=3, pad_mode='pad', padding=1,has_bias=True,weight_init = initializer(HeUniform(1), shape=(features,features,3,3), dtype=mstype.float32).init_data())
        if use_p5:
            self.conv_out6 = nn.Conv2d(features, features, kernel_size=3, pad_mode='pad', padding=1, stride=2,has_bias=True,weight_init = initializer(HeUniform(1), shape=(features,features,3,3), dtype=mstype.float32).init_data())
        else:
            self.conv_out6 = nn.Conv2d(2048, features, kernel_size=3, pad_mode='pad', padding=1, stride=2,has_bias=True,weight_init = initializer(HeUniform(1), shape=(features,2048,3,3), dtype=mstype.float32).init_data())
        self.conv_out7 = nn.Conv2d(features, features, kernel_size=3, pad_mode='pad', padding=1, stride=2,has_bias=True,weight_init = initializer(HeUniform(1), shape=(features,features,3,3), dtype=mstype.float32).init_data())
        self.use_p5 = use_p5
        # if isinstance(self, nn.Conv2d):
        #     self.weight.set_data(weight_init.initializer(weight_init.HeUniform(), self.weight.shape, self.weight.dtype))
        #     if self.bias is not None:
        #         self.weight.set_data(
        #             weight_init.initializer(weight_init.Constant([0, self.bias]), self.weight.shape, self.weight.dtype))
        constant_init = mindspore.common.initializer.Constant(0)
        constant_init(self.prj_5.bias)
        constant_init(self.prj_4.bias)
        constant_init(self.prj_3.bias)
        constant_init(self.conv_5.bias)
        constant_init(self.conv_4.bias)
        constant_init(self.conv_3.bias)
        constant_init(self.conv_out6.bias)
        constant_init(self.conv_out7.bias)

    def upsamplelike(self, inputs):
        src, target = inputs
        resize = ResizeNearestNeighbor((target.shape[2], target.shape[3]))
        return resize(src)

    def construct(self, x):
        C3, C4, C5 = x
        P5 = self.prj_5(C5)
        P4 = self.prj_4(C4)
        P3 = self.prj_3(C3)

        P4 = P4 + self.upsamplelike((P5, C4))
        P3 = P3 + self.upsamplelike((P4, C3))

        P3 = self.conv_3(P3)
        P4 = self.conv_4(P4)
        P5 = self.conv_5(P5)

        P5 = P5 if self.use_p5 else C5
        P6 = self.conv_out6(P5)
        relu = ops.ReLU()
        P7 = self.conv_out7(relu(P6))
        return (P3, P4, P5, P6, P7)