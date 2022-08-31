# from .basics import *
# import pickle
# import os
# import codecs
import paddle.nn as nn
import paddle
import paddle.nn.functional as F

class Bitparm(nn.Layer):
    '''
    save params
    '''
    def __init__(self, channel, final=False):
        super(Bitparm, self).__init__()
        self.final = final
        self.h = paddle.create_parameter(paddle.empty([1,channel,1,1]).shape,dtype='float32')
        self.b = paddle.create_parameter(paddle.empty([1,channel,1,1]).shape,dtype='float32')

        # self.h = paddle.normal(0, 0.01, self.h.shape)
        # self.b = paddle.normal(0, 0.01, paddle.create_parameter(paddle.empty([1,channel,1,1]).shape,dtype='float32').shape)
       
        if not final:
            # self.a = paddle.normal(0, 0.01, paddle.create_parameter(paddle.empty([1,channel,1,1]).shape,dtype='float32').shape)
            self.a = paddle.create_parameter(paddle.empty([1,channel,1,1]).shape,dtype='float32')

        else:
            self.a = None

    def forward(self, x):
        if self.final:
            return F.sigmoid(x * F.softplus(self.h) + self.b)
        else:
            x = x * F.softplus(self.h) + self.b
            return x + paddle.tanh(x) * paddle.tanh(self.a)

class BitEstimator(nn.Layer):
    '''
    Estimate bit
    '''
    def __init__(self, channel):
        super(BitEstimator, self).__init__()
        self.f1 = Bitparm(channel)
        self.f2 = Bitparm(channel)
        self.f3 = Bitparm(channel)
        self.f4 = Bitparm(channel, True)
        
    def forward(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        return self.f4(x)


def build_model():
    z = paddle.zeros([4,16,32,32])
    model = BitEstimator(channel=16)
    x = model(z)

if __name__ == "__main__":
    build_model()