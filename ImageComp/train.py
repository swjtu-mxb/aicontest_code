import math
import numpy as np
import matplotlib.pyplot as plt
import paddle
# paddle.disable_static()
import paddle.nn as nn
# from paddle.vision.datasets import MNIST,Cifar10
from paddle.io import Dataset
import warnings
warnings.filterwarnings("ignore")

from encoder import AutoEncoder
from util import *
from PIL import Image

import cv2

train_data_dir = '../Kodak/train/'
image_size = 256
train_dataset = Datasets(train_data_dir, image_size)
train_loader = paddle.io.DataLoader(train_dataset, batch_size=4, shuffle=True)

network = AutoEncoder()
# model = paddle.Model(network)


paddle.framework.seed(1)

EPOCH = 1000
BATCH_SIZE = 4
LR = 0.0005
N_TEST_IMG = 10
train_lambda = 1000

optimizer = paddle.optimizer.Adam(parameters=network.parameters(), learning_rate=LR)
# loss_func = nn.MSELoss()


step_loss_list = []
network.train()

for epoch in range(EPOCH):
    for x in train_loader:
        
        mse_loss, bpp, encoded, decoded = network(x)
        # loss = loss_func(decoded, train_x)
        loss = train_lambda * mse_loss + bpp
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        step_loss_list.append(loss.numpy())
    if (epoch + 1) % 5 == 0:
        print('Epoch: %2d' % (epoch + 1), '| train loss: %.4f' % np.mean(step_loss_list), "mse: %.6f" % mse_loss, "bpp: %.6f" % bpp)


paddle.save(network.state_dict(), "./lambda1000/ae.pdparams")
paddle.save(optimizer.state_dict(), "./lambda1000/opt.pdopt")

# model.save('./ckpt/ae')
# loss_func = nn.MSELoss()
# test_dataset = TestDataset(data_dir='./kodak/test/')
# test_loader = paddle.io.DataLoader(dataset=test_dataset, shuffle=False, batch_size=1)


# # @paddle.no_grad()
# for _, (inputs, filename) in enumerate(test_loader):
#     _,_,_,result = network(inputs)
    
#     mse_loss = loss_func(result, inputs)

#     fakepath = "recon/{}.png".format(os.path.basename(filename[0]).split('.')[0])
#     save_image(result, fakepath)

# place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()