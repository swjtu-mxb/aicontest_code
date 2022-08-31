import math
import numpy as np
import matplotlib.pyplot as plt
import paddle
import paddle.nn as nn
# from paddle.vision.datasets import MNIST,Cifar10
from paddle.io import Dataset
import warnings
warnings.filterwarnings("ignore")

from encoder import AutoEncoder
from util import *

import cv2
LR = 0.0005

test_dataset = TestDataset(data_dir='../Kodak/test/')

test_loader = paddle.io.DataLoader(dataset=test_dataset, shuffle=False, batch_size=1)

network = AutoEncoder()
optimizer = paddle.optimizer.Adam(parameters=network.parameters(), learning_rate=LR)

# load
layer_state_dict = paddle.load("./lambda1000/ae.pdparams")
opt_state_dict = paddle.load("./lambda1000/opt.pdopt")

network.set_state_dict(layer_state_dict)
optimizer.set_state_dict(opt_state_dict)

loss_func = nn.MSELoss()

# model.eval()
sumBpp = 0
sumPsnr = 0
cnt = 0
for inputs, filename in test_loader:
    mse_loss,bpp,_,result = network(inputs)
    psnr = 10 * (paddle.log(1. / mse_loss) / np.log(10))

    sumBpp += bpp
    sumPsnr += psnr
    cnt += 1

    print("Num: {}, name: {}, Bpp:{:.6f}, PSNR:{:.6f}".format(cnt, filename[0], bpp.item(), psnr.item()))
    fakepath = "recon/{}.png".format(os.path.basename(filename[0]).split('.')[0])
    # cv2.imwrite(fakepath, result*255.0)
    save_image(result, fakepath)

sumBpp /= cnt
sumPsnr /= cnt
print("Dataset Average result---Dataset Num: {}, Bpp:{:.6f}, PSNR:{:.6f}".format(cnt, sumBpp.item(), sumPsnr.item()))




