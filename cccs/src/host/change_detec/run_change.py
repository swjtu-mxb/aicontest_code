from src.utils.process import *
import multiprocessing
from multiprocessing import Queue
import time
import random
from src.utils.check import check_fifo

import os
import cv2
import numpy as np

import torch
import src.host.change_detec.vitis_model as cdp
import albumentations as A
from PIL import Image

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def predict(image_id, image_B, model):
    
    test_transform = A.Compose([
                A.Normalize()])
    
    path1 = './datas/images/A/test_' + str(image_id) + '.png'
    img1 = cv2.imread(path1)
    img1 = test_transform(image = img1)['image'].transpose(2, 0, 1)
    img1 = torch.Tensor(np.expand_dims(img1,0)).to(DEVICE)
    
    # path2 = './datas/images/B/test_' + str(image_id) + '.png'
    # img2 = cv2.imread(path2)
    # RGB => BGR
    img2 = image_B[:, :, ::-1]
    img2 = test_transform(image = img2)['image'].transpose(2, 0, 1)
    img2 = torch.Tensor(np.expand_dims(img2,0)).to(DEVICE)
    
    pre = model(img1,img2)
    pre = torch.argmax(pre, dim=1).cpu().data.numpy()
    pre = pre * 255.0
    path3 = './datas/images/result/test_' + str(image_id) + '_pre.png'
    cv2.imwrite(path3, pre[0])
    # img = cv2.merge([pre[0]])
    # cv2.imshow("change detect", img)
    img = pre[0].astype(np.uint8)
    return img
    # img.show("change detect")
   
    # img1 = cv2.imread(path3)
    # cv2.imshow('result of change detection', img1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def send():
  q:Queue = getToQueue("receive")
  for i in range(10):
    data = np.random.randint(0, 100, [1024,1024,3]).astype(np.int8)
    id = i
    q.put((i, data))
    time.sleep(.6)

def receive():
  q:Queue = getFromQueue("send")
  for _ in range(10):
    id, image = q.get(True)
    predict(id, image)
  
def write():
  pass

if __name__ == "__main__":
  # receive 依赖于 send
  send = AppProcess("send", target=send) 
  rec = AppProcess("receive", target=receive)
  rec.fromProcess("send")
  # 将依赖和被依赖进程建立联系
  
  build()
  # 启动所有进程
  allStart()
  # 等待所有进程over
  allJoin()
  