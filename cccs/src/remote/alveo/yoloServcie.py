from multiprocessing import Process
from src.utils.check import check_fifo
from src.utils.process import *

from src.proto.host import cmd_pb2_grpc, cmd_pb2

from src.remote.pipe import *

import cv2
import os
import time
import numpy as np
from PIL import Image
from multiprocessing import Manager, Queue

  
class Yolov3Sevice(cmd_pb2_grpc.Yolov3Sevice):
  def __init__(self):
    self.wrpath = "/tmp/fifo_recon_wr"; check_fifo(self.wrpath)
    self.rdpath = "/tmp/fifo_recon_rd"; check_fifo(self.rdpath)
    self.img_queue = Queue(QUEUE_SIZE)
    os.system("cd /workspace/tools/Vitis-AI-Library/yolov3/test && ./test_yolov3_server /workspace/data/sample_models/landslide_caffe_model/yolov4_landslide.xmodel {} {} &".format(self.wrpath, self.rdpath))

  def runyolov3(self, dat, context): # 多了个context参数
    img, width, height = dat.cameraData, dat.width, dat.height
    print("start write {}".format(self.wrpath))
    f = os.open(self.wrpath, os.O_WRONLY)
    for i in range(0, len(img), 1 << 16):
      os.write(f, img[i:i+(1 << 16)])
    os.close(f)

    with open(self.rdpath, "rb") as f:
      pos = f.read() 
    position = np.frombuffer(pos, dtype=np.int32)
    print("box num = {}".format(position[0]))
    res = cmd_pb2.YoloResponse()
    res.position.extend(position)
    return res
  
  def getImage(self, dat, context): # 多了个context参数
    return cmd_pb2.YoloImg()