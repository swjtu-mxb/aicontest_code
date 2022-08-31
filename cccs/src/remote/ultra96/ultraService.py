#-*- coding:utf-8 -*-
import os
import time
import numpy as np
from multiprocessing import Queue
from src.utils.check import check_fifo
from src.config.cfg import QUEUE_SIZE, PORT

from src.config.cfg import ULTRA_96_PORT

import time

# class Yolov3Sevice(cmd_pb2_grpc.Yolov3Sevice):
#   def __init__(self):
#     self.wrpath = "/tmp/fifo_recon_wr"; check_fifo(self.wrpath)
#     self.rdpath = "/tmp/fifo_recon_rd"; check_fifo(self.rdpath)
#     self.img_queue = Queue(QUEUE_SIZE)
#     os.system("cd /workspace/tools/Vitis-AI-Library/yolov3/test && ./test_yolov3_server /workspace/data/sample_models/landslide_caffe_model/yolov4_landslide.xmodel {} {} &".format(self.wrpath, self.rdpath))

#   def runyolov3(self, dat, context): # 多了个context参数
#     img, width, height = dat.cameraData, dat.width, dat.height
#     print("start write {}".format(self.wrpath))
#     f = os.open(self.wrpath, os.O_WRONLY)
#     for i in range(0, len(img), 1 << 16):
#       os.write(f, img[i:i+(1 << 16)])
#     os.close(f)

#     with open(self.rdpath, "rb") as f:
#       pos = f.read() 
#     position = np.frombuffer(pos, dtype=np.int32)
#     print("box num = {}".format(position[0]))

#     img_arr = np.frombuffer(img, dtype=np.uint8).reshape([height, width, 3])
#     self.result = [draw(img_arr, position).tobytes(), height, width]

#     res = cmd_pb2.YoloResponse()
#     res.position.extend(position)
#     return res
  
#   def getImage(self, dat, context): # 多了个context参数
#     return cmd_pb2.YoloImg(data=self.result[0], height=self.result[1], width=self.result[2])

# def serve():
#   # 1. 开始server 
#   server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
#   # 2. 把service加入到server监控中 
#   cmd_pb2_grpc.add_Yolov3SeviceServicer_to_server(Yolov3Sevice(), server)
#   # 3. 指定端口
#   server.add_insecure_port(ULTRA_96_PORT)
#   server.start()
#   try:
#     while True:
#       time.sleep(.6)
#   except  KeyboardInterrupt:
#     server.stop(0)
 
from rpyc import Service
import numpy as np
from multiprocessing import Manager
from rpyc.utils.server import ThreadedServer

class UltraService(Service):
  def __init__(self):
    super().__init__()
    self.wrpath = "/tmp/fifo_recon_wr"; check_fifo(self.wrpath)
    self.rdpath = "/tmp/fifo_recon_rd"; check_fifo(self.rdpath)
    os.system("cd /home/root/Vitis-AI/demo/Vitis-AI-Library/samples/yolov3 && ./test_yolov3_server /home/root/landslide_ultra96_caffe_model/yolov4_landslide.xmodel {} {} &".format(self.wrpath, self.rdpath))

  def exposed_runyolov3(self, dat): # 多了个context参数
    img = bytearray(dat)
    print("start write {}".format(self.wrpath))
    f = os.open(self.wrpath, os.O_WRONLY)
    for i in range(0, len(img), 1 << 16):
      os.write(f, img[i:i+(1 << 16)])
    os.close(f)
    with open(self.rdpath, "rb") as f:
      pos = f.read() 
    position = np.frombuffer(pos, dtype=np.int32)
    print("box num = {}".format(position[0]))
    return position

def serve():
  server = ThreadedServer(service=UltraService, port = (int)(PORT),auto_register=False) 
  server.start()
  try:
    while True:
      time.sleep(.6)
  except  KeyboardInterrupt:
    server.stop(0)
 
if __name__ == "__main__":
  serve()
  