from src.remote.compress.runcompress import CompressService
from src.proto.host import cmd_pb2_grpc 
from src.config.cfg import ULTRA_96_IP, PORT, REMOTE_PORT

import grpc
from concurrent import futures
import time
import numpy as np
import rpyc

def init_local_server():
  try:
    stub = rpyc.connect(ULTRA_96_IP, PORT)
  except Exception as e:
    print(e)
    print("can not connect ultra96")
    stub = None

  # 1. 开始server 
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
  # 2. 把service加入到server监控中 
  cmpSer = CompressService(stub)
  cmd_pb2_grpc.add_CompressServiceServicer_to_server(cmpSer, server)
  # 3. 指定端口
  server.add_insecure_port(REMOTE_PORT)
  start_process()

  server.start()
  # cmpSer.img_queue.put((readImage("datas/images/B/test_1.png"), "test_1"))
  # cmpSer.img_queue.put((readImage("datas/images/testImage/cat_resize.jpg"), "cat_resize"))
  # cmpSer.img_queue.put((readImage("datas/images/testImage/dog_clip.jpg"), "dog_clip"))

  try:
    while True:
      time.sleep(.6)
  except  KeyboardInterrupt:
    server.stop(0)

def readImage(path="images/test_7.png"):
  from PIL import Image
  img = Image.open(path).convert("RGB")
  arr = np.array(img)
  return arr.transpose(2, 0, 1)

def start_process():
  build()
  # 启动所有进程
  allStart()

if __name__ == "__main__":
  from src.utils.process import *
  init_local_server()
