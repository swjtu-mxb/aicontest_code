import math
from src.proto.host import cmd_pb2_grpc, cmd_pb2
from src.config.cfg import DECOMPRESS_PROCESS_NUM, REMOTE_PORT
from src.host.status import sta, queue

from src.host.ACRecon.parallel_decomp import arithmeticDecompress, decompress
from src.host.status import RunCls

import grpc
from multiprocessing import Pool
import numpy as np
import time
from functools import reduce



class ImageDecompress:
  def __init__(self, glb):
    self.channel = grpc.insecure_channel(REMOTE_PORT)
    self.dcmp_pool = Pool(DECOMPRESS_PROCESS_NUM)
    self.glb = glb

  # push img and name to process
  def __push_img_2_app(self, img_and_name):
    global sta
    cls_now = sta.get_app_cls()
    for k, v in queue.items():
      if cls_now in k:
        v.put(img_and_name)
    if sta.get_app_cls() == RunCls.CHANGE:
      if self.glb.change.isStart == False:
        self.glb.change.process.start()
        self.glb.change.isStart = True
      self.glb.change.queue.put(img_and_name)

  # return: (img: Numpy array, name: str)
  def get_img(self, total_time, cmp_time, data_len, speed, speed_incr):
    stub = cmd_pb2_grpc.CompressServiceStub(self.channel)
    group = []
    res = []
    name = ""
    start = time.time()
    total = 0.
    data = 0
    speed_num = 0
    for e in stub.RunCompress(cmd_pb2.CompressCmd(start=True)):
      if name == "":
        name = e.name
      group.append(e)
      if len(group) == DECOMPRESS_PROCESS_NUM or e.last == True:
        byte = list(map(lambda x: x.data, group))
        p = self.dcmp_pool.map(arithmeticDecompress, byte)
        res += p
        group = []
      
      last = time.time()
      total = last - start
      data += len(e.data)
      speed_num = min(speed_incr + speed_num, 100)

      total_time.emit(total)
      cmp_time.emit(e.time)
      data_len.emit(data)
      speed.emit(speed_num)

    speed.emit(100)
    img = decompress(np.array(res))
    # print("decompress time total = {:.2f}, arth time = {:.2f}, gpu time = {:.2f}, ".format(end - start, middle - start, end - middle))
    if sta.is_run_cnn():
      self.__push_img_2_app((img, name))
    return (img, name)

if __name__ == "__main__":
  from PIL import Image
  gimg = ImageDecompress()
  res = gimg.get_img()
  # img = Image.fromarray(res[0])
  # img.save("datas/images/tmp.jpg")