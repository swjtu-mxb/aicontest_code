from functools import reduce
import multiprocessing
from src.config.cfg import PYNQ_IMG_SIZE, PYNQ_IP, PYNQ_NUM, PYNQ_PORT, USB_DEVICE
from src.utils.check import check_fifo
from src.utils.process import *

from src.proto.host import cmd_pb2_grpc
from src.proto.host.cmd_pb2 import CompressResult
from src.proto.host import cmd_pb2

from src.remote.pipe import *

from src.remote.compress.runcnn import RunCnn

import os
import cv2
import numpy as np
from multiprocessing import Manager
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import time
from queue import Empty
from PIL import Image
import serial
from multiprocessing import Queue

from src.utils.img import draw


class ImageGen:
  def __init__(self, remote_sense, landslide, ultraPath, idx=0):
    self.camera = cv2.VideoCapture(idx)
    self.camera.set(3, 1280)
    self.camera.set(4, 720)
    self.x_base = (1280 - 512) // 2
    self.y_base = (720 - 512) // 2
    self.remote_img = list(map(lambda x: os.path.join(remote_sense, x), os.listdir(remote_sense)))
    self.land_img = list(map(lambda x: os.path.join(landslide, x), os.listdir(landslide)))
    self.ultra_img = list(map(lambda x: os.path.join(ultraPath, x), os.listdir(ultraPath)))

    self.image_class = cmd_pb2.REMOTE_SENSING
    self.remote_idx = 0
    self.land_idx = 0
    self.camera_idx = 0
    self.ultra_idx = 0
    self.is_ultra = False
    self.show_process = AppProcess("show", target=self.__show_process)
    self.show_img_queue = Queue(QUEUE_SIZE)
    # self.show_process.start()
    # self.window = cv2.setWindowTitle("raw")
  
  def __show_process(self):
    while True:
      img = self.show_img_queue.get()
      cv2.imshow("raw", img)
      cv2.waitKey(10)
  
  def getUltraImg(self):
    name = self.ultra_img[self.ultra_idx]
    self.ultra_idx += 1 
    if self.ultra_idx >= len(self.ultra_img):
      self.ultra_idx = 0
    img = Image.open(name).resize([512, 512])
    prefix_name = name.split("/")[-1].split(".")[0]
    return (np.array(img, dtype=np.uint8), prefix_name)

  
  def setImageClass(self, cls):
    self.image_class = cls.imgType
  

  def setUltraResult(self, img, name):
    self.ultra_result = img
    self.ultra_name = name
    self.is_ultra = True

  '''
  return: cv2 Mat 
  '''
  def getImge(self):
    if self.is_ultra:
      self.is_ultra = False
      return (self.ultra_result.transpose([2, 0, 1]), self.ultra_name)

    if self.image_class == cmd_pb2.CAMERA:
      ret, img = self.camera.read()
      cv2.waitKey(10)
      img = img[self.y_base:self.y_base+512, self.x_base:self.x_base+512]
      self.show_img_queue.put(img)

      img = np.array(img, dtype=np.uint8).transpose(2, 0, 1)
      name = "camera_" + str(self.camera_idx)
      self.camera_idx += 1
      return (img[::-1, :, :], name)
    elif self.image_class == cmd_pb2.REMOTE_SENSING:
      img = self.remote_img[self.remote_idx]
      name = img.split("/")[-1].split(".")[0]
      self.remote_idx += 1
    else:
      img = self.land_img[self.land_idx]
      name = img.split("/")[-1].split(".")[0]
      self.land_idx += 1
    
    img = Image.open(img)

    if self.remote_idx == len(self.remote_img):
      self.remote_idx = 0

    if self.land_idx == len(self.land_img):
      self.land_idx = 0

    if self.image_class == cmd_pb2.LANDSLIDE:
      img = img.resize([512, 512])

    img = np.array(img, dtype=np.uint8).transpose(2, 0, 1)
    showImg = cv2.merge((img[2], img[1], img[0]))
    self.show_img_queue.put(showImg)
    return (img, name)

class CompressRes:
  def __init__(self, data, last, width, height, time, name="") -> None:
    self.data = data
    self.last = last
    self.width = width
    self.height = height
    self.time = time
    self.name = name

class Lattice:
  def __init__(self):
    super().__init__()
  
  '''
  code: LatticeBitsCmd
  '''
  def complie(self, code):
    code_asiic = str(code.code, encoding='utf-8')
    with open("src/remote/lattice/tmp.v", "w") as f:
      f.write(code_asiic)

    constrain = str(code.constrain, encoding='utf-8')
    with open("src/remote/lattice/io.pcf", "w") as f:
      f.write(constrain)

    os.system("cd src/remote/lattice/ && make &")

  
class CompressService(cmd_pb2_grpc.CompressService):
  def __init__(self, connect):
    self.ultra_connect = connect
    # pynq
    self.run_cnn_handles = [RunCnn(i) for i in PYNQ_PORT]

    self.manager = multiprocessing.Manager()
    self.enable = self.manager.list([True for i in PYNQ_PORT])

    self.img_queue = Queue(QUEUE_SIZE) # 其他进程push image进来
    self.main_name = multiprocessing.current_process().name
 
    self.th_pool = ThreadPoolExecutor(max_workers=PYNQ_NUM)

    self.always_run_compress_process = AppProcess("always_run_compress", self.alwaysRunCompress, self.enable)
    self.compress_result = Queue(100)

    self.img_gen = ImageGen("datas/images/B", landslide="datas/datasets/landslide", ultraPath="datas/datasets/landslide")

    self.get_gen_th = threading.Thread(target=self.get_image_process, args=(self.img_gen, ))
    self.to_gen_th = queue.Queue(2)

    self.lattice = Lattice()
    self.get_gen_th.start()
    try: 
      self.ser = serial.Serial(port=USB_DEVICE, baudrate=115200)
    except Exception as e:
      print(e)
      print("can not open {}".format(USB_DEVICE))
      self.ser = None
  
  # pos: (int, int)
  def __send_pos(self, pos):
    cmd = 'ff'+str("{:02X}".format(pos[0]))+''+str("{:02X}".format(pos[1]))+'aa'
    self.__serial_tx(cmd)

  def __serial_tx(self, data):
    send_data = bytes.fromhex(data)
    result = self.ser.write(send_data)
    count = self.ser.inWaiting()
    self.ser.flushInput()
  
  '''
  get_obj: ImageGen
  '''
  def get_image_process(self, get_obj: ImageGen):
    isContinue = 0
    while True:
      if isContinue == 2:
        self.img_queue.put(get_obj.getImge())
        if not self.img_queue.empty():
          if self.to_gen_th.get() == 1:
            isContinue = 0
        time.sleep(0.010)
      else:
        isContinue = self.to_gen_th.get()
        self.img_queue.put(get_obj.getImge())
  
  def setImage(self, cls, context):
    self.img_gen.setImageClass(cls)
    # clear queue ?
    try:
      while True:
        self.img_queue.get_nowait()
    except Empty:
      pass
    return cmd_pb2.Empty()
  
  def setContinue(self, cls, context):
    print("set continue")
    self.to_gen_th.put(2)
    return cmd_pb2.Empty()

  def setNoContinue(self, cls, context):
    self.to_gen_th.put(1)
    return cmd_pb2.Empty()

  def getMonitorResult(self, cls, context):
    img, name = self.img_gen.getUltraImg()
    res = self.ultra_connect.root.runyolov3(img.tobytes())
    if res[0] > 0:
      self.img_gen.setUltraResult(img, name)
    ret = cmd_pb2.YoloResponse()
    ret.position.extend(res)
    return ret
  
  # def getMonitorImg(self, cls, context):

  def setPYNQ(self, cmd, context):
    if cmd.idx < PYNQ_NUM:
      self.enable[cmd.idx] = cmd.enable
    return cmd_pb2.Empty()

  def uploadPYNQ(self, cls, context):
    with open("src/remote/pynq/part.bin", "wb") as f:
      f.write(cls.bits)
    args = [str(m) for m in cls.idx]
    args = reduce(lambda x, y: x + " " + y, args)
    print("pynq bits downloads")
    # os.system("cd src/remote/pynq/ && bash run.sh {}".format(args))
    return cmd_pb2.Empty()

  def uploadLattice(self, cmd, context):
    self.lattice.complie(cmd)
    return cmd_pb2.Empty()
  
  # img: [3, 512, 512]
  def split_img(self, img):
    # [3, 256, 512]
    h_slice = img.shape[1] // PYNQ_IMG_SIZE; w_slice = img.shape[2] // PYNQ_IMG_SIZE
    slice_img = np.split(img, h_slice, axis=1)
    slice_img = list(map(lambda x: np.split(x, w_slice, axis=2), slice_img))
    flatten_img = []
    for l in slice_img:
      flatten_img += l
    # [4, 3, 256, 256]
    slice_img = np.concatenate(list(map(lambda x: x[None, :, :, :], flatten_img)), axis=0)
    return slice_img

  def every_thread_work(self, zipped):
    img, cnn_handle = zipped
    return cnn_handle.run(img)
  
  # pos: (int, int)
  def __send_pos(self, pos):
    cmd = 'ff'+str("{:02X}".format(pos[0]))+''+str("{:02X}".format(pos[1]))+'aa'
    self.__serial_tx(cmd)

  def __serial_tx(self, data):
    send_data = bytes.fromhex(data)
    result = self.ser.write(send_data)
    count = self.ser.inWaiting()
    self.ser.flushInput()

  def setMotor(self, cmd, context):
    if self.ser == None:
      print("serial not enable")
      return cmd_pb2.Empty()
    self.__send_pos((cmd.yaw, cmd.pitch))
    return cmd_pb2.Empty()
  
  def alwaysRunCompress(self, enable):
    # img: [512, 512, 3]
    while True:
      img, name = self.img_queue.get(True) 
      # 选择设备
      ziped = zip(enable, self.run_cnn_handles) 
      ziped = list(filter(lambda x:x[0], ziped))
      device = list(map(lambda x: x[1], ziped))
      print(name)
      height, width = img.shape[1], img.shape[2]
      slice_img = self.split_img(img)
      total_len = len(slice_img)
      NUM = len(device)
      start = time.time()
      for i in range(0, slice_img.shape[0], NUM):
        part_img = slice_img[i:(i+NUM)]
        zipped = zip(part_img, device[:len(part_img)])
        for re, idx in zip(self.th_pool.map(self.every_thread_work, zipped), range(len(part_img))):
          end = time.time()
          result = CompressRes(data=re, last=(total_len == (idx + i + 1)), width = width, height=height, time=(end - start))
          # 最后一个发送名字
          if result.last:
            result.name = name
          self.compress_result.put(result)
    
  def RunCompress(self, cmd, context): # 多了个context参数
    self.to_gen_th.put(0) #拿到新图片
    res = self.compress_result.get(True)
    first = False
    while res.last == False:
      if first:
        res = self.compress_result.get(True)
      first = True
      ret = CompressResult(data=res.data, last=res.last, width=res.width, height=res.height, time=res.time, name=res.name) 
      yield ret
    
if __name__ == "__main__":
  import threading
  import time
  class A:
    def __init__(self) -> None:
      self.a = "init"
      self.th = threading.Thread(target=self.run) 
    def set(self, strv):
      self.a = strv
    def printInfo(self):
      print(self.a)
    def run(self):
      while True:
        self.printInfo()
        time.sleep(2)

  v = A()
  v.th.start()
  time.sleep(1)
  v.set("over")
  v.th.join()