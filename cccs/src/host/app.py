from src.utils.process import AppProcess, getFromQueueFirst

from src.host.status import sta, RunCls, register_app
from src.config.cfg import *

from src.proto.host import cmd_pb2_grpc, cmd_pb2
from src.host.change_detec.run_change import predict, DEVICE
from src.host.ACRecon.parallel_decomp import ctx

import grpc
from math import sqrt
import torch
import numpy as np
from multiprocessing import Queue
from PIL import Image, ImageDraw


class YoloApp:
  # name: app's name
  # cls: list, all condition to run
  def __init__(self, name, cls):
    self.channel = grpc.insecure_channel(U50_PORT)
    register_app(self.detect, name + "_app", cls, self.channel)
    self.outQueue = Queue(1)

  def detect(self, channel):
    img_q = getFromQueueFirst()
    stub = cmd_pb2_grpc.Yolov3SeviceStub(channel)
    while True:
      # numpy, str
      img, name = img_q.get(True)
      width = img.shape[0]
      if width > 512:
        bias = (width - 512) // 2
        img = img[bias:bias+512, bias:bias+512, :]
      dat = img[:, :, ::-1].tobytes()
      result = self.draw(img, stub.runyolov3(cmd_pb2.Yolov3Datain(cameraData=dat, width=512, height=512)))
      self.outQueue.put((result.tobytes(), [512, 512, 3]))
    
  def draw(self, img, res):
    pos = res.position
    if pos[0] > 0:
      img = Image.fromarray(img)
      draw = ImageDraw.Draw(img)
      for i in range(pos[0]):
        start = i * 4 + 1
        axi = pos[start:start+4]
        axi = tuple(map(lambda x: float(x), axi))
        draw.rectangle(axi, outline=(255, 0, 0))
      return np.array(img)
    else:
      return img


class ChangeDetectApp:
  def __init__(self, name, cls):
    self.model_path = './datas/ckpt/best_model.pth'
    self.queue = ctx.Queue(1)
    self.outQueue = ctx.Queue(1)
    self.process = ctx.Process(target=self.change_detect, args=(self.model_path, self.queue))
    self.isStart = False
    # register_app(self.change_detect, name + "_app", cls, self.model_path)
  
  def change_detect(self, model_p, queue):
    import src.host.change_detec.vitis_model as cdp
    model = cdp.Unet1(
        encoder_name="resnet18",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=2,  # model output channels (number of classes in your datasets)
        siam_encoder=True,  # whether to use a siamese encoder
        fusion_form='concat',  # the form of fusing features from two branches. e.g. concat, sum, diff, or abs_diff.
    )
    model = torch.load(model_p)
    model.to(DEVICE)
    model.eval()
    while True:
      img, name = queue.get(True)
      id = name.split("_")[-1]
      if img.shape[0] == 1024:
        self.outQueue.put((predict(id, img, model).tobytes(), [1024, 1024, 1])) # put to queue: Image object

