
from rpyc import Service
import numpy as np
from multiprocessing import Manager

class CnnService(Service):
  def __init__(self):
    super().__init__()

  def exposed_runCnn(self, img): # 多了个context参数
    img = np.frombuffer(img, dtype=np.int16)
    ret = np.random.randint(0, 100, [64, 32, 32]).astype(np.int32)
    return ret.tobytes()

  def exposed_downloadBits(self, cmd, context):
    return True
