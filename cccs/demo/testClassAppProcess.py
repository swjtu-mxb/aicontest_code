from src.utils.process import *
import multiprocessing
from multiprocessing import Queue
import time
import random


class Test:
  def __init__(self) -> None:
    self.send = AppProcess("send", target=self.send) 
    self.rec = AppProcess("receive", target=self.receive)
    self.rec.fromProcess("send")
    pass

  def send(self):
    q:Queue = getToQueue("receive")
    for i in range(10):
      try: 
        d = random.randint(1, 5)
        q.put(d)
        time.sleep(d)
      except KeyboardInterrupt:
        print("key stop")
        break

  def receive(self):
    q:Queue = getFromQueue("send")
    for _ in range(10):
      try: 
        v = q.get(True)
        print(v)
      except KeyboardInterrupt:
        print("key stop")
        break

if __name__ == "__main__":
  # receive 依赖于 send
  # 将依赖和被依赖进程建立联系
  a = Test()
  build()
  # 启动所有进程
  allStart()
  # 等待所有进程over
  allJoin()