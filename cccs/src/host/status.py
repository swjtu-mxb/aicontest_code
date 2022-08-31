from src.proto.host import cmd_pb2
from src.utils.process import AppProcess

from enum import Enum
from multiprocessing.managers import BaseManager

class RunCls(Enum):
  CHANGE = 1
  YOLO = 2 
  NON = 3

process = {} # 各种cls -> 进程
queue = {}   # 各种cls -> queue
def register_app(func, name, cls, *args):
  process[cls] = AppProcess(name, func, *args)

def set_from_img_process(prc):
  for k, v in process.items():
    v.fromProcess(prc)
    queue[k] = v.getFromQueue(prc) 
 
class Status:
  def __init__(self):
    self.class_of_app = RunCls.CHANGE
    self._run_cnn = True
    self.img_type = cmd_pb2.REMOTE_SENSING
  
  def disable_run_cnn(self):
    self._run_cnn = False

  def enable_run_cnn(self):
    self._run_cnn = True
  
  def is_run_cnn(self):
    return self._run_cnn
  
  def set_app_cls(self, app_cls):
    self.class_of_app = app_cls
  
  def get_app_cls(self):
    return self.class_of_app
 
  def get_queue_by_cls(self, cls):
    return queue(cls)
    

# global: baseManger sta
# baseManger = BaseManager()
# baseManger.register("Status", Status)
# baseManger.start()
# sta:Status = baseManger.Status()
sta = Status()

if __name__ == "__main__":
  def func(a):
    print(retrieve_name(a))
    print(a)
  from src.utils.common import retrieve_name
  func(1)
