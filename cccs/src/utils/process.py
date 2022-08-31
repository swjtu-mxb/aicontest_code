from src.config.cfg import QUEUE_SIZE


from functools import singledispatch, update_wrapper
from multiprocessing import Queue, Process
from multiprocessing import current_process
import abc

graph = {}
allProcess = {}

def methdispatch(func):
  dispatcher = singledispatch(func)
  def wrapper(*args, **kw):
      return dispatcher.dispatch(args[1].__class__)(*args, **kw)
  wrapper.register = dispatcher.register
  update_wrapper(wrapper, func)
  return wrapper

def getFromQueueFirst():
  name = current_process().name
  return list(graph[name].fromQueue.values())[0]

@singledispatch
def getFromQueue(fromPro):
  name = current_process().name
  return graph[name].fromQueue[fromPro.name]

@getFromQueue.register(str)
def _(fromName):
  name = current_process().name
  return graph[name].fromQueue[fromName]

@singledispatch
def getToQueue(toPro):
  name = current_process().name
  return graph[name].toQueue[toPro.name]

@getToQueue.register(str)
def _(toName):
  name = current_process().name
  return graph[name].toQueue[toName]

def build():
  for k, m in graph.items():
    for sk, sm in m.fromQueue.items():
      if sk != "MainProcess":
        graph[sk].toQueue[k] = sm

def allStart():
  for m in allProcess.values():
    m.start()

def allJoin():
  for m in allProcess.values():
    m.join()

class DependGraph:
  def __init__(self, name):
    self.name = name
    self.fromQueue = {}
    self.toQueue = {}
    graph[name] = self
  
  def addFrom(self, fromName):
    queue = Queue(QUEUE_SIZE)
    self.fromQueue[fromName] = queue

class AppProcess(Process):
  def __init__(self, name, target=None,*args, **kwargs):
    super(AppProcess, self).__init__(name=name, target=target, args=args, kwargs=kwargs)
    self.graph = DependGraph(name)
    allProcess[name] = self # 加入全局的进程池

  @methdispatch
  def fromProcess(self, pro):
    self.graph.addFrom(pro.name)

  @fromProcess.register
  def _(self, name: str):
    self.graph.addFrom(name)

  def getFromQueue(self, fromPro):
    if isinstance(fromPro, str):
      return self.graph.fromQueue[fromPro]
    else: 
      return self.graph.fromQueue[fromPro.name]
  
  def getToQueue(self, toName):
    if isinstance(toName, str):
      return self.graph.toQueue[toName]
    else: 
      return self.graph.toQueue[toName.name]