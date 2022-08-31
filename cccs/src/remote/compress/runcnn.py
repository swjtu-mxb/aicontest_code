from src.utils.check import check_fifo, delete_fifo
from src.utils.process import *
from src.proto.host.cmd_pb2 import CompressResult
from src.remote.pipe import *

import os, numpy as np
import rpyc
import multiprocessing
from multiprocessing import Manager, Pool
import time

class PynqNames:
  def __init__(self, id) -> None:
    self.write_name = 'write_' + id
    self.read_name = 'read_' + id
    self.pynq_name = 'pynq_' + id
    self.write_fifo_name = '/tmp/' + self.write_name
    self.read_fifo_name = '/tmp/' + self.read_name

class RunCnn:
  def __init__(self, pynq_ip: str):
    self.pynq_ip = pynq_ip
    self.id = pynq_ip.split('.')[(-1)].split(':')[0]
    self.ip = pynq_ip.split(":")[0]
    self.port = pynq_ip.split(":")[-1]
    self.names = PynqNames(self.id)
    self.conn = rpyc.connect(self.ip, (int)(self.port))
    self.init_process()
    self.main_process_name = multiprocessing.current_process().name

  def init_process(self):
    check_fifo(self.names.write_fifo_name)
    check_fifo(self.names.read_fifo_name)
    self.write_process = AppProcess(self.names.write_name, self.write, self.names.write_fifo_name)
    self.read_process = AppProcess(self.names.read_name, self.read, self.names.read_fifo_name)
    self.pynq_process = AppProcess(self.names.pynq_name, self.pynq, self.names.write_fifo_name, self.names.read_fifo_name)
    self.write_process.fromProcess(self.pynq_process)
    self.read_process.fromProcess(self.pynq_process)
    self.pynq_process.fromProcess(multiprocessing.current_process())
    self.main_queue = Queue(QUEUE_SIZE)

  def run(self, img):
    to_pynq_queue = self.pynq_process.getFromQueue(self.main_process_name)
    to_pynq_queue.put(img)
    return self.main_queue.get(True)
  

  # img: numpy array
  def img_preprocess(self, img):
    return (img / 255 * 1024).astype(np.uint16)

  # pynq process
  def pynq(self, wrpath, rdpath):
    q_img = getFromQueue(self.main_process_name)
    wr_q = getToQueue(self.write_process)
    rd_q = getToQueue(self.read_process)
    while True:
      img = q_img.get(True)
      img =  self.img_preprocess(img)
      byte = img.tobytes(order='C')
      res = self.conn.root.runCnn(byte) # runCnn
      wr_q.put(res)
      rd_q.put(True)
      os.system('./datas/cmd/acad/ArithmeticCompress {} {}'.format(wrpath, rdpath))

  def write(self, wrpath):
    pynq_q = getFromQueue(self.pynq_process)
    while True:
      cmp_byte = pynq_q.get(True)
      wrpipe = os.open(wrpath, os.O_WRONLY)
      os.write(wrpipe, cmp_byte)
      os.close(wrpipe)

  def read(self, rdpath):
    q = getFromQueue(self.pynq_process)
    while True:
      _ = q.get(True)
      with open(rdpath, 'rb') as f:
        byte = f.read()
      self.main_queue.put(byte)

  def __del__(self):
    self.conn.close()
    delete_fifo(self.names.write_fifo_name)
    delete_fifo(self.names.read_fifo_name)
