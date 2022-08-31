from tkinter import Image
from src.remote.compress.runcompress import CompressService
from src.proto.host import cmd_pb2_grpc 

import grpc
from concurrent import futures
import time

def init_local_server():
  # 1. 开始server 
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
  # 2. 把service加入到server监控中 
  cmd_pb2_grpc.add_CompressServiceServicer_to_server(CompressService(), server)
  # 3. 指定端口
  server.add_insecure_port('localhost:6666')
  server.start()
  try:
    while True:
      time.sleep(.6)
  except  KeyboardInterrupt:
    server.stop(0)

if __name__ == "__main__":
  # import numpy as np
  # def trans_pos(pos_list):
  #   lit=[]
  #   for i in range(len(pos_list)):
  #     ret = []
  #     ret.append(np.arccos((200-pos_list[i][0])/200)/np.pi*180)
  #     if((pos_list[i][1]) <= 200):
  #       ret.append(np.arccos((200-pos_list[i][1])/200)/np.pi*180)
  #     else:
  #       ret.append(180-np.arccos((pos_list[i][1]-200)/200)/np.pi*180)
  #     lit.append(ret)
  #   return lit
  # list=[(100,100),(100,200),(200,0),(200,400),(200,300)]
  # list1=trans_pos(list)  
  # print(list1)
  from PIL import Image
  img = Image.open("datas/gui/PYNQ.jpg")
  img = img.resize((150, 85)).convert("RGB")
  img.save("datas/gui/pynq_icon.jpg")