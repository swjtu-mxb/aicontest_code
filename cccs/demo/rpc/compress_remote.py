from src.remote.compress.runcompress import CompressService
from src.proto.host import cmd_pb2_grpc 

import grpc
from concurrent import futures
import time

def serve():
  # 1. 开始server 
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
  # 2. 把service加入到server监控中 
  cmd_pb2_grpc.add_CompressServiceServicer_to_server(CompressService(), server)
  # 3. 指定端口
  server.add_insecure_port('localhost:5001')
  server.start()
  try:
    while True:
      time.sleep(.6)
  except  KeyboardInterrupt:
    server.stop(0)
  
if __name__ == "__main__":
  serve()
 
