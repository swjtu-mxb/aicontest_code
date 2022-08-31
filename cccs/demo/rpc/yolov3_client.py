from src.remote.alveo.yoloServcie import yolov3Service
from src.proto.host import cmd_pb2_grpc, cmd_pb2

import grpc
from concurrent import futures
import time
import numpy as np

def run():
  # NOTE(gRPC Python Team): .close() is possible on a channel and should be
  # used in circumstances in which the with statement does not fit the needs
  # of the code.
  with grpc.insecure_channel('localhost:5002') as channel:
    stub = cmd_pb2_grpc.yolov3SeviceStub(channel)
    response = stub.Runyolov3(cmd_pb2.yolov3Datain(cameraData = 1))
  print(np.array(response.data))

if __name__ == "__main__":
  run()
 