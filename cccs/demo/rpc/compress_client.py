from src.remote.compress.runcompress import CompressService
from src.proto.host import cmd_pb2_grpc, cmd_pb2

import grpc
from concurrent import futures
import time

def run():
  # NOTE(gRPC Python Team): .close() is possible on a channel and should be
  # used in circumstances in which the with statement does not fit the needs
  # of the code.
  with grpc.insecure_channel('localhost:5001') as channel:
    stub = cmd_pb2_grpc.CompressServiceStub(channel)
    response = stub.RunCompress(cmd_pb2.CompressCmd(start=True))
  print(bytearray(response.data))

if __name__ == "__main__":
  run()
 