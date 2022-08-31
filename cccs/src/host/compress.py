from src.proto.host import cmd_pb2_grpc, cmd_pb2

import grpc
import time

def save(res, idx):
  with open("datas/tmp/test_7_{}.bin".format(idx), "wb") as f:
    f.write(res.data)

def run():
  with grpc.insecure_channel('localhost:5678') as channel:
    stub = cmd_pb2_grpc.CompressServiceStub(channel)
    idx = 0
    for res in stub.RunCompress(cmd_pb2.CompressCmd(start=True)):
      save(res, idx)
      idx += 1

if __name__ == "__main__":
  run()