from matplotlib.colors import NoNorm
from src.host.app import YoloApp, ChangeDetectApp
from src.host.status import RunCls, sta, set_from_img_process, register_app
from src.host.decompress import ImageDecompress
from src.utils.process import allJoin, build, allStart
from src.proto.host import cmd_pb2_grpc

from multiprocessing import current_process

# stub = None
# change = None
# yolo = None
# img_cmp = None

class GlobalData:
  def __init__(self):
    self.yolo = YoloApp("yolov3", (RunCls.YOLO,))
    self.change = ChangeDetectApp("change_detect", (RunCls.CHANGE,))
    # sta.set_app_cls(RunCls.FACE)
    build()
    set_from_img_process(current_process())
    self.img_cmp = ImageDecompress(self)
    allStart()
    self.stub = cmd_pb2_grpc.CompressServiceStub(self.img_cmp.channel)

glb = GlobalData()

def init_host():
  yolo = YoloApp("yolov3", (RunCls.YOLO,))
  change = ChangeDetectApp("change_detect", (RunCls.CHANGE,))
  sta.set_app_cls(RunCls.YOLO)
  build()
  set_from_img_process(current_process())
  img_cmp = ImageDecompress()
  allStart()
  stub = cmd_pb2_grpc.CompressServiceStub(img_cmp.channel)


if __name__ == "__main__":
  init_host()