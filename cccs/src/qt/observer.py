from concurrent.futures import thread
from functools import partial
from pickletools import pystring
from src.qt.ui.start import Ui_MainWindow as Start_MainWindow

from src.host.run_main import glb
from src.host.status import sta, RunCls
from src.proto.host import cmd_pb2
from src.qt.startQt import Start_MainWindow, StartQt
from src.config.cfg import DECOMPRESS_PROCESS_NUM
from src.utils.img import draw

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QMovie
from PyQt5.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QFileDialog

import queue
import enum
import time
import numpy as np
from PIL import Image, ImageDraw
import os
import ptvsd
import imageio
import threading

debug = lambda : 1
# debug = ptvsd.debug_this_thread

class IMDSolver:
  def __init__(self, start: Start_MainWindow):
    self.startw = start
    self.__set_images()
     
  def __set_images(self):
    change_func = lambda : self.set_image_type(cmd_pb2.ImageClass(imgType=cmd_pb2.REMOTE_SENSING))
    landslide_func = lambda : self.set_image_type(cmd_pb2.ImageClass(imgType=cmd_pb2.LANDSLIDE))
    camera_func = lambda : self.set_image_type(cmd_pb2.ImageClass(imgType=cmd_pb2.CAMERA))

    self.startw.changeImgButton.toggled.connect(change_func)
    self.startw.landImgButton.toggled.connect(landslide_func)
    self.startw.cameaImgButton.toggled.connect(camera_func)
  
  def set_image_type(self, cls):
    glb.stub.setImage(cls) 
    sta.img_type = cls.imgType


class DetectThread(QThread):
  img = pyqtSignal(tuple)
  global glb
  def __init__(self, start: Start_MainWindow):
    super(DetectThread, self).__init__()
    self.startw = start
    self.state = RunCls.CHANGE
    self.__set_button()
    self.graph = self.startw.resGraph
    self.img.connect(self.__draw_img)
  
  def __draw_img(self, img_size):
    byte, size = img_size[0], img_size[1]
    img = self.__post_process(byte, size)
    if size[-1] == 1:
      frame = QImage(np.array(img, dtype=np.uint8), img.size[1], img.size[0], QImage.Format_Grayscale8)
    else:
      frame = QImage(np.array(img, dtype=np.uint8), img.size[1], img.size[0], QImage.Format_RGB888)
    pix = QPixmap.fromImage(frame)
    self.item = QGraphicsPixmapItem(pix)  
    self.scene= QGraphicsScene() 
    self.scene.addItem(self.item)
    self.graph.setScene(self.scene) 
  
  def __set_state(self, s):
    self.state = s
    if s != RunCls.NON:
      sta.set_app_cls(s)
      sta.enable_run_cnn()
    else:
      sta.disable_run_cnn()
  
  def __set_button(self):
    self.startw.changeButton.toggled.connect(lambda : self.__set_state(RunCls.CHANGE) if self.startw.changeButton.isChecked() else 0)
    self.startw.detectButton.toggled.connect(lambda : self.__set_state(RunCls.YOLO) if self.startw.detectButton.isChecked() else 0)
    self.startw.nonButton.toggled.connect(lambda : self.__set_state(RunCls.NON) if self.startw.nonButton.isChecked() else 0)
  
  def __resize(self, img, size=[512, 512]):
    if img.size[0] != size[0]:
      return img.resize(size)
    return img
  
  # byte: image data byte
  # size: image size
  # return: Image object
  def __post_process(self, byte, size):
    if size[-1] == 1:
      size = size[:2]
    img = np.frombuffer(byte, dtype=np.uint8).reshape(size)
    img = Image.fromarray(img)
    img = self.__resize(img)
    return img
  
  def __get_change(self):
    byte = glb.change.outQueue.get()
    return byte

  def __get_yolo(self):
    byte = glb.yolo.outQueue.get()
    return byte
  
  def run(self) -> None:
    debug()
    while True:
      if not glb.change.outQueue.empty():
        self.img.emit(glb.change.outQueue.get())
      elif not glb.yolo.outQueue.empty():
        self.img.emit(glb.yolo.outQueue.get())
      else:
        time.sleep(0.01)

class ImageModel(enum.Enum):
  SINGLE = 1
  MULTI  = 2
  MONITOR = 3

class ImageReceiveShow:
  # signal: (signal, graph, condition)
  def __init__(self, slover, signal, size=[512, 512]) -> None:
    self.slover = slover
    self.signal = signal
    self.size = size
    for i in self.signal:
      (s, g, _, _) = i
      s.connect(partial(self.__show, graph=g))
  
  def __show(self, imgByte, graph):
    img = np.frombuffer(imgByte, dtype=np.uint8).reshape(self.size + [3])
    frame = QImage(img, img.shape[1], img.shape[0], QImage.Format_RGB888)
    pix = QPixmap.fromImage(frame)
    item = QGraphicsPixmapItem(pix)  
    scene= QGraphicsScene() 
    scene.addItem(item)
    graph.setScene(scene)  
    graph.repaint()

  '''
  res: (Image object)
  '''
  def showImage(self, res, name, cond, position=None):
    img = Image.fromarray(res)
    rawImg = img
    if img.size[0] != self.size[0]:
      img = img.resize(self.size)
    img = np.array(img)
    if position != None:
      img = draw(img, position)
    for i in self.signal:
      (s, g, c, p) = i
      if c == cond:
        rawImg.save(os.path.join(p, name + ".jpg"))
        s.emit(img.tobytes())

  def __speed_incr(self):
    if sta.img_type == cmd_pb2.REMOTE_SENSING:
      return 100 // (4 * 4)
    else:
      return 100 / 4

  '''
  return: (img: numpy arr, name: str)
  '''
  def getImage(self, mode):
    if mode == cmd_pb2.CPU:
      res = glb.img_cmp.get_img(self.slover.total_time, self.slover.cmp_time, self.slover.data_len, self.slover.speed, self.__speed_incr())
    elif mode == cmd_pb2.PYNQ:
      res = glb.img_cmp.get_img(self.slover.total_time, self.slover.cmp_time, self.slover.data_len, self.slover.speed, self.__speed_incr())
    else:
      res = glb.img_cmp.get_img(self.slover.total_time, self.slover.cmp_time, self.slover.data_len, self.slover.speed, self.__speed_incr())
    return res

class ShowGraph(enum.Enum):
  RAW = 1
  CAMERA = 2

class GetImageThread(QThread):
  total_time = pyqtSignal(float)
  cmp_time = pyqtSignal(float)
  data_len = pyqtSignal(int)
  speed = pyqtSignal(int)
  rawSignal = pyqtSignal(bytes)
  cameraSignal = pyqtSignal(bytes)
  def __init__(self, start: Start_MainWindow, camera_graph):
    super(GetImageThread, self).__init__()
    self.startw = start
    self.__speed_init()
    self.__set_button()
    self.getShowObj = ImageReceiveShow(slover=self, signal=((self.rawSignal, self.startw.rawGraph, ShowGraph.RAW, "datas/result"), (self.cameraSignal, camera_graph, ShowGraph.CAMERA, "datas/camera")))
    self.queue_mode = queue.Queue(1)
    self.queue_device = queue.Queue(1)
  
  def __speed_init(self):

    self.total_time.connect(self.__draw_total)
    self.cmp_time.connect(self.__draw_cmp)
    self.data_len.connect(self.__draw_data)
    self.speed.connect(self.__draw_speed)
  
  def __draw_speed(self, v):
    self.startw.progressBar.setValue(v)
  
  def __draw_total(self, t):
    self.startw.time.setText("{:.2f}s".format(t))

  def __draw_cmp(self, t):
    self.startw.cmpTime.setText("{:.2f}s".format(t))

  def __draw_data(self, l):
    self.startw.data.setText("{:.3f}KB".format(l / 1024))

  def __set_button(self):
    def cpu_func():
      self.queue_device.put(cmd_pb2.CPU)
    def pynq_func():
      self.queue_device.put(cmd_pb2.PYNQ)
    def none_func():
      self.queue_device.put(cmd_pb2.NONE)

    self.startw.cpuButton.toggled.connect(cpu_func)
    self.startw.pynqButton.toggled.connect(pynq_func)
    self.startw.noCmpButton.toggled.connect(none_func)

    def single_func():
      self.queue_mode.put((ImageModel.SINGLE, ShowGraph.RAW))
    self.startw.onceButton.clicked.connect(single_func)
  
    def multi_func():
      self.queue_mode.put((ImageModel.MULTI, ShowGraph.RAW))
    self.startw.continueButton.clicked.connect(multi_func)

    def monitor_func():
      self.queue_mode.put((ImageModel.MONITOR, ShowGraph.RAW))
    self.startw.monitorButton.clicked.connect(monitor_func)
  
  def __run_monitor(self):
    while True:
      res = glb.stub.getMonitorResult(cmd_pb2.Empty())
      if res.position[0] > 0:
        return res.position
  
  def __get_land_image(self, name):
    path = os.path.join("datas/images", name + ".jpg")
    img = Image.open(path)
  
  # img: numpy arr 
  # position: 
  def __draw(self, img, position):
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    for i in range(position[0]):
      start = i * 4 + 1
      axi = position[start:start+4]
      axi = tuple(map(lambda x: float(x), axi))
      draw.rectangle(axi, outline=(255, 0, 0))
    self.rawSignal.emit(np.array(img, dtype=np.uint8).tobytes())

  def run(self):
    debug()
    mode = ImageModel.SINGLE
    show = ShowGraph.RAW
    device = cmd_pb2.PYNQ
    position = []
    while True:
      if mode != ImageModel.SINGLE:
        if not self.queue_mode.empty():
          mode, show = self.queue_mode.get(False)

      if not self.queue_device.empty():
        device = self.queue_device.get()

      if mode == ImageModel.MULTI:
        img_arr, name = self.getShowObj.getImage(device)
        self.getShowObj.showImage(img_arr, name, show)
      elif mode == ImageModel.SINGLE:
        mode, show = self.queue_mode.get(True)
        if mode == ImageModel.SINGLE:
          img_arr, name = self.getShowObj.getImage(device)
          if len(position) != 0:
            self.getShowObj.showImage(img_arr, name, show, position)
            position.clear()
          else:
            self.getShowObj.showImage(img_arr, name, show)
      elif mode == ImageModel.MONITOR: 
          position = list(self.__run_monitor())
          white_img = np.full([512, 512, 3], 255, dtype=np.uint8)
          img_rect = draw(white_img, position)
          self.rawSignal.emit(img_rect.tobytes())
          self.data_len.emit(len(position) * 4)
          mode, show = self.queue_mode.get()
      else:
        time.sleep(0.2)

class DialSlover: #旋钮 调节pynq数量
  def __init__(self, start: StartQt) -> None:
    self.startw = start
    self.enable = [True for i in range(8)]
    self.ch_list = [
      self.startw.pynq_f.ch0,
      self.startw.pynq_f.ch1,
      self.startw.pynq_f.ch2,
      self.startw.pynq_f.ch3,
      self.startw.pynq_f.ch4,
      self.startw.pynq_f.ch5,
      self.startw.pynq_f.ch6,
      self.startw.pynq_f.ch7
    ]
    self.__init_check_button()
    self.__init_dial()

  def __init_check_button(self):
    for k, v in enumerate(self.ch_list):
      v.stateChanged.connect(partial(self.__inv, k, v))

  def __init_dial(self):
    self.startw.root.pynqDial.valueChanged.connect(self.__dial_change)
  
  def __dial_change(self):
    idx = self.startw.root.pynqDial.value()
    for k, v in enumerate(self.ch_list):
      v.disconnect()
      if self.enable[k] ^ (k < idx):
        v.setChecked(k < idx)
        self.enable[k] = k < idx
        glb.stub.setPYNQ(cmd_pb2.PYNQEnableCmd(enable=self.enable[k], idx=k))
    print(self.enable)
    self.__init_check_button()
    num = self.__enable_number()
    self.startw.pynq_f.number.setText(str(num))
    self.startw.root.pynqNumLabel.setText(str(num))
  
  def __enable_number(self):
    return np.array(self.enable).sum()

  def __inv(self, idx, v):
    self.enable[idx] = v.isChecked()
    glb.stub.setPYNQ(cmd_pb2.PYNQEnableCmd(enable=self.enable[idx], idx=idx))
    print(v.isChecked())
    print(self.enable)
    num = self.__enable_number()
    self.startw.root.pynqDial.disconnect()
    self.startw.root.pynqDial.setValue(num)
    self.__init_dial()
    self.startw.pynq_f.number.setText(str(num))
    self.startw.root.pynqNumLabel.setText(str(num))
  
class FpgaSlover:
  def __init__(self, start: StartQt):
    self.startw = start
    self.fpga_f = start.fpga_f
    self.is_hand_write = False
    self.__init_button()
  
  def __init_button(self):
    self.fpga_f.saveButton.clicked.connect(self.__save_code)
    self.fpga_f.uploadCode.clicked.connect(self.__code_path)
    self.fpga_f.uploadCon.clicked.connect(self.__con_path)
    self.fpga_f.compileButton.clicked.connect(self.__compile_download)

  def __code_path(self):
    self.is_hand_write = False
    self.code_path = QFileDialog.getOpenFileNames(self, '选择文件')[0]
    self.fpga_f.codePath.setText(self.code_path)

  def __con_path(self):
    self.constrain_path = QFileDialog.getOpenFileNames(self, '选择文件')[0]
    self.fpga_f.conPath.setText(self.constrain_path)

  def __save_code(self):
    self.is_hand_write = True
    self.hand_code = self.fpga_f.plainTextEdit.toPlainText() 
    print(self.hand_code)
  
  def __compile_download(self):
    if self.is_hand_write:
      code = self.hand_code
      self.is_hand_write = False
    else:
      with open(self.code_path, "r") as f:
        code = f.read()
    with open(self.constrain_path, "r") as f:
      constrain = f.read()
    code = bytes(code, encoding = "utf8")
    constrain = bytes(constrain, encoding = "utf8")
    cmd = cmd_pb2.LatticeBitsCmd(code=code, constrain=constrain)
    glb.stub.uploadLattice(cmd)
    print("compile download")

class CameraThread(QThread):
  posSingal = pyqtSignal(tuple)
  def __init__(self, start: StartQt, queue, interval=40, uart_device="/dev/ttyUSB0"):
    super().__init__()
    self.startw = start
    self.camera_f = self.startw.camera_f
    self.paintBoard = self.camera_f.paintBoard
    self.queue = queue
    self.__init_button()
    self.time = 10.
    self.is_hand_paint = False
    self.pos = []
    self.posSingal.connect(self.__pos_controll)
    self.now_pos = (0, 0)
    self.interval = interval
    self.start_flag = False
    glb.stub.setMotor(cmd_pb2.MotorCmd(pitch=0x40, yaw=0x40))
    # self.get_img_thread = threading.Thread(target=self.run)
    # self.get_img_thread.start()
  
  def __pos_controll(self, position):
    self.camera_f.pitch.setText(str(position[1]))
    self.camera_f.yaw.setText(str(position[0]))
    self.camera_f.pitch.repaint()
    self.camera_f.yaw.repaint()
    yaw = position[0]
    pitch = position[1]
    print((yaw, pitch))
    glb.stub.setMotor(cmd_pb2.MotorCmd(pitch=pitch, yaw=yaw))
    
  def __init_button(self):
    self.camera_f.clearPaintButton.clicked.connect(self.paintBoard.Clear)
    self.camera_f.singleButton.clicked.connect(self.__single)
    self.camera_f.setPaintButton.clicked.connect(self.__set_hand)
    self.camera_f.pathButton.clicked.connect(self.__get_path)
    self.camera_f.timeButton.clicked.connect(self.__set_time)

    self.camera_f.trackButton.clicked.connect(self.__track)
  
  def __track(self):
    # 删除数据
    os.system("rm -rf datas/camera/*")
    times = int((self.time * 1000) // self.interval)
    pos_arr = np.array(self.pos)
    # 点太多gg
    if times < len(self.pos):
      rate = len(self.pos) // times
      final_pos = pos_arr[::rate]
      intveral = self.interval
    # 点太少 
    else:
      intveral = (self.time * 1000) // len(self.pos)
      final_pos = pos_arr
    self.startw.root.cameaImgButton.click()
    send = False
    for i in range(final_pos.shape[0]):
      self.posSingal.emit(tuple(final_pos[i]))
      time.sleep(intveral / 1000)

      if send == False and ((i * intveral) > 600):
        self.queue.put((ImageModel.MULTI, ShowGraph.CAMERA))
        send = True
    self.queue.put((ImageModel.SINGLE, ShowGraph.CAMERA))
    self.save_gif("datas/camera", "datas/tmp/result.gif") 
    self.show_gif(self.startw.img_f.imgLabel, "datas/tmp/result.gif")
    self.startw.img_frame.show()
  
  def show_gif(self, label, gif_path):
    gif = QMovie(gif_path)
    label.setMovie(gif)
    gif.start()
  
  def sort_img(self, names):
    names.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    return names
  
  def save_gif(self, file_dir, save_path):
    files = os.listdir(file_dir)
    files = self.sort_img(files)
    file_name = list(map(lambda x: os.path.join(file_dir, x), files))
    frames = []
    for image_name in file_name:
        im = imageio.imread(image_name)           # 读取方式上存在略微区别，由于是直接读取数据，并不需要后续处理
        frames.append(im)
    imageio.mimsave(save_path, frames, 'GIF', duration=0.1)

  def __set_time(self):
    self.time = float(self.camera_f.timeText.text())
  
  def __set_hand(self):
    self.is_hand_paint = True
    self.pos = self.__trans_pos(self.paintBoard.pos_xy)
  
  # pos_list: list
  # return: list
  def __trans_pos(self, pos_list):
    lit=[]
    for i in range(len(pos_list)):
      ret = []
      # if((pos_list[i][0]) <= 200):
      #   # ret.append(int(np.arccos(((200-pos_list[i][0])/200)/np.pi*180+90)/1.40625))
      #   ret.append(int(  (np.arccos(pos_list[i][0]-200/200)/np.pi*180)/1.40625  ))
      # else:
      ret.append(int((np.arccos((pos_list[i][0]-200)/200)/np.pi*180)/1.40625))
      ret.append(int(np.arccos((200-pos_list[i][1])/200)/np.pi*180/1.40625))
      lit.append(ret)
    return lit

  def __get_path(self):
    self.file_path = QFileDialog.getOpenFileNames(self, '选择文件')[0]
    self.camera_f.pathText.setText(self.file_path)
    self.__read_file_pos(self.file_path)
    self.is_hand_paint = False
  
  def __read_file_pos(self, path):
    with open(path, "r") as f:
      c = f.read()
    self.pos.clear()
    for i in c:
      p = c.split(",")
      self.pos.append((int(p[0], int(p[1]))))
  
  def __single(self):
    self.startw.root.cameaImgButton.click()
    time.sleep(0.005)
    self.queue.put((ImageModel.SINGLE, ShowGraph.CAMERA))
  
  def run(self):
    while True:
      if self.start_flag:
        self.startw.root.cameaImgButton.click()
        self.queue.put((ImageModel.SINGLE, ShowGraph.CAMERA))
      else:
        time.sleep(0.01)

if __name__ == "__main__":
  a = np.random.randint(0, 255, [1024, 1024, 1], dtype=np.uint8)
  img = Image.fromarray(a)
  img.show()