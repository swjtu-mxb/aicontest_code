from pdb import runcall
from sqlite3 import Row
import threading
from tkinter import *

from src.utils.process import AppProcess, getFromQueueFirst
from PIL import Image, ImageTk
from functools import partial
import queue

from tkinter import filedialog
import time
import os

img = None
im = None

from src.proto.host import cmd_pb2
from src.host.status import sta, RunCls
# from src.host.run_main import stub, yolo, change, img_cmp
from src.host.run_main import glb

class ControllerFrame(Frame):
  def __init__(self, root):
    self.root = root
    super().__init__(root)
    self.img_p = Image.open("datas/gui/PYNQ.png").resize([170, 400])
    self.img = ImageTk.PhotoImage(self.img_p)
    self.enable = [True for i in range(4)]
    self.load_bits = [BooleanVar() for i in range(4)]
    self.plist = []
    self.load_path = StringVar()
    self.create_button()
    self.create_bit()
  
  def create_button(self):
    Label(self).grid(row=0, column=0)
    self.p1 = Button(self, text="text1", image=self.img, command=partial(self.setEnable, 0))
    self.p2 = Button(self, text="text2", image=self.img, command=partial(self.setEnable, 1))
    self.p3 = Button(self, text="text3", image=self.img, command=partial(self.setEnable, 2))
    self.p4 = Button(self, text="text4", image=self.img, command=partial(self.setEnable, 3))
    self.plist = [self.p1, self.p2, self.p3, self.p4]
    for m in range(len(self.plist)):
      self.plist[m].configure(command=partial(self.enable_pynq, m))
      self.plist[m].grid(row=0, column=m, padx=35)
    self.set_bg()
  
  def create_bit(self):
    Checkbutton(self, variable=self.load_bits[0], onvalue=True, offvalue=False).grid(row=1, column=0, padx=35, pady=10)
    Checkbutton(self, variable=self.load_bits[1], onvalue=True, offvalue=False).grid(row=1, column=1, padx=35, pady=10)
    Checkbutton(self, variable=self.load_bits[2], onvalue=True, offvalue=False).grid(row=1, column=2, padx=35, pady=10)
    Checkbutton(self, variable=self.load_bits[3], onvalue=True, offvalue=False).grid(row=1, column=3, padx=35, pady=10)
    self.load_entry = Entry(self, width='80', textvariable=self.load_path)
    self.load_entry.grid(row=2, column=1, columnspan=2)
    Button(self, text="更新Bits：", command=self.unload_bits, width=20).grid(row=2, column=0, pady=10)
    Button(self, text="Start code bits", command=self.start_upload_bit, bg="red", width=20).grid(row=3, pady=10)

  '''
  idx: enable or disable PYNQ index
  '''
  def setEnable(self, idx):
    glb.stub.setPYNQ(cmd_pb2.PYNQEnableCmd(enable=self.load_bits[idx].get(), idx=idx))
    print(self.load_bits[idx].get())
    print("enable ", idx)
  
  def start_upload_bit(self):
    idx_list = []
    for i in range(len(self.load_bits)):
      if self.load_bits[i].get():
        idx_list.append(i)
    cmd = cmd_pb2.PYNQBitsCmd()
    cmd.idx.extend(idx_list)
    with open(self.load_path.get(), "rb") as f:
      bits = f.read()
    cmd.bits = bits
    glb.stub.uploadPYNQ(cmd)
  
  def unload_bits(self):
    selectFile = filedialog.askopenfilename()  # askopenfilename 1次上传1个；askopenfilenames1次上传多个
    self.load_entry.insert(0, selectFile)
    print(self.load_path.get())
  
  def set_bg(self):
    for k, v in zip(self.enable, self.plist):
      if k: 
        v.configure(bg="green")
      else:
        v.configure(bg="white")
  
  def enable_pynq(self, idx):
    self.enable[idx] = not self.enable[idx]
    self.set_bg()

class CollFrame(Frame):
  def __init__(self, root: Tk):
    self.root = root
    super().__init__(self.root)
    self.canvas = Canvas(self.root, bg='white', width=512, height=512)
    self.creat_frame()

    self.trigger_queue = queue.Queue(5)
    self.get_img_th = threading.Thread(target=self.get_image_and_run)
    self.get_img_th.start()
    self.save_path = "datas/images/receive"

    self.monitor_th = threading.Thread(target=self.run_monitor)
    self.monitor_th.start()

    self.is_continue = False
  

  def run_monitor(self):
    while True:
      res = glb.stub.getMonitorResult(cmd_pb2.Empty(isOK=True))
      if res.isOK:
        self.trigger_queue.put(0) # put 0 
      time.sleep(0.1)
  
  def __run_app(self, img):
    if (sta.get_app_cls() == RunCls.CHANGE) and sta.is_run_cnn():
      glb.change.queue.put(img)
      if glb.change.isStart == False:
        glb.change.isStart = True
        glb.change.process.start()

  def get_image_and_run(self):
    flag = 0 
    while True:
      if flag == 2:
        image = glb.img_cmp.get_img()
        if not self.trigger_queue.empty():
            if self.trigger_queue.get() == 1:
              flag = 0
      else:
        flag = self.trigger_queue.get(True)
        image = glb.img_cmp.get_img()
      self.__run_app(image)
      self.show_image(image[0], image[1])
      if flag == 2:
        time.sleep(0.1)
      
  # image: numpy array
  def show_image(self, image, name):
    global img, im
    img = Image.fromarray(image)
    img.save(os.path.join(self.save_path, name + ".jpg"))
    img = img.resize((512, 512))
    im = ImageTk.PhotoImage(image=img)
    self.canvas.create_image(0, 0, anchor='nw', image=im)
    self.root.update()
    self.root.after(1000)
  
  def creat_frame(self):
    remote_func = partial(self.set_image_type, cmd_pb2.ImageClass(imgType=cmd_pb2.REMOTE_SENSING))
    land_func = partial(self.set_image_type, cmd_pb2.ImageClass(imgType=cmd_pb2.LANDSLIDE))
    camera_func = partial(self.set_image_type, cmd_pb2.ImageClass(imgType=cmd_pb2.CAMERA))

    Label(self, text="数据类别").grid(row=0, column=1, pady=10)
    Button(self, text="变化检测遥感图", command=remote_func).grid(row=1, column=0, padx=20, pady=10)
    Button(self, text="目标检测遥感图", command=land_func).grid(row=1, column=2,  padx=20, pady=10)
    # Button(self, text="相机", command=camera_func).grid(row=1, column=2, padx=20, pady=10)

    Label(self, text="检测模式").grid(row=2, column=1, pady=10)
    Button(self, text="单次", command=partial(self.set_run_times, 1), width=13).grid(row=3, column=0, pady=10, padx=20)
    Button(self, text="连续", command=partial(self.set_run_times, 2), width=13).grid(row=3, column=2, pady=10, padx=20)
    # Button(self, text="监控", command=self.run_camera).grid(row=3, column=2, pady=10, padx=20)

    Label(self, text="检测").grid(row=4, column=1, pady=10)
    Button(self, text="None", command=self.disable_run_cnn, width=13).grid(row=5, column=0, pady=10, padx=20)
    Button(self, text="目标检测", command=partial(self.set_run_app, RunCls.FACE), width=13).grid(row=5, column=2, pady=10, padx=20)
    Button(self, text="变化检测", command=partial(self.set_run_app, RunCls.CHANGE), width=13).grid(row=5, column=1, pady=10, padx=20)
  
  def disable_run_cnn(self):
    sta.disable_run_cnn()
  
  def set_run_app(self, cmd):
    sta.enable_run_cnn()
    sta.set_app_cls(cmd)
  
  def set_run_times(self, cmd):
    if cmd == 2:
      glb.stub.setContinue(cmd_pb2.Empty(isOK=True))
      self.is_continue = True
    elif self.is_continue:
      glb.stub.setNoContinue(cmd_pb2.Empty(isOK=False))
      self.is_continue = False
    self.trigger_queue.put(cmd)
  
  def pack(self, **kwargs):
    super().pack(**kwargs)
    self.canvas.place(x=470, y=(720 - 512) // 2)
  
  def pack_forget(self) -> None:
    self.canvas.place_forget()
    return super().pack_forget()

  def set_image_type(self, cls):
    glb.stub.setImage(cls) 

  def run_hp(self):
    global img, im
    img = Image.open("datas/images/testImage/dog.jpg")
    img = img.resize((512, 512))
    im = ImageTk.PhotoImage(image=img)
    self.canvas.create_image(0, 0, anchor='nw', image=im)
    self.root.update()
    self.root.after(1000)

  def run_camera(self):
    print("remote camera")

class FpgaFrame(Frame):
  def __init__(self, root):
    self.root = root
    super().__init__(self.root)
    self.hand_code = ""
    self.is_hand_write = False
    self.code_path = StringVar()
    self.constrain_path = StringVar()
    self.create_text()

  def create_text(self):
    Label(self, text="实现你的代码").grid(row=0, column=0, columnspan=2)
    self.text = Text(self, width=60, height=30)
    self.text.grid(row=1, column=0, columnspan=2)
    Button(self, text="保存代码", command=self.save_code).grid(row=2, column=0, columnspan=2)

    # 选择 path
    Button(self, text="或者选择上传的代码：", command=self.unload_bits, width='20').grid(row=3, column=0)
    self.load_entry = Entry(self, textvariable=self.code_path, width='100')
    self.load_entry.grid(row=3, column=1)

    Button(self, text="选择约束文件：", command=self.unload_constrain, width='20').grid(row=4, column=0)
    self.constrain_entry = Entry(self, textvariable=self.constrain_path, width='100')
    self.constrain_entry.grid(row=4, column=1)

    Button(self, text="编译下载", command=self.compile_download, bg='red', width='20').grid(row=5, column=0)
  
  def unload_constrain(self):
    selectFile = filedialog.askopenfilename()  # askopenfilename 1次上传1个；askopenfilenames1次上传多个
    self.constrain_entry.insert(0, selectFile)
    print(self.constrain_path.get())

  def unload_bits(self):
    selectFile = filedialog.askopenfilename()  # askopenfilename 1次上传1个；askopenfilenames1次上传多个
    self.load_entry.insert(0, selectFile)
    print(self.code_path.get())
  
  def compile_download(self):
    if self.is_hand_write:
      code = self.hand_code
    else:
      with open(self.code_path.get(), "r") as f:
        code = f.read()

    with open(self.constrain_path.get(), "r") as f:
      constrain = f.read()
    code = bytes(code, encoding = "utf8")
    constrain = bytes(constrain, encoding = "utf8")
    cmd = cmd_pb2.LatticeBitsCmd(code=code, constrain=constrain)
    glb.stub.uploadLattice(cmd)
    print("compile download")

  def upload_code(self):
    print("upload")
  
  def save_code(self):
    self.is_hand_write = True
    self.hand_code = self.text.get("1.0", "end") 
    print(self.hand_code)
