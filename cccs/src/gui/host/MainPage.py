from multiprocessing import Queue
import tkinter as tk
import time

from src.gui.host.CollectionFrame import *
from src.utils.process import AppProcess, getFromQueueFirst
from src.host.run_main import init_host

class MainPage:
  def __init__(self, root: tk.Tk):
    self.root = root
    self.root.geometry("1100x720")
    # init_host()
    # self.root.resizable(width=False, height=False)
    self.create_menu()

    self.collo = CollFrame(self.root)
    self.collo.pack(side="left", expand='no')
    self.controll = ControllerFrame(root)
    self.fpga = FpgaFrame(root)
    # self.collo.grid()
    self.root.mainloop()

  def create_menu(self):
    menuBar = tk.Menu(self.root)
    menuBar.add_command(label="采集系统", command=self.toCollection)
    menuBar.add_command(label="PYNQ控制", command=self.toControll)
    menuBar.add_command(label="FPGA控制", command=self.toFpga)
    self.root['menu'] = menuBar
  
  def toCollection(self):
    self.root.geometry("1100x720")
    self.controll.pack_forget()
    self.collo.pack(side="left", expand='no')
    self.fpga.pack_forget()

  def toFpga(self):
    self.root.geometry("960x720")
    self.controll.pack_forget()
    self.collo.pack_forget()
    self.fpga.pack()

  def toControll(self):
    self.root.geometry("1100x720")
    self.collo.pack_forget()
    self.controll.pack()
    self.fpga.pack_forget()

  def create_page(self):
    pass

if __name__ == "__main__":
  root = tk.Tk()
  m = MainPage(root)