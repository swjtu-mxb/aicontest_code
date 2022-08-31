from src.qt.ui.fpga import Ui_Form as FPGA_From
from src.qt.ui.pynq import Ui_Form as PYNQ_From
from src.qt.ui.start import Ui_MainWindow as Start_MainWindow

from src.qt.startQt import StartQt
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from src.qt.observer import *


class MyMainWindow(QMainWindow):
  def __init__(self) -> None:
    super().__init__()
    self.ui = Start_MainWindow()
    self.ui.setupUi(self)
    self.startQt = StartQt(self.ui)

    self.imdSlover = IMDSolver(self.ui)
    self.detectTh = DetectThread(self.ui)
    self.getimgTh = GetImageThread(self.ui, self.startQt.camera_f.graphicsView)
    self.dialSlover = DialSlover(self.startQt)
    self.cameraSlover = CameraThread(self.startQt, self.getimgTh.queue_mode)
    self.getimgTh.start()
    self.detectTh.start()

if __name__ == '__main__':
  app = QApplication(sys.argv)
  MainWindow = MyMainWindow()
  
  MainWindow.show()
  sys.exit(app.exec_())
