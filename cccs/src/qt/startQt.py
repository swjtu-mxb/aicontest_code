from src.qt.ui.fpga import Ui_Form as FPGA_From
from src.qt.ui.pynq import Ui_Form as PYNQ_From
from src.qt.ui.camera import Ui_Form as CAMERA_From
from src.qt.ui.image import Ui_Form as IMG_From
from src.qt.ui.start import Ui_MainWindow as Start_MainWindow

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QFont, QPixmap, QPen, QPainter, QColor
from PyQt5.Qt import QSize
from PyQt5.QtGui import QPixmap

class PaintFrame(QFrame):
  def __init__(self) -> None:
    super().__init__()
    self.pos_xy = []
    self.setMouseTracking(False)
  
  def setStartQt(self, camera: CAMERA_From):
    self.camera = camera
    self.__size = QSize(400, 200)

    self.board = QPixmap(self.__size)
    self.board.fill(Qt.white)
    self.__painter = QPainter()
    self.__pen = QPen(Qt.red, 2, Qt.SolidLine)
    self.__painter.setPen(self.__pen)
  
  def paintEvent(self, event):
    self.__painter.begin(self)
    self.__painter.drawPixmap(0, 0, self.board)
    if len(self.pos_xy) > 1:
        point_start = self.pos_xy[0]
        for pos_tmp in self.pos_xy:
            point_end = pos_tmp
            if self.__get_dist(point_start, point_end) < 20:
              self.painter.drawLine(point_start[0], point_start[1], point_end[0], point_end[1])
            point_start = point_end
    self.__painter.end()
    self.setGrpah(self.board)
    print("paint")

  def setGrpah(self, pixmap):
    item = QGraphicsPixmapItem(pixmap)  
    scene= QGraphicsScene() 
    scene.addItem(item)
    self.camera.paintGraph.setScene(scene) 

  def mouseMoveEvent(self, event):
    pos_tmp = (event.pos().x(), event.pos().y())
    print("hello")
    self.pos_xy.append(pos_tmp)
    self.update()
  

class StartQt:
  def __init__(self, root: Start_MainWindow):
    self.root = root
    self.__init_frame()
    self.__set_init_button()
 
  def __init_frame(self):
    font = QFont()
    font.setPointSize(12)

    self.pynq_action = QAction(self.root.menubar)
    self.pynq_action.setText("PYNQ Controll")
    self.pynq_action.triggered.connect(self.run_pynq)
    self.root.menubar.addAction(self.pynq_action)  #添加子菜单
    self.pynq_frame = QFrame()
    self.pynq_frame.setWindowTitle("PYNQ Controll")

    self.pynq_f = PYNQ_From()
    self.pynq_f.setupUi(self.pynq_frame)
    pix = QPixmap('datas/gui/pynq_icon.jpg')
    self.pynq_f.pynqImage.setPixmap(pix)

    self.fpga_action = QAction(self.root.menubar)
    self.fpga_action.setText("FPGA Controll")
    self.fpga_action.triggered.connect(self.run_fpga)
    self.root.menubar.addAction(self.fpga_action)
    self.fpga_frame = QFrame()
    self.fpga_frame.setWindowTitle("FPGA Controll")

    self.fpga_f = FPGA_From()
    self.fpga_f.setupUi(self.fpga_frame)

    self.camera_action = QAction(self.root.menubar)
    self.camera_action.setText("Camera")
    self.camera_action.triggered.connect(self.run_camera)
    self.root.menubar.addAction(self.camera_action)

    self.camera_frame = QFrame()
    self.camera_frame.setWindowTitle("Camera Controller")
    self.camera_f = CAMERA_From()
    self.camera_f.setupUi(self.camera_frame)

    self.img_frame = QFrame()
    self.img_f = IMG_From()
    self.img_f.setupUi(self.img_frame)
  
  def __set_init_button(self):
    # start 界面
    self.root.changeImgButton.setChecked(True)
    self.root.changeButton.setChecked(True)
    self.root.pynqButton.setChecked(True)
    self.root.pynqDial.setValue(8)
    self.root.pynqNumLabel.setText(str(self.root.pynqDial.value()))
    # self.root.pynqDial.valueChanged.connect(lambda :self.root.pynqNumLabel.setText(str(self.root.pynqDial.value())))

    # PYNQ 界面
    self.pynq_f.ch0.setChecked(True)
    self.pynq_f.ch1.setChecked(True)
    self.pynq_f.ch2.setChecked(True)
    self.pynq_f.ch3.setChecked(True)
    self.pynq_f.ch4.setChecked(True)
    self.pynq_f.ch5.setChecked(True)
    self.pynq_f.ch6.setChecked(True)
    self.pynq_f.ch7.setChecked(True)
    self.pynq_f.number.setText('8')
  
  def run_camera(self):
    self.camera_frame.show()
  
  def run_pynq(self):
    self.pynq_frame.show()

  def run_fpga(self):
    self.fpga_frame.show()
