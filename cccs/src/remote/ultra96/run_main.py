import multiprocessing
from multiprocessing.dummy import Process
from src.utils.process import AppProcess
from src.utils.check import check_fifo

import os
import cv2
import numpy as np
from multiprocessing import Process, Queue

# from rpyc import Service
# from rpyc.utils.server import ThreadedServer

import serial
import binascii, time
from simple_pid import PID
#import matplotlib.pyplot as plts
from multiprocessing import Process, Queue
import pdb
# 01  稍快
# 02  慢 顺时针
# 04  停止
# 08  慢
# 0a  稍快

x_des_point = 0.5
y_des_point = 0.5

x_delta = 0.
x_last_delta = 0.
x_target = 0.5
kp = -150
# kp = 0
kd = 80
x_data = 64

y_delta = 0.
y_last_delta = 0.
y_target = 0.5
ykp = 150
# ykp = 0
ykd = 80
y_data = 32

def serial_tx(data):
    ser = serial.Serial(port='/dev/console', baudrate=115200)
    send_data = bytes.fromhex(data)
    result = ser.write(send_data)
    count = ser.inWaiting()
    ser.flushInput()
    ser.close()

def run_pid(x_current_point, y_current_point):
    global x_delta, kp, kd, x_last_delta, x_data
    x_last_delta = x_delta
    x_delta = x_current_point - x_target
    output = kp * x_delta + kd * (x_delta - x_last_delta)
    if ((x_data == 1) & (output <-20) | (x_data == 127) & (output > 20)) | (abs(x_delta) < 0.05):
        x_data = x_data
    elif output < -20:
        x_data = x_data + 1
    elif output > 20:
        x_data = x_data - 1
    else:
        x_data = x_data

    global y_delta, ykp, ykd, y_last_delta, y_data
    y_last_delta = y_delta
    y_delta = y_current_point - y_target
    youtput = ykp * y_delta + ykd * (y_delta - y_last_delta)
    print('out=',youtput)
    
    if ((y_data == 1) & (youtput < -20)) | ((y_data == 63) & (youtput > 20)) | (abs(y_delta) < 0.05):
        y_data = y_data
    elif youtput > 20:
        y_data = y_data + 1
    elif youtput < -20:
        y_data = y_data - 1
    else:
        y_data = y_data
    print('ydata=',y_data)
    xy_data = 'ff'+str("{:02X}".format(x_data))+''+str("{:02X}".format(y_data))+'aa'
    serial_tx(xy_data)

class Ultra96Detect:
    def __init__(self):
        self.write_fifo_name = "/tmp/ultra96_write"
        self.write_fifo = check_fifo(self.write_fifo_name)
        self.read_fifo_name = "/tmp/ultra96_read"
        self.read_fifo = check_fifo(self.read_fifo_name)
        self.model_path = "densebox_320_320"
        self.process = AppProcess("test", self.run_ultra)
        # self.process.start()
        self.signal = Queue(10)
        self.x_base = (1280 - 512) // 2
        self.y_base = (720 - 512) // 2
        self.__init_camera()
        os.system("cd /home/root/Vitis-AI/demo/Vitis-AI-Library/samples/facedetect && ./test_facedetec_server {} {} {} &".format(self.model_path, self.write_fifo_name, self.read_fifo_name))
        # self.get_img()

    
    def run_ultra(self):
        while True:
            self.signal.get()
            
    
    def __init_camera(self):
        self.camera = cv2.VideoCapture(0)
        self.camera.set(3, 1280)  # width=1920
        self.camera.set(4, 720)  # height=1080
    
    def get_img(self):
       ret, frame = self.camera.read()
       cv2.waitKey(2)
       res = frame[self.y_base:self.y_base+512, self.x_base:self.x_base+512]
       return res    

    def exposed_capture(self, ctrl):
        if ctrl:
            return self.get_img()
        else:
            return False
            
    def exposed_detect(self, ctrl):
        pass

    def exposed_coordinate(self, ctrl):
        pass
    
    def draw(self, arr, img):
        h = img.shape[0]
        w = img.shape[1]
        arr = arr.reshape([-1, 4])
        for i in range(arr.shape[0]):
            m = arr[i]
            start = ((int)(w * m[0]), (int)(h * m[1]))
            end = ((int)(w * m[2]) + start[0], (int)(h * m[3]) + start[1])
            #print(start, end)
            img = cv2.rectangle(img, start, end, (0,255,0), 2)
        return img
    
    # img: numpy arr
    def run_ultra96(self, img):
        byte = img.tobytes(order='C')
        # write
        # self.signal.put(True)
        #print("byte len = {}".format(len(byte)))

        f = os.open(self.write_fifo_name, os.O_WRONLY)
        for i in range(0, len(byte), 1 << 16):
            os.write(f, byte[i:i+(1 << 16)])
        os.close(f)

        # read 
        with open(self.read_fifo_name, "rb") as f:
            res = f.read()
        print(res)
        if len(res) > 4:
            arr = np.frombuffer(res, dtype=np.float32)
            return arr
        else:
            return False
    

def demo():
    obj = Ultra96Detect()
    i = 0
    x_point = []
    y_point = []
    times = []
    while True:

        try:
            img = obj.get_img()  
            arr = np.array(img, dtype=np.uint8)
            #print(arr.shape)
            #print("send img")
            start = time.time()
            res = obj.run_ultra96(arr)
            end = time.time()
            #print("run time = {:.2f}".format(end - start))
            if not isinstance(res, bool):
                img = obj.draw(res, img)
                res = res.reshape([-1, 4])
                m = res[0]
                x = m[0] + m[2] / 2
                y = m[1] + m[3] / 2
                # print('{} {}'.format(x,y))
                x_point.append(x)
                y_point.append(y)
                print('x=',x)
                print('y=',y)
                run_pid(x_pid, y_pid, x, y)
                i = i + 1
                times.append(i)
                # send res
            cv2.imshow("test", img)
            #print(res)
        except KeyboardInterrupt:
            obj.camera.release()
            return

    
if __name__ == "__main__":
    demo()