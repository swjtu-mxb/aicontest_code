# aicontest_code
---
PaddleDetection为目标检测  
BIT-CD-Paddle为变换检测  
ImageComp为图像压缩部分  
训练相关的数据集如下：

图像压缩数据集： http://r0k.us/graphics/kodak/

变化检测数据集： https://justchenhao.github.io/LEVIR/

目标检测（滑坡）数据集： https://pan.baidu.com/s/1950sOcFfDFU6UWz-Dsm_7Q 提取码：mnd6

---
cccs为硬件系统的整体代码，其中

./src/remote 文件夹为其他各部分硬件所需运行的代码

./src/remote/alveo 是U50加速卡所需运行的代码，实现目标检测

./src/remote/compress 是PYNQ集群所需运行的代码，实现图像压缩

.src/remote/ultra96 是ultra96所需运行的代码，实现在轨AI目标检测
