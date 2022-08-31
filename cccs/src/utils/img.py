import cv2
from PIL import Image, ImageDraw
import numpy as np

def letterbox_image(image_src, dst_size, pad_color=(114, 114, 114)):
  """
  缩放图片，保持长宽比。
  :param image_src:       原图(numpy)
  :param dst_size:        (h, w)
  :param pad_color:       填充颜色，默认是灰色
  :return:
  """
  src_h, src_w = image_src.shape[:2]
  dst_h, dst_w = dst_size
  scale = min(dst_h / src_h, dst_w / src_w)
  pad_h, pad_w = int(round(src_h * scale)), int(round(src_w * scale))
  
  if image_src.shape[0:2] != (pad_w, pad_h):
      image_dst = cv2.resize(image_src, (pad_w, pad_h), interpolation=cv2.INTER_LINEAR)
  else:
      image_dst = image_src
  
  top = int((dst_h - pad_h) / 2)
  down = int((dst_h - pad_h + 1) / 2)
  left = int((dst_w - pad_w) / 2)
  right = int((dst_w - pad_w + 1) / 2)
  
  # add border
  image_dst = cv2.copyMakeBorder(image_dst, top, down, left, right, cv2.BORDER_CONSTANT, value=pad_color)
  
  x_offset, y_offset = max(left, right) / dst_w, max(top, down) / dst_h
  return image_dst, x_offset, y_offset

# img: numpy arr 
# position: 
def draw(img, position):
  img = Image.fromarray(img)
  draw = ImageDraw.Draw(img) # 创建 Draw 对象
  for i in range(position[0]):
    start = i * 4 + 1
    axi = position[start:start+4]
    axi = tuple(map(lambda x: float(x), axi))
    draw.rectangle(axi, outline=(255, 0, 0))
  return np.array(img)
  
if __name__ == "__main__":
  img = cv2.imread("datas/images/testImage/dog.jpg")
  dst, _, _ = letterbox_image(img, dst_size=[512, 512])
  cv2.imwrite("datas/images/testImage/dog_clip.jpg", dst)