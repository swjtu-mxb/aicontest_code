from copy import copy
import os 
import shutil
from PIL import Image

def check_dir(path):
  if not os.path.exists(path):
    os.mkdir(path)

def split(path, save_path):
  all = os.listdir(path)
  for m in all:
    if m.endswith(".tif"):
      idx = m.split("_")[0]
      idx = (int)(idx)
      idx = idx // 30
      sub = os.path.join(save_path, "sub_" + str(idx))
      check_dir(sub)
      final_path = os.path.join(sub, m)
      img = Image.open(os.path.join(path, m))
      img.save(final_path.split(".")[0] + ".jpg")

if __name__ == "__main__":
  split("datas/datasets/images", "datas/datasets/split_images")