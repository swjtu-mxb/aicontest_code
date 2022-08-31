import os

# 检测是否存在 不存在就生成
def check_fifo(path):
  if not os.path.exists(path):
    os.mkfifo(path)

def delete_fifo(path):
  if os.path.exists(path):
    os.remove(path)