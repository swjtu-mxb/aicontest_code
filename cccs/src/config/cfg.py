QUEUE_SIZE = 10
PORT = "5678"
PYNQ_IP = [
  # "192.168.3.153",
  # "192.168.3.105"
  # "192.168.3.35",
  # "192.168.3.31",
  # "192.168.3.32",
  "192.168.3.34",
  # "192.168.3.32",
]
PYNQ_NUM = len(PYNQ_IP)
PYNQ_PORT = list(map(lambda x: x + ":" + PORT, PYNQ_IP))
PYNQ_IMG_SIZE = 256
USB_DEVICE = "/dev/ttyUSB0"

REMOTE_IP = "localhost"
REMOTE_PORT = REMOTE_IP + ":" + "6636"

# U50_PORT = "192.168.3.107" + ":" + PORT
U50_PORT = "localhost" + ":" + PORT

ULTRA_96_IP = "192.168.3.155"
ULTRA_96_PORT = ULTRA_96_IP + ":" + PORT

DECOMPRESS_PROCESS_NUM = 6