from src.config.cfg import PORT
from src.pynq.pynq_service import CnnService
from rpyc.utils.server import ThreadedServer
import time

def serve():
  server = ThreadedServer(service=CnnService, port = (int)(PORT),auto_register=False) 
  server.start()
  try:
    while True:
      time.sleep(.6)
  except  KeyboardInterrupt:
    server.stop(0)
  
if __name__ == "__main__":
  serve()
