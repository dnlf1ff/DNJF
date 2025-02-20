from loguru import logger
import sys
import os
from util import make_dir
import time

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Logger(metaclass=Singleton):
    SCREEN_WIDTH=120

class LogCapture:
    def __init__(self, prefix=""):
        self.prefix = prefix
        self.buffer = []
    
    def write(self, msg):
        if msg.strip():
            self.buffer.append(msg.strip())
            logger.info(f"{self.prefix} {msg.strip()}")
    
    def flush(self):
        pass

    def close(self):
        pass
    
def log_ase(logfile):
    if os.path.exists(logfile):
        with open(logfile, 'r') as f:
            for line in f:
                logger.debug(f"ASE LOG: {line.strip()}")

# todo: make loggerfunction
# TODO: custom logs

def get_logger(system, logfile, job):
    if logfile is None:
        logfile = 'dnjf.log'
    
    path = make_dir(os.path.join(os.environ['LOG'], job))
    log_filename=os.path.join(path, logfile)
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    # logger.add(log_filename, level="OWELL", format="[{time:HH:mm:ss}] | {level} | {message} | [{time:DD-MM-YYYY}]", rotation="10MB", retention="7days")
    return logger
