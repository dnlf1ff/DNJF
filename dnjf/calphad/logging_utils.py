from loguru import logger
import sys
import os

from mob_utils import *

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

def get_logger(logfile=None):
    if logfile is None:
        logfile = 'dnjf.log'

    log_filename=os.path.join(os.environ['LOG'], logfile)
    logger.remove()
    try:
        logger.level("FROST", no=15, color="<white><italic>")
        logger.level("TLQKF", no=8, color="<red><bold>")
    except Exception as e:
        print(e)
    logger.add(sys.stdout, level="INFO")
    logger.add(log_filename, level="TLQKF", format="[{time:HH:mm:ss}] | {level} | {message} | [{time:DD-MM-YYYY}]", rotation="10MB", retention="7days")
    return logger
