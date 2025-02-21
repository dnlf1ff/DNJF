from loguru import logger
import sys
import os

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
    
    path = os.path.join(os.environ['LOG'], job,system.lower())
    os.makedirs(path, exist_ok=True)
    log_filename=os.path.join(path, logfile)
    logger.remove()
    try:
        logger.level("FROST", no=15, color="<white><italic>")
        logger.level("OWELL", no=8, color="<red><bold>")
    except Exception as e:
        pass

    logger.add(sys.stdout, level="INFO")
    logger.add(log_filename, level="OWELL", format="[{time:HH:mm:ss}] | {level} | {message} | [{time:DD-MM-YYYY}]", rotation="10MB", retention="7days")
    return logger
