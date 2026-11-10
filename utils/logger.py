"""Console + file logger; every line starts with the seconds since the program started."""
import logging
import sys
import time


class _Elapsed(logging.Formatter):
    start = time.time()

    def format(self, record):
        return "[%8.3f] %s" % (time.time() - self.start, record.getMessage())


def setup_logger(log_file=None):
    logger = logging.getLogger("placer")
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):             # a second call in one process: close the previous log file
        logger.removeHandler(h)
        h.close()
    logger.propagate = False
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode="w"))
    for h in handlers:
        h.setFormatter(_Elapsed())
        logger.addHandler(h)
    return logger
