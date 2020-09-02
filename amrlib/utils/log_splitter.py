import os
import sys
import logging

logger = logging.getLogger(__name__)


# This simple class is designed to allow console output to additionally be
# written to a separate file.  It is intended for user messages which need to be captured.
# This does not replace the "logging" utility which is intended for lower-level messages.
# Statements sent to this class can also be re-directed to the system log
class LogSplitter(object):
    def __init__(self, fname=None, path=None, mode='w', to_logging=False):
        self.logfh      = None
        self.to_logging = to_logging
        if path is not None and fname is not None:
            fname = os.path.join(path, fname)
        if fname:
            self.logfh = open(fname, mode)

    def print(self, string='', end='\n'):
        sys.stdout.write(string + end)
        if self.logfh:
            self.logfh.write(string + end)
            self.logfh.flush()
        if self.to_logging:
            logger.info(string)

    def close(self):
        self.logfh.close()
