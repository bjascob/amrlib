import os
import atexit
import logging
from   logging import DEBUG, INFO, WARN, ERROR


def setup_logging(logfname=None, level=None):
    # Remove any existing handler (ie.. penman has logging.basicConfig() in __init__.py)
    # Note that in python 3.6 there is no "force" in basicConfig()
    # From https://stackoverflow.com/questions/12158048/changing-loggings-basicconfig-which-is-already-set
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # Setup the logger
    if level is None:
        level  = logging.INFO
    format = '[%(levelname)s %(filename)s ln=%(lineno)s] %(message)s'
    if logfname is not None:
        os.makedirs(os.path.dirname(logfname), exist_ok=True)
        logging.basicConfig(level=level, filename=logfname, filemode='w', format=format)
    else:
        logging.basicConfig(level=level, format=format)


# Penman spews a lot of messages
def silence_penman():
    logging.getLogger('penman').setLevel(logging.ERROR)
    #logging.getLogger('penman.layout').setLevel(logging.ERROR)
    #logging.getLogger('penman._lexer').setLevel(logging.ERROR)
    logging.getLogger('pe').setLevel(logging.ERROR)     # pre v1.1, penman._parse.py


# Silense the request library
def silence_requests():
    logging.getLogger('urllib3.connectionpool').setLevel(logging.ERROR)


# smatch/amr logs to stderr.  Redirect to a file
def setup_smatch_log(fpath):
    import amr
    import smatch
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    smatch_log_f = open(fpath, 'w', buffering=1)    # line buffering
    atexit.register(smatch_log_f.close)
    amr.ERROR_LOG    = smatch_log_f
    amr.DEBUG_LOG    = smatch_log_f
    smatch.ERROR_LOG = smatch_log_f
    smatch.DEBUG_LOG = smatch_log_f
