import sys
import os
import logging
import json

PACKAGE_PARENT = '.'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
PREFIX_PATH = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-2]) + "/"
sys.path.append(PREFIX_PATH)

logging.basicConfig(filename=PREFIX_PATH + 'logs/unseen_class_accuracy.log', format='%(asctime)s %(message)s', encoding='utf-8', level=logging.DEBUG)
logger = logging.getLogger("UCAlogger")