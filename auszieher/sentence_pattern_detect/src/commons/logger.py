# -*- coding: utf-8 -*-
"""
Author: weisi.wu
Date: 2022/5/26
Note: logger module
Version: v1.0
"""

import logging
import sys
from loguru import logger

from .configer import config
from ..utils import check_dir

import os
log_dir = os.path.join(os.path.dirname(__file__), '../../', config["LOG_DIR"])
# at first, check if the log dir exists
check_dir(log_dir)

class InterceptHandler(logging.Handler):
    def emit(self, record):
        logger_opt = logger.opt(depth=6, exception=record.exc_info)
        msg = self.format(record)
        logger_opt.log(record.levelno, msg)


logging.basicConfig(handlers=[InterceptHandler()], level=0)
logger.configure(handlers=[{"sink": sys.stderr, "level": "INFO"}])
logger.add(
    f"{ log_dir }/server.log",
    rotation="100 MB",
    encoding="utf-8",
    colorize=False,
    level="INFO"
)
