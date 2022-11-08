#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2022 Fisher. All rights reserved.
#   
#   文件名称：logger.py
#   创 建 者：YuLianghua
#   创建日期：2022年05月09日
#   描    述： logger 
#
#================================================================

import sys
from loguru import logger

logger.remove()
fmt = '<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>'

logger.add(sys.stderr, format=fmt, level="INFO")
