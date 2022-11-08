#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2022 Fisher. All rights reserved.
#   
#   文件名称：coref_parser.py
#   创 建 者：YuLianghua
#   创建日期：2022年07月07日
#   描    述：
#
#================================================================

from coref_parserSDK import CorefParseClient


class CorefParser(object):
    def __init__(self, hosts, timeout=3):
        self.coref_clinet = CorefParseClient(hosts)
        self.timeout = timeout

    def __call__(self, text):
        ret = self.coref_clinet.parse(text, timeout=self.timeout)
        if not ret.success:
            return {}
        return ret.result.corefmap
