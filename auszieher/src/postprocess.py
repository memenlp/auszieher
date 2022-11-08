#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2022 Fisher. All rights reserved.
#   
#   文件名称：postprocess.py
#   创 建 者：YuLianghua
#   创建日期：2022年06月24日
#   描    述：对抽取结果进行后置处理，包含：
#             1. 情感值重置；
#             2. 过滤自定义无意义结果；
#
#================================================================

from typing import List

from ..src import dataset
from ..src import ExtractResult
from ..utils.logger import logger

class PostProcessor(object):
    def __init__(self):
        self.tag_normer = None

    def filte(self, holder, emotion, fobject, reason):
        uniqk = '\t'.join([holder.text, emotion.text, fobject.text, reason.text]).lower()
        return True if \
            uniqk in dataset.filted_case else False

    def modify(self, fobject, reason, ori_score):
        uniqk = '\t'.join([fobject.text, reason.text]).lower()
        return dataset.modify_case.get(uniqk, ori_score)

    def normtag(self, result):
        _obj, _reason = self.tag_normer(result.object, result.reason)
        pass

    def __call__(self, results:List[ExtractResult]):
        _results = []
        for result in results:
            holder, emotion, fobject, reason, sent_score = \
                result.holder,result.emotion, result.object, result.reason, result.sent_score 
            if self.filte(holder, emotion, fobject, reason):
                logger.info(f"filted: holder:[{holder.text}], emotion:[{emotion.text}], "\
                            f"object:[{fobject.text}], reason:[{reason.text}]")
                continue
            result.sent_score = self.modify(fobject, reason, sent_score)
            _results.append(result)

        # TODO 标签归一化
        # for result in results:
        #     self.normtag(result)

        return _results