#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2022 Fisher. All rights reserved.
#   
#   文件名称：main.py
#   创 建 者：YuLianghua
#   创建日期：2022年05月24日
#   描    述：
#
#================================================================

import gc
import sys
import time
import json
from importlib import reload


from auszieher import Executor

import auszieher.src
import auszieher.src.pattern_parser 
import auszieher.src.token_parser 
import auszieher.src.pattern_matcher
import auszieher.src.extractor

from auszieher.utils.logger import logger

logger.remove()
fmt = '<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>'
logger.add(sys.stderr, format=fmt, level="INFO")

config = {
    "neuralcoref_hosts":"ai.wgine-dev.com:32429",
    "sentence_pattern" : 
        {
            "interrogative":"./data/model/sentence_pattern/interrogative/xgb.model"
        }
    }
executor = Executor(config)

# queries = [q.strip() for q in open('./diff/test_set', 'r').readlines() if q.strip()]
# for query in queries:

def reload_vocab():
    del executor.token_parser
    gc.collect()
    reload(auszieher.src)
    reload(auszieher.src.token_parser)
    executor.token_parser = auszieher.src.token_parser.TokenParser(model_name="en_core_web_md")

def reload_rule():
    del executor.pattern_matcher
    del executor.extractor
    del executor.pattern_object
    gc.collect()
    reload(auszieher.src.pattern_parser)
    reload(auszieher.src.pattern_matcher)
    reload(auszieher.src.extractor)
    executor.pattern_object = auszieher.src.pattern_parser.PatternParser(pattern_file_path='./data/rule/rule.en')()
    executor.pattern_matcher= auszieher.src.pattern_matcher.PatternMatcher(
                                    pattern_map=executor.pattern_object.pattern_map,
                                    vocab = executor.token_parser.model.vocab
                                )
    executor.extractor = auszieher.src.extractor.Extractor(executor.pattern_object)

while True:
    query = input("query:")
    if query.strip().startswith("reload"):
        if query.strip() == 'reload':
            reload_vocab()
            reload_rule()
        else:
            items = query.strip().split(":")
            if items[-1]=="vocab": reload_vocab()
            elif items[-1]=='rule': reload_rule()
            else:
                print("illegal reload part!!!")
        continue
    if query.strip() == 'exit':
        sys.exit()
    start = time.time()
    results = executor.extract(query, coref=False)
    for r in results:
        mesge = json.dumps(r.to_dict(), ensure_ascii=False, indent=4)
        logger.info(mesge)
    
    print(f"cost time: {(time.time()-start)*1000}")
