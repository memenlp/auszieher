#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2022 Fisher. All rights reserved.
#   
#   文件名称：diff_check.py
#   创 建 者：YuLianghua
#   创建日期：2022年07月27日
#   描    述：
#
#================================================================

import sys
import json
import argparse
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict


def load_dataset():
    test_set = []
    with open('./diff/test_set', 'r') as rf:
        for line in rf:
            test_set.append(line.strip())
    return test_set

def load_bresult():
    r''' load baseline result
    '''
    bresults = []
    with open('./diff/result.base', 'r') as rf:
        for record in rf:
            bresults.append(json.loads(record.strip()))

    return bresults

def get_extractor():
    from auszieher import Executor
    from auszieher.utils.logger import logger
    logger.remove()
    fmt = '<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>'
    logger.add(sys.stderr, format=fmt, level="ERROR")

    config = {
        "neuralcoref_hosts":"ai.wgine-dev.com:32429",
        "sentence_pattern" : 
            {
                "interrogative":"./data/model/sentence_pattern/interrogative/xgb.model"
            }
        }
    executor = Executor(config)
    return executor

def generate_result(save=False, detail='true'):
    executor = get_extractor()
    test_set = load_dataset()
    results = []
    for query in tqdm(test_set, desc="processing..."):
        result_unit = {"query": query, "result":[]}
        _results = executor.extract(query)
        for r in _results:
            result_unit['result'].append(r.to_dict())

        results.append(result_unit)

    if save:
        suffix = datetime.strftime(datetime.now(), "%m.%d_%H:%M:%S")
        if detail=='true':
            records = [json.dumps(record) for record in results]
            with open(f'./diff/result.{suffix}', 'w') as wf:
                wf.write('\n'.join(records))
        elif detail=='false':
            lines = []
            for record in results:
                query = record['query']
                lines.append(query)
                uniqstrs = [uniqstr(r) for r in record['result']]
                for ustr in uniqstrs:
                    lines.append(f'\t\t{ustr}')
                lines.append('**********' * 4)
            
            with open(f'./diff/result.{suffix}', 'w') as wf:
                wf.write('\n'.join(lines))

        else:
            raise ValueError('detail must in [true | false]')

    return results

def generate_dataset():
    lines = []
    with open('./data/rule/rule.en', 'r') as rf:
        for line in tqdm(rf, desc="processing..."):
            if line.strip() and line.strip()[0]=='#' and line.strip()[1]!='#':
                lines.append(line.strip()[1:].strip())

    print(f'generate [ {len(lines)} ] texts !!!')
    with open('./diff/test_set', 'w') as wf:
        wf.write('\n'.join(lines))


def uniqstr(result):
    match_pattern = result['match_pattern']
    holder = result['holder']['text']
    emotion= result['emotion']['text']
    fobject= result['object']['text']
    reason = result['reason']['text']
    uniqstr = f"{match_pattern}: {holder}|{emotion}|{fobject}|{reason}"
    return uniqstr

def diff():
    diff_results = []
    base_results = load_bresult()
    new_results  = generate_result()
    assert len(base_results) == len(new_results)

    for i in range(len(base_results)):
        brecord = base_results[i]
        nrecord = new_results[i]
        assert brecord['query'] == nrecord['query']
        diff_unit = {"query":brecord['query'], 'info':defaultdict(list)}
        bresults = brecord['result']
        nresults = nrecord['result']

        buniqstrs = sorted([uniqstr(r) for r in bresults])
        nuniqstrs = sorted([uniqstr(r) for r in nresults])

        # compare
        if buniqstrs==nuniqstrs:
            continue
        for buniqstr in buniqstrs:
            if buniqstr not in set(nuniqstrs):
                diff_unit['info']["base"].append(buniqstr)

        for nuniqstr in nuniqstrs:
            if nuniqstr not in set(buniqstrs):
                diff_unit['info']['new'].append(nuniqstr)

        diff_results.append(diff_unit)

    # compute diff rate
    diff_len = len(diff_results)
    diff_ratio= diff_len /(len(base_results)+1e-5)
    print(f"diff ratio: {round(diff_ratio, 3)}")

    # write diff info
    lines = []
    for diffunit in diff_results:
        query = diffunit['query']
        lines.append(query)
        if diffunit['info']['base']:
            lines.append('\tBASE:')
            for l in diffunit['info']['base']:
                lines.append(f"\t\t{l}")
        if diffunit['info']['new']:
            lines.append('\tNEW:')
            for l in diffunit['info']['new']:
                lines.append(f"\t\t{l}")
        lines.append('************' * 4)
    suffix = datetime.strftime(datetime.now(), "%m.%d_%H:%M:%S")
    with open(f'./diff/diffinfo.{suffix}', 'w') as wf:
        wf.write('\n'.join(lines))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", "-t", help="task name, [ diff | genbase ]", \
                        default="diff")
    parser.add_argument("--detail", "-d", help="get detail result, [true | false]", \
                        default="true")

    args = parser.parse_args()
    if args.task == "diff":
        print("start to process task: diff ...")
        diff()
    elif args.task == "genbase":
        print("start to process task: genbase ...")
        detail = args.detail
        generate_result(save=True, detail=detail)
    elif args.task == "gendata":
        print("start to process task: gendata ...")
        generate_dataset()