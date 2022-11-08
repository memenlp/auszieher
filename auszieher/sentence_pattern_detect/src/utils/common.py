# -*- coding: utf-8 -*-
"""
Author: weisi.wu
Date: 2022/5/26
Note: The common tools (check log dir, compute cost time)
Version: v1.0
"""

import os
import time
import re


def split_sentence(document, limit=510):
    """
    Args:
        document:
        flag: Type:str, "all" 中英文标点分句，"zh" 中文标点分句，"en" 英文标点分句
        limit: 默认单句最大长度为510个字符
    Returns: Type:list
    """
    sent_list = []
    try:
        document = re.sub('(?P<quotation_mark>([.?!](?![”’"\'])))', 
                           r'\g<quotation_mark>\n', document)  # 英文单字符断句符
        document = re.sub('(?P<quotation_mark>([?!.]["\']))', 
                               r'\g<quotation_mark>\n', document)  # 特殊引号

        sent_list_ori = document.splitlines()
        for sent in sent_list_ori:
            sent = sent.strip()
            if not sent:
                continue
            else:
                while len(sent) > limit:
                    temp = sent[0:limit]
                    sent_list.append(temp)
                    sent = sent[limit:]
                sent_list.append(sent)
    except:
        sent_list.clear()
        sent_list.append(document)

    merge_list = []
    sent_list_length = len(sent_list)
    i = 0
    while i<sent_list_length:
        if len(sent_list[i]) <= 4 and i<(sent_list_length-1):
            merge_list.append(sent_list[i]+sent_list[i+1])
            i += 2
        else:
            merge_list.append(sent_list[i])
            i += 1

    return merge_list

def check_dir(path):
    """
    check 文件目录
    """
    if os.path.exists(path):
        return

    old_musk = os.umask(0o022)
    os.makedirs(path)
    os.umask(old_musk)


def diff_time(start, end, point=2, unit="ms", sep=""):
    """
    diff 时间
    """
    diff = end - start
    diff = diff * 1000 if unit == "ms" else diff

    return f"{round(diff, point)}{sep}{unit}"


def cost_time(func):
    def wrapper(*args, **kwargs):
        trace_id: str = kwargs.get("trace_id")
        logger = kwargs.get("logger")

        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info(f"[{trace_id}] function '{func.__qualname__}' cost: {diff_time(start, end)}")

        return result

    return wrapper


def async_cost_time(func):
    async def wrapper(*args, **kwargs):
        trace_id = kwargs.get("trace_id")
        logger = kwargs.get("logger")

        start = time.time()
        result = await func(*args, **kwargs)
        end = time.time()
        logger.info(f"[{trace_id}] function '{func.__qualname__}' cost: {diff_time(start, end)}")

        return result

    return wrapper

