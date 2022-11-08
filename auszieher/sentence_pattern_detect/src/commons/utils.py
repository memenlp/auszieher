#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2021 Fisher. All rights reserved.
#   
#   文件名称：utils.py
#   创 建 者：YuLianghua
#   创建日期：2021年07月03日
#   描    述：
#
#================================================================

import regex as re
from typing import List

PUNC_SET= {'!', '"', "'", '(', ')', ',', '-', '.', ':', ';', '<', '>', '?', '[', ']', '{', '}', \
           '·', '—', '‘', '’', '“', '”', '、', '。', '《', '》', '「', '」', '『', '』', '【', '】', \
           '〔', '〕', '﹃', '﹄', '！', '（', '）', '，', '：', '；', '？'}

def lang_detect(text):
    ''' 语种识别: 中文(zh)/英文(en)
    '''
    def is_contains_chinese(strs):
        for _char in strs:
            if '\u4e00' <= _char <= '\u9fa5':
                return True
        return False

    if is_contains_chinese(text):
        return 'zh'
    else:
        return 'en'

def split_sentence(document: str, flag: str = "all", limit: int = 510) -> List[str]:
    """
    Args:
        document:
        flag: Type:str, "all" 中英文标点分句，"zh" 中文标点分句，"en" 英文标点分句
        limit: 默认单句最大长度为510个字符
    Returns: Type:list
    """
    sent_list = []
    try:
        if flag == "zh":
            document = re.sub('(?P<quotation_mark>([。？！…](?![”’"\'])))', 
                               r'\g<quotation_mark>\n', document)  # 单字符断句符
            document = re.sub('(?P<quotation_mark>([。？！]|…{1,2})[”’"\'])', 
                               r'\g<quotation_mark>\n', document)  # 特殊引号
        elif flag == "en":
            document = re.sub('(?P<quotation_mark>([.?!](?![”’"\'])))', 
                               r'\g<quotation_mark>\n', document)  # 英文单字符断句符
            document = re.sub('(?P<quotation_mark>([?!.]["\']))', 
                               r'\g<quotation_mark>\n', document)  # 特殊引号
        else:
            document = re.sub('(?P<quotation_mark>([。？！….?!](?![”’"\'])))', 
                               r'\g<quotation_mark>\n', document)  # 单字符断句符
            document = re.sub('(?P<quotation_mark>(([。？！.!?]|…{1,2})[”’"\']))', 
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
