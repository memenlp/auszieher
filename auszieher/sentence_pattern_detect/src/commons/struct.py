#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2021 Fisher. All rights reserved.
#   
#   文件名称：struct.py
#   创 建 者：YuLianghua
#   创建日期：2021年07月05日
#   描    述：
#
#================================================================

from collections import defaultdict
import json


class Word(object):
    def __init__(self, id='', text='', lemma='', head='', upos='', 
                       deprel='', start_char='', end_char=''):
        self.id = id
        self.text = text
        self.lemma= lemma
        self.head = head
        self.upos = upos
        self.xpos = '_'
        self.deprel = deprel
        self.feats  = '_'
        self.start_char = start_char
        self.end_char = end_char

    def to_dict(self):
        return self.__dict__

    def __repr__(self) -> str:
        return json.dumps(self.__dict__, indent=2, ensure_ascii=False)

class Sentence(object):
    def __init__(self, text=''):
        self.words = []
        self.text = text
        self.offset = []

    def to_dict(self):
        ret = []
        for word in self.words:
            ret.append(word.to_dict())
        return ret

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

class Document(object):
    def __init__(self, text=''):
        self.sentences = []
        self.text = text

    def to_dict(self):
        return [sentence.to_dict() for sentence in self.sentences]

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


class ExtractedUnit(object):
    def __init__(self):
        self.span = ''
        self.comment= ''
        self.match_rule = ''
        self.match_inx = None
        self.extract_type = ''
        self.offset = []

    def to_dict(self, fields=['span', 'comment', 'match_rule', 'extract_type', 'match_inx', 'offset']):
        unit_dict = {}
        for field in fields:
            if getattr(self, field) is not None:
                unit_dict[field] = getattr(self, field)
        return unit_dict
