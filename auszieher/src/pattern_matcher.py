#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2022 Fisher. All rights reserved.
#   
#   文件名称：pattern_matcher.py
#   创 建 者：YuLianghua
#   创建日期：2022年04月21日
#   描    述：规则匹配器
#
#================================================================

from collections import defaultdict

from spacy.tokens import Doc
from spacy.matcher import DependencyMatcher

from ..utils.logger import logger
from ..utils.utils import timeit

prep_pobj = [
{
    "RIGHT_ID": "anchor_founded",
    "RIGHT_ATTRS": {"DEP":"prep", "POS":"ADP"}
},
{
    "LEFT_ID": "anchor_founded",
    "REL_OP": ">",
    "RIGHT_ID": "1",
    "RIGHT_ATTRS": {
        "DEP": "pobj", 
        "POS":{"IN":["NOUN","PROPN"]}
    },
},
{
    "LEFT_ID": "anchor_founded",
    "REL_OP": "<",
    "RIGHT_ID": "unit_root",
    "RIGHT_ATTRS": {
        "POS": {"IN": ["VERB", "NOUN", "AUX", "ADJ"]}
    }
}
]

prep_prep_pobj = [
{
    "RIGHT_ID": "anchor_founded",
    "RIGHT_ATTRS": {"DEP":"prep", "POS":"ADP"}
},
{
    "LEFT_ID": "anchor_founded",
    "REL_OP": ">",
    "RIGHT_ID": "1",
    "RIGHT_ATTRS": { "DEP": "prep", "POS": "ADP"}
},
{
    "LEFT_ID": "1",
    "REL_OP": ">",
    "RIGHT_ID": "2",
    "RIGHT_ATTRS": {
        "DEP": "pobj", 
        "POS":{"IN":["NOUN","PROPN"]}
    },
},
{
    "LEFT_ID": "anchor_founded",
    "REL_OP": "<",
    "RIGHT_ID": "unit_root",
    "RIGHT_ATTRS": {
        "POS": {"IN": ["VERB", "NOUN", "AUX", "ADJ"]}
    }
}
]

better_than_obj = [
{
    "RIGHT_ID": "unit_root",
    "RIGHT_ATTRS": { "ORTH":{"IN": ["better", "worst", "worse", "good", "well", "bad"]} }
},
{
    "LEFT_ID": "unit_root",
    "REL_OP": ">",
    "RIGHT_ID": "1",
    "RIGHT_ATTRS": { "LEMMA": "than", "DEP": "prep" }
},
{
    "LEFT_ID": "1",
    "REL_OP": ">",
    "RIGHT_ID": "2",
    "RIGHT_ATTRS": { "DEP": "pobj" },
}
]

default_patterns = {
    "prep_pobj":[prep_pobj],
    "prep_prep_pobj":[prep_prep_pobj],
    "better_than_obj":[better_than_obj]
}

class PatternMatcher(object):
    def __init__(self, pattern_map={}, vocab=None):
        self.default_pattern_key2name = dict()
        self.matcher = DependencyMatcher(vocab)
        for pattern_name, pattern in default_patterns.items():
            pattern_key = self.pattern_key(pattern_name)
            self.default_pattern_key2name[pattern_key] = pattern_name
            self.matcher.add(pattern_name, pattern)

        for pattern_id, pattern_list in pattern_map.items():
            assert isinstance(pattern_id, int)
            assert isinstance(pattern_list, list)

            self.matcher.add(pattern_id, pattern_list)

    def pattern_key(self, pattern_name):
        # 如果pattern_name 为 string 类型时，获取pattern_name对应的hashid
        return self.matcher._normalize_key(pattern_name)

    @timeit
    def __call__(self, doc):
        assert isinstance(doc, Doc)
        _matches =  self.matcher(doc)
        matches = []
        default_match_group = defaultdict(dict)
        for match_id, token_ids in _matches:
            # 避免匹配环问题
            if len(set(token_ids))!=len(token_ids):
                logger.info(f"[FILTER]: pattern_matcher [{match_id}]: {token_ids}")
                continue

            if match_id in self.default_pattern_key2name:
                name = self.default_pattern_key2name[match_id]
                unit_key = None
                unit_tokens = set()
                for i, token_id in enumerate(token_ids):
                    pattern = default_patterns[name][0]
                    right_id= pattern[i]["RIGHT_ID"]
                    if right_id == 'unit_root':
                        unit_key = token_id
                    else:
                        unit_tokens.add(token_id)
                if (unit_key is not None) and abs(min(unit_tokens)-unit_key)<=3:
                    default_match_group[name][unit_key] = unit_tokens
            else:
                matches.append((match_id, token_ids))

        return matches, default_match_group