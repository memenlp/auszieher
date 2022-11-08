#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2022 Fisher. All rights reserved.
#   
#   文件名称：pattern_parser.py
#   创 建 者：YuLianghua
#   创建日期：2022年04月21日
#   描    述：规则解析器
#
#================================================================
## 匹配单元结构定义： <(id, dep, pos, lemma, type, special_unit), (id, dep, pos, lemma, type, special_unit)>,<(), ()>
### id: 单元索引    dep: 依存关系    pos: 词性特征     lemma: 词干表示     type: 元素类别(A:aspect, O:opinion, N:not sure)
### id: 单元索引    dep: 依存关系    pos: 词性特征     lemma: 词干表示     type: 元素类别(H:holder, E:emotion, O:object, R:reason, P:placeholder)

## 匹配操作逻辑：在一个匹配单元中，如果已知节点在前则使用 `>` 操作，反之使用 `<` 操作；
# <(1,,NOUN,thing,N), (0)><(2,,AUX,be,N), (1)><(2), (3,attr,,,O)><(3), (4,prep,,,O)><(4), (5,pobj|amod,,,O)>

## 规则检查:
#  1.节点ID必须从小到大, 0表示anchor节点(必须);
#  2.在匹配单元中`<(),()>`必须包含一个新节点ID和一个一定有的节点ID;

import re
import json
from typing import Dict, List
from copy import deepcopy

from ..utils.logger import logger
from ..utils.utils import condition_lamb
from ..src import dataset

class PatternObject(object):
    pattern_map = dict()
    pattern_unit_etype_dict = dict()
    pattern_trace = dict()
    pattern_special_unit = dict()
    pattern_context_cond = dict()

class PatternParser(object):
    UNIT_MODULE = re.compile('\((.+?)\)', flags=re.DOTALL)
    MATCH_UNIT_MODULE = re.compile("<(.+?)>", flags=re.DOTALL)
    UNIT_STRUCT = {
        "LEFT_ID" : '',
        "REL_OP"  : '',
        "RIGHT_ID": '',
        "RIGHT_ATTRS": dict()
    }
    EXISTED_UNITS = set([0])
    PATTERNS = [
        {
        "RIGHT_ID": 0,
        "RIGHT_ATTRS": {"_": {"anchor": True}},
        },
    ]

    def __init__(self, pattern_file_path):
        # load pattern rules
        logger.info("start to load pattern rules...")
        self.pattern_rules = [
            rule.strip() for rule in open(pattern_file_path, 'r').readlines() \
                if rule.strip() and rule[0]!='#'
        ]

    @staticmethod
    def anchor_adjunction(fields, patterns:List[Dict]):
        r""" 对anchor添加附加定义
        """
        if len(fields)==1:
            return

        dep, pos, lemma, utype = fields[1:] if len(fields)==5 else fields[1:-1]
        special_type = '' if len(fields)==5 else fields[-1]
        for item in patterns:
            if item["RIGHT_ID"]==0:
                # 表示自定义的 special type 规则
                if special_type=='#ANCHOR#':
                    if '_' in item['RIGHT_ATTRS']:
                        item['RIGHT_ATTRS'].pop('_')
                if dep:
                    dep_value = dict()
                    dep_list = [d.strip() for d in dep.split('|') if d]
                    in_list  = [d for d in dep_list if not d.startswith('!')]
                    not_in_list = [d[1:] for d in dep_list if d.startswith('!')]
                    if in_list:
                        dep_value["IN"] = in_list
                    if not_in_list:
                        dep_value["NOT_IN"] = not_in_list

                    if dep_value:
                        item["RIGHT_ATTRS"]["DEP"] = dep_value
                if pos:
                    pos_value = dict()
                    pos_list = [p.strip() for p in pos.split('|') if p]
                    in_list  = [p for p in pos_list if not p.startswith('!')]
                    not_in_list = [p[1:] for p in pos_list if p.startswith('!')]
                    if in_list:
                        pos_value["IN"] = in_list
                    if not_in_list:
                        pos_value["NOT_IN"] = not_in_list

                    if pos_value:
                        item["RIGHT_ATTRS"]["POS"] = pos_value
                if lemma:
                    lemma_value = dict()
                    lemma_list = [l.strip() for l in lemma.split('|') if l]
                    in_list = [l for l in lemma_list if not l.startswith('!')]
                    not_in_list = [l[1:] for l in lemma_list if l.startswith('!')]
                    if in_list:
                        lemma_value["IN"] = in_list
                    if not_in_list:
                        lemma_value["NOT_IN"] = not_in_list

                    if lemma_value:
                        item["RIGHT_ATTRS"]["LEMMA"] = lemma_value
                # emotion lemma set
                elif not lemma and utype=="E":
                    lemma_value = dict()
                    in_list = list(dataset.emotion_vocab)
                    lemma_value["IN"] = in_list
                    item["RIGHT_ATTRS"]["LEMMA"] = lemma_value

    def _call(self, pattern_id, pattern_str):
        # 保存已经存在的节点单元
        existed_units = deepcopy(self.EXISTED_UNITS)

        match_units = self.MATCH_UNIT_MODULE.findall(pattern_str)
        logger.debug(f"[ {pattern_id} ] match_units: {match_units}")
        patterns = deepcopy(self.PATTERNS)
        unit_type_dict = dict()
        pattern_trace = []
        pattern_special_unit = dict()
        pattern_context_cond = dict()
        for match_unit in match_units:
            units = self.UNIT_MODULE.findall(match_unit)
            assert len(units)==2, "pattern is illegal"
            left, right = units[0], units[1]
            left_fields = [_.strip() for _ in left.split(',')]
            right_fields= [_.strip() for _ in right.split(',')]
            left_unit_id = int(left_fields[0])
            right_unit_id= int(right_fields[0])

            if left_unit_id==0 or right_unit_id==0:
                fields = left_fields if left_unit_id==0 else right_fields 
                assert len(fields) in (1,5,6,8), \
                    f"fields length must [1, 5, 6, 8], {len(fields)} is illegal!!!"
                utype = ''
                if len(fields)==5:
                    dep, pos, lemma, utype = fields[1:]
                elif len(fields)==6:
                    dep, pos, lemma, utype = fields[1:-1]
                    if fields[-1]:
                        pattern_special_unit[0] = fields[-1]
                elif len(fields)==8:
                    dep, pos, lemma, utype = fields[1:-3]
                    context_cond = {}
                    left_context_cond, right_context_cond = fields[-2:]
                    context_cond['left'] = condition_lamb(left_context_cond)
                    context_cond['right']= condition_lamb(right_context_cond)
                    pattern_context_cond[0] = context_cond

                if utype:
                    unit_type_dict[0] = utype

                self.anchor_adjunction(fields, patterns)

            # 必须保证一个节点单元已经存在，且不能重新定义
            # 左节点单元已知 e.g: <(2), (3,attr,,O)>
            unit_struct = deepcopy(self.UNIT_STRUCT)
            if (left_unit_id in existed_units) and \
              right_unit_id not in existed_units:
                assert len(right_fields) in (5,6,8), \
                  f"define unit must include 5/6 fields[id, dep, pos, lemma, type, special] not {len(right_fields)}"
                info_fields = []
                if len(right_fields)==5:
                    info_fields = right_fields[1:]
                elif len(right_fields)==6: 
                    info_fields = right_fields[1:-1]
                    if right_fields[-1]:
                        pattern_special_unit[right_unit_id] = right_fields[-1]
                elif len(right_fields)==8:
                    context_cond = {}
                    left_context_cond, right_context_cond = right_fields[-2:]
                    context_cond['left'] = condition_lamb(left_context_cond)
                    context_cond['right']= condition_lamb(right_context_cond)
                    pattern_context_cond[right_unit_id] = context_cond
                    info_fields = right_fields[1:-3]

                assert len(info_fields) == 4, f"info_fields length must==4, not {len(info_fields)}, {len(right_fields)}"
                dep, pos, lemma, utype = info_fields
                unit_type_dict[right_unit_id] = utype
                unit_struct["LEFT_ID"] = left_unit_id
                unit_struct["REL_OP"]  = '>'
                unit_struct["RIGHT_ID"]= right_unit_id
                if dep:
                    dep_value = dict()
                    dep_list = [d.strip() for d in dep.split('|') if d]
                    in_list  = [d for d in dep_list if not d.startswith('!')]
                    not_in_list = [d[1:] for d in dep_list if d.startswith('!')]
                    if in_list:
                        dep_value["IN"] = in_list
                    if not_in_list:
                        dep_value["NOT_IN"] = not_in_list

                    if dep_value:
                        unit_struct["RIGHT_ATTRS"]["DEP"] = dep_value
                if pos:
                    pos_value = dict()
                    pos_list = [p.strip() for p in pos.split('|') if p]
                    in_list  = [p for p in pos_list if not p.startswith('!')]
                    not_in_list = [p[1:] for p in pos_list if p.startswith('!')]
                    if in_list:
                        pos_value["IN"] = in_list
                    if not_in_list:
                        pos_value["NOT_IN"] = not_in_list

                    if pos_value:
                        unit_struct["RIGHT_ATTRS"]["POS"] = pos_value
                if lemma:
                    lemma_value = dict()
                    lemma_list = [l.strip() for l in lemma.split('|') if l]
                    in_list = [l for l in lemma_list if not l.startswith('!')]
                    not_in_list = [l[1:] for l in lemma_list if l.startswith('!')]
                    if in_list:
                        lemma_value["IN"] = in_list
                    if not_in_list:
                        lemma_value["NOT_IN"] = not_in_list
                    
                    if lemma_value:
                        unit_struct["RIGHT_ATTRS"]["LEMMA"] = lemma_value

                elif not lemma and utype=="E":
                    lemma_value = dict()
                    in_list = list(dataset.emotion_vocab)
                    lemma_value["IN"] = in_list
                    unit_struct["RIGHT_ATTRS"]["LEMMA"] = lemma_value

                pattern_trace.append((left_unit_id, right_unit_id))
                patterns.append(unit_struct)
                existed_units.add(right_unit_id)

            # 右节点单元已知 e.g: <(1,,NOUN,thing,N), (0)>
            elif (right_unit_id in existed_units) and \
              left_unit_id not in existed_units:
                assert len(left_fields) in (5,6,8), \
                  f"define unit must include 5 fields[id, dep, pos, lemma, type] not {len(left_fields)}"
                info_fields = []
                if len(left_fields)==5:
                    info_fields = left_fields[1:]
                elif len(left_fields)==6:
                    info_fields = left_fields[1:-1]
                    if left_fields[-1]:
                        pattern_special_unit[left_unit_id] = left_fields[-1]
                elif len(left_fields)==8:
                    context_cond = {}
                    left_context_cond, right_context_cond = left_fields[-2:]
                    context_cond['left'] = condition_lamb(left_context_cond)
                    context_cond['right']= condition_lamb(right_context_cond)
                    pattern_context_cond[left_unit_id] = context_cond
                    info_fields = left_fields[1:-3]

                assert len(info_fields) == 4, f"info_fields length must==4, not {len(info_fields)}"
                dep, pos, lemma, utype = info_fields
                unit_type_dict[left_unit_id] = utype
                unit_struct["LEFT_ID"] = right_unit_id 
                unit_struct["REL_OP"]  = '<'
                unit_struct["RIGHT_ID"]= left_unit_id
                if dep:
                    dep_value = dict()
                    dep_list = [d.strip() for d in dep.split('|') if d]
                    in_list  = [d for d in dep_list if not d.startswith('!')]
                    not_in_list = [d[1:] for d in dep_list if d.startswith('!')]
                    if in_list:
                        dep_value["IN"] = in_list
                    if not_in_list:
                        dep_value["NOT_IN"] = not_in_list

                    if dep_value:
                        unit_struct["RIGHT_ATTRS"]["DEP"] = dep_value
                if pos:
                    pos_value = dict()
                    pos_list = [p.strip() for p in pos.split('|') if p]
                    in_list  = [p for p in pos_list if not p.startswith('!')]
                    not_in_list = [p[1:] for p in pos_list if p.startswith('!')]
                    if in_list:
                        pos_value["IN"] = in_list
                    if not_in_list:
                        pos_value["NOT_IN"] = not_in_list
                    
                    if pos_value:
                        unit_struct["RIGHT_ATTRS"]["POS"] = pos_value
                if lemma:
                    lemma_value = dict()
                    lemma_list = [l.strip() for l in lemma.split('|') if l]
                    in_list = [l for l in lemma_list if not l.startswith('!')]
                    not_in_list = [l[1:] for l in lemma_list if l.startswith('!')]
                    if in_list:
                        lemma_value["IN"] = in_list
                    if not_in_list:
                        lemma_value["NOT_IN"] = not_in_list

                    if lemma_value:
                        unit_struct["RIGHT_ATTRS"]["LEMMA"] = lemma_value
                elif not lemma and utype=="E":
                    lemma_value = dict()
                    in_list = list(dataset.emotion_vocab)
                    lemma_value["IN"] = in_list
                    unit_struct["RIGHT_ATTRS"]["LEMMA"] = lemma_value

                pattern_trace.append((right_unit_id, left_unit_id))
                patterns.append(unit_struct)
                existed_units.add(left_unit_id)
        logger.debug(f"[ {pattern_id} ] pattern_struct: {json.dumps(patterns, indent=2)}")

        return (patterns, unit_type_dict, pattern_trace, pattern_special_unit, pattern_context_cond)

    def __call__(self):
        pattern_object = PatternObject()
        for pattern_rule in self.pattern_rules:
            fields = pattern_rule.split('\t')
            assert len(fields)==2
            pattern_id, rule_str = int(fields[0]), fields[1]
            p_obj, unit_type_dict, pattern_trace, pattern_special_unit, pattern_context_cond = \
                                                self._call(pattern_id, rule_str)
            pattern_object.pattern_map[pattern_id] = [p_obj]
            pattern_object.pattern_unit_etype_dict[pattern_id] = unit_type_dict
            pattern_object.pattern_trace[pattern_id] = pattern_trace
            pattern_object.pattern_special_unit[pattern_id] = pattern_special_unit
            pattern_object.pattern_context_cond[pattern_id] = pattern_context_cond

        return pattern_object