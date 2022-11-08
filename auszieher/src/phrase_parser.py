#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2022 Fisher. All rights reserved.
#   
#   文件名称：phrase_parser.py
#   创 建 者：YuLianghua
#   创建日期：2022年04月26日
#   描    述： 抽取 组合词组 如： compound, prt
#
#================================================================

from typing import List
from collections import defaultdict

from spacy.tokens import Doc

from ..src import dataset
from ..utils.trie import TrieTree
from ..utils.utils import timeit

class ConjUnit(object):
    def __init__(self, i, distance):
        self.i = i
        self.distance = distance
class ConjRootUnit(object):
    def __init__(self):
        self.conjs = []

class Groups(object):
    def __init__(self):
        self.phrase_groups = []           # 保存 compound/prt 等固定组合词
        self.cc_groups = dict()           # 保存cc (并列依存) 关系组
        self.anchor_set = set()           # 保存anchor(锚点)结合
        self.anchor_neg = dict()          # 保存anchor(锚点)相关的否定修饰词
        self.pron_map = dict()            # 指示代词集合
        self.tacl_set = set()             # 指示名称修饰语集合, e.g.: i love the `fact`<指示名词> that xxxxx
        self.degree_head_dict = dict()    # 程度副词对应的dep head集合
        self.conj_root_unit_map = dict()
        self.dep_set_map = defaultdict(lambda:defaultdict(list))
        self.dep_map = dict()
        self.aux_group = dict()           # e.g. can see xxx
        self.prt_group = dict()           # e.g. set up
        self.xcomp_group = dict()         # e.g. The app is also very user friendly and easy to navigate.
        self.poss_group = dict()          # e.g. your husband
        self.default_match_group = defaultdict(dict)
        self.defined_vocab_group = dict()

class PhraseParser(object):
    PHRASE_DEP = {"compound", "prt"}
    # 处理并列关系
    CC_DEP = "conj"
    ACL_DEP = "acl"
    PREP_DEP= 'prep'
    CC_POS = {"VERB", "NOUN", "ADJ", "AUX", "PROPN"}  #并列关系分组中必须是同为 VERB|NOUN|ADJ 等有意义的pos才处理;
    PRON_POS = "PRON"
    NOUN_POS = "NOUN"
    PRON_CLASS = {
        # 人称代词
        "PERSONAL":['I', 'you', 'she', 'he', 'we','it', 'they', 
                    'me', 'her', 'him', 'us', 'you', 'them', 'these', 'this'],
        # # 指示代词
        # "DEMONSTRATIVE": ['this', 'these', 'that', 'those'],
        # 疑问代词
        "INTERROGATIVE": ['who', 'whom', 'which', 'what'],
        # 关系代词
        "RELATIVE": ['who', 'whom', 'that', 'which', 'whoever', 'whomever', 'whichever'],

        # 不定代词
        "INDEFINITE": ["something","anything","everything","somebody","someone","anybody","anyone",
                       "nothing","nobody","everybody","everyone"]
    }

    def __init__(self):
        self.trie_tree = TrieTree()
        self.pron2class= defaultdict(set)
        for k,vs in self.PRON_CLASS.items():
            for v in vs:
                self.pron2class[v].add(k)
        self._init_trie()

    def _init_trie(self):
        for phrase in dataset.phrase_vocab:
            self.trie_tree.insert(phrase)
        for phrase in dataset.joint_sentiment_words:
            self.trie_tree.insert(phrase)

    def phrase_solve(self, doc):
        phrase_group = dict()
        doc_lemma = [t.lemma_ for t in doc]
        outlst, iswlst,_,_ = self.trie_tree.solve(doc_lemma)
        start = 0
        for i, _iswlst in enumerate(iswlst):
            if _iswlst:
                end = start + len(outlst[i])
                group = set([i for i in range(start, end)])
                for i in range(start, end):
                    phrase_group[i] = group
            else:
                start += len(outlst[i])

        return phrase_group

    @staticmethod
    def reorganization(groups: List[set]):
        if len(groups)<=0:
            return {}, groups

        new_groups = [groups[0]]
        for group in groups[1:]:
            flag = False
            for i in range(len(new_groups)):
                if new_groups[i]&group:
                    new_groups[i].update(group)
                    flag = True
                    break

            if not flag:
                new_groups.append(group)

        id_group_map = dict()
        for ngroup in new_groups:
            for _id in ngroup:
                id_group_map[_id] = ngroup

        return id_group_map, new_groups

    def conj_dep(self, doc, head):
        head = head
        distance = 0
        while doc[head].dep_=="conj":
            head = doc[head].head.i
            _dis = abs(doc[head].i - doc[head].head.i)
            if _dis>distance:
                distance = _dis
        return head, distance

    def get_cc_groups(self, doc):
        def reverse_dep(doc):
            rdep = defaultdict(set)
            for token in doc:
                dep = token.dep_
                head = token.head.i
                rdep[head].add(dep)
            return rdep

        rdep = reverse_dep(doc)
        cc_groups = []
        conj_root_unit_map = dict()
        for token in doc:
            if token.dep_ != self.CC_DEP or \
              (token.pos_ not in self.CC_POS) or \
                (token.pos_ != token.head.pos_):
                continue

            # badcase: The repair bill was painful, but it is good to know your snowboard still works.
            if ((token.pos_==token.head.pos_) and (token.pos_ in {"VERB", "AUX"})) \
              and ('nsubj' in rdep[token.i] and 'nsubj' in rdep[token.head.i]) :
                continue

            # badcase: It is good you got back safely and only had a few minor problems.
            dobj_in = ('dobj' in pos for pos in [rdep[token.i], rdep[token.head.i]])
            if any(dobj_in) and not all(dobj_in):
                continue

            # 1. 短距离
            conj_root_i, _distance = self.conj_dep(doc, token.head.i)
            distance = max(_distance, abs(token.i-token.head.i))
            # if abs(token.i-token.head.i)<=3:
            if distance<=3:
                cc_groups.append({token.i, token.head.i})
                # conj_root_i = self.conj_dep(doc, token.head.i)
                conj_root_unit = conj_root_unit_map.get(conj_root_i, ConjRootUnit())
                conj_unit = ConjUnit(token.i, abs(token.i-token.head.i))
                conj_root_unit.conjs.append(conj_unit)
                conj_root_unit_map[conj_root_i] = conj_root_unit

            # 2. 超长距离
            # badcase: query:The watch does not have the option to respond to messages or use for calling, but you can get notification alerts so you don't have to pull out your phone to read them or see who is calling you
            # elif abs(token.i-token.head.i)>=8:
            elif distance >= 7:
                continue

            # 3. 长距离要相同的词性且有相同依存关系子节点
            # elif token.pos_==token.head.pos_ and \
            elif rdep.get(token.i, set()).intersection(rdep.get(token.head.i, set())):
                cc_groups.append({token.i, token.head.i})
                conj_root_unit = conj_root_unit_map.get(conj_root_i, ConjRootUnit())
                conj_unit = ConjUnit(token.i, abs(token.i-conj_root_i))
                conj_root_unit.conjs.append(conj_unit)
                conj_root_unit_map[conj_root_i] = conj_root_unit

        return cc_groups, conj_root_unit_map

    @timeit
    def __call__(self, doc: Doc):
        assert isinstance(doc, Doc)

        groups = Groups()

        phrase_groups = []
        for inx, token in enumerate(doc):
            if token._.anchor:
                groups.anchor_set.add(token.i)

            # dep set map
            groups.dep_set_map[token.head.i][token.dep_].append(token.i)
            # dep map
            groups.dep_map[token.i] = token.dep_

            # anchor_neg
            if token.dep_ == 'neg':
                groups.anchor_neg[token.head.i] = token.i

            # aux
            if token.dep_ == 'aux' and \
              token.pos_ in {"PART", "AUX"} and \
                token.head.pos_ == "VERB" and abs(token.head.i-token.i)<5:
                groups.aux_group[token.head.i] = token.i
            
            # prt
            if token.dep_ == 'prt' and \
              token.pos_ == "ADP" and \
                token.head.pos_ == "VERB" and abs(token.head.i-token.i)<5:
                groups.prt_group[token.head.i] = token.i
            
            # xcomp
            if token.dep_ == 'xcomp' and \
              token.pos_ == "VERB" and  \
                token.head.pos_ in {"ADJ","ADV"} and abs(token.head.i-token.i)<5:
                groups.xcomp_group[token.head.i] = token.i

            # poss
            if token.dep_ == "poss" and token.pos_ == "PRON":
                groups.poss_group[token.head.i] = token.i

            # phrase_groups:
            ## compound
            if token.dep_ in self.PHRASE_DEP:
                phrase_groups.append({token.i, token.head.i})
            ## VERB-amod-NOUN: 【charging performance】 is also very good.
            elif token.pos_ in {"VERB","ADJ"} and token.head.pos_ == "NOUN" \
              and token.dep_=="amod" and token._.anchor==False and \
                token.i<token.head.i and (token.head.i-token.i)<3: 
                phrase_groups.append({token.i, token.head.i})
            ## DET(no)-det-NOUN: 【no money】 特殊限定词修饰关系
            elif token.pos_ == "DET" and token.head.pos_ == "NOUN" \
              and token.dep_ == "det" and token.lemma_ == 'no':
                phrase_groups.append({token.i, token.head.i})
            ## 99%
            elif token.pos_ == "NOUN" and token.is_punct and token.lemma_=='%' and \
              (inx>=1 and doc[inx-1].like_num):
                phrase_groups.append({token.i, doc[inx-1].i})

            # cc
            # acl 保存指示名称修饰语集合
            elif token.dep_ == self.ACL_DEP and \
              token.head.pos_==self.NOUN_POS and \
                token.head.dep_ == "dobj":
                groups.tacl_set.add(token.head.i)

            # PRON 保存指示代词集合
            if token.pos_ == self.PRON_POS:
                groups.pron_map[token.i] = self.pron2class.get(token.lemma_, "OTHERS")

            # degree 程度副词修饰
            if token.lemma_ in dataset.degree_intensity_dict or \
                token.text  in dataset.degree_intensity_dict:
                degree_adverbs_info = dict()
                degree_adverbs_info['tail'] = token.i
                degree_adverbs_info['intensity'] = dataset.degree_intensity(token.lemma_) or \
                    dataset.degree_intensity(token.text)
                groups.degree_head_dict[token.head.i] = degree_adverbs_info

        cc_groups, conj_root_unit_map = self.get_cc_groups(doc)
        groups.cc_groups = cc_groups
        groups.conj_root_unit_map = conj_root_unit_map
        groups.phrase_groups, _ = self.reorganization(phrase_groups)

        defined_vocab_group = self.phrase_solve(doc)
        groups.defined_vocab_group = defined_vocab_group

        return groups 


if __name__ == "__main__":
    groups = [{1,2}, {1,3}, {2,4}, {0,9}, {5,6}, {6,7}]
    import time
    start = time.time()
    id_group_map = PhraseParser.reorganization(groups)
    print("cost", time.time()-start)
    print(id_group_map)
