#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2022 Fisher. All rights reserved.
#   
#   文件名称：utils.py
#   创 建 者：YuLianghua
#   创建日期：2022年04月25日
#   描    述：
#
#================================================================

import os
import re
import time
from copy import deepcopy
from typing import List
from collections import defaultdict

from spacy.tokens import Doc
from treelib import Tree

from tytextnorm import TextNorm


_punct = (
    r"… …… , : . ; ! ? ¿ ؟ ¡ ( ) [ ] { } < > _ # * & 。 ？ ！ ， 、 ； ： ～ · । ، ۔ ؛ ٪"
)
split_chars = lambda char: list(char.strip().split(" "))

PUNCT_SET = set(split_chars(_punct))
TIMEIT_ENV = True if 'TIMEIT' in os.environ else False

textnorm  = TextNorm(lemma=False)

class Term(object):
    aspect_term = ''
    opinion_term = ''
    normed_aspect_term = ''
    normed_opinion_term = ''

class AspectItem(object):
    clause = ""
    clause_index = ""
    aspect_polarity = ""
    terms = []
    prob = None

class DocTree(object):
    def __init__(self, doc:Doc):
        self.doc = doc
        self.token2tree = self._build_tree(doc)

    def _build_tree(self, doc:Doc):
        token2tree = dict()
        # 使用递归方式构建树
        def _build(tree, root, data):
            if root not in data:
                return
            for child in data[root]:
                headi = doc[child].head.i
                tree.create_node(doc[child].text, child, parent=root, data=headi)
                _build(tree, child, data)

        for sent in doc.sents:
            tree = Tree()
            doc_data = defaultdict(list)
            root = None
            tokenids = []
            for token in sent:
                tokenids.append(token.i)
                if token.dep_ == "ROOT":
                    root = token.i
                else:
                    doc_data[token.head.i].append(token.i)
            assert root is not None, "ROOT must be valid int!!!"
            tree.create_node(root, root)
            _build(tree, root, doc_data)
            for tokenid in tokenids:
                token2tree[tokenid] = tree

        return token2tree

    def subtree(self, node_id):
        assert node_id in self.token2tree, f"{node_id} not in tree!!!"
        tree = self.token2tree[node_id]
        return tree.subtree(node_id)

    def subtree_nodes(self, node_id, limited=True):
        subtree = self.subtree(node_id)

        # nodes = [node.identifier for node in subtree.all_nodes()]
        #OPTIMIZE 添加限制条件：比如子节点深度不得超过3层且者子节点的叶子结点数不得超过3个
        non_leaf2children = defaultdict(list)
        invalid_level = set()
        nodes = set()
        for node in subtree.all_nodes():
            node_id = node.identifier
            nodes.add(node_id)
            head = node.data
            if head==subtree.root or node_id==subtree.root:
                continue
            non_leaf2children[head].append(node_id)
            level = subtree.level(node_id)
            if level>3:
                invalid_level.add(node_id)

        pop_set = set()
        for non_leaf,children in non_leaf2children.items():
            if len(children)>3:
                pop_set.add(non_leaf)
                q = children
                while q:
                    r = q.pop()
                    pop_set.add(r)
                    q += non_leaf2children.get(r, [])

        q = list(invalid_level - pop_set)
        while q:
            r = q.pop()
            pop_set.add(r)
            q += non_leaf2children.get(r, [])

        nodes = list(nodes - pop_set)

        return sorted(nodes)

    def subtree_tokens(self, node_id):
        nodes = self.subtree_nodes(node_id)
        tokens = [self.doc[node_id] for node_id in nodes]
        return tokens

    def show(self):
        trees = list(set(self.token2tree.values()))
        for tree in trees:
            tree.show()

def split_sentence(document: str, flag: str = "all", limit: int = 10000) -> List[str]:
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

    if len(sent_list)<=1:
        return sent_list

    merge_list = [sent_list[0]]
    sent_list_length = len(sent_list)
    i = 1
    while i<sent_list_length:
        j = 0
        while j < len(sent_list[i]):
            if sent_list[i][j] in PUNCT_SET:
                merge_list[-1] += sent_list[i][j]
                j += 1
            else:
                break

        if j!=0:
            if not sent_list[i][j:]:
                i += 1
                continue
            sent_list[i] = sent_list[i][j:]

        if len(sent_list[i]) <= 4 and i<(sent_list_length-1):
            merge_list.append(sent_list[i]+sent_list[i+1])
            i += 2
        else:
            if sent_list[i]:
                merge_list.append(sent_list[i])
            i += 1

    return merge_list

def graph_root2leaf_paths(graph, root):
    history_set = set()
    results = []
    stack= [root]
    def dfs(graph,x):   
        # 叶子节点
        if x not in graph:
            r = deepcopy(stack)
            results.append(r)
            history_set.add(stack.pop(-1))
            return 
        
        for y in graph[x]: 
            if y not in history_set:
                stack.append(y)
                dfs(graph,y) 
        history_set.add(stack.pop(-1))
        return        
    dfs(graph, root)

    return results

def pattern_key(matcher, pattern_name):
    return matcher._normalize_key(pattern_name)

def text_norm(text):
    text = textnorm(text)['text']
    return text

def select(extract_r1, extract_r2):
    r'''当两个结果相差不大的情况下'''
    return extract_r1 if extract_r1.prob>extract_r2.prob else extract_r2

def get_subroot(deps):
    m = defaultdict(set)
    ss = []
    for s,t in deps:
        m[s].add(t)
        ss.append(s)
        
    subroots = []
    done = set()
    q = [_ for _ in ss]
    while q:
        s = q.pop()
        if s in done:
            continue
            
        if s in m:
            rs = m[s]
            q += rs
        else:
            subroots.append(s)
        done.add(s)

    return subroots

def timeit(method):
    def timed(*args, **kw):
        if TIMEIT_ENV:
            ts = time.time()
            result = method(*args, **kw)
            te = time.time()
            if 'log_time' in kw:
                if hasattr(method, '__qualname__'):
                    name = kw.get('log_name', method.__qualname__.upper())
                else:
                    name = kw.get('log_name', method.__name__.upper())
                kw['log_time'][name] = int((te - ts) * 1000)
            else:
                name = method.__qualname__ if hasattr(method, '__qualname__') else \
                    method.__name__
                print('%r  %2.2f ms' % \
                    (name, (te - ts) * 1000))
        else:
            result = method(*args, **kw)

        return result
    return timed

def condition_lamb(text):
    cond = {"not_in":set(), "in":set()}
    for token in text.strip().split('|'):
        token = token.strip()
        if not token:continue
        if token[0]=='!' and token[1:]:
            cond['not_in'].add(token[1:])
        else:
            cond['in'].add(token)

    return cond