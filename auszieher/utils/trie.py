#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 Fisher. All rights reserved.
#   
#   文件名称：trie.py
#   创 建 者：YuLianghua
#   创建日期：2020年09月07日
#   描    述：
#
#================================================================

from typing import List

class TrieNode:
    """建立词典的Trie树节点"""

    def __init__(self, isword):
        self.isword = isword
        self.children = {}
        self.ilegal_pos = set()
        self.ilegal_dep = set()
        self.info = None


class TrieTree:
    """预处理器，在用户词典中的词强制分割"""

    def __init__(self): 
        """初始化建立Trie树"""
        self.trie = TrieNode(False)

    def insert(self, word, pos=set(), dep=set(), info=None):
        """Trie树中插入单词"""
        tokens = word.split()
        l = len(tokens)
        now = self.trie
        for i in range(l):
            c = tokens[i]
            if not c in now.children:
                now.children[c] = TrieNode(False)
            now = now.children[c]
        now.isword = True
        if pos:
            now.ilegal_pos=pos
        if dep:
            now.ilegal_dep=dep
        if info:
            now.info = info

    def solve(self, doc, rinfo=False):
        """对文本进行预处理"""
        outlst = []
        iswlst = []
        ilegal_postaglst = []
        ilegal_deptaglst = []
        infolst = []
        l = len(doc)
        last = 0
        i = 0
        while i < l:
            now = self.trie
            j = i
            found = False
            ilegal_pos = set()
            ilegal_dep = set()
            info = None
            last_word_idx = -1 # 表示从当前位置i往后匹配，最长匹配词词尾的idx
            while True:
                c = doc[j]
                if not c in now.children and last_word_idx != -1:
                    found = True
                    break
                if not c in now.children and last_word_idx == -1:
                    break
                now = now.children[c]
                if now.isword:
                    last_word_idx = j
                    ilegal_pos = now.ilegal_pos
                    ilegal_dep = now.ilegal_dep
                    info = now.info
                j += 1
                if j == l and last_word_idx == -1:
                    break
                if j == l and last_word_idx != -1 :
                    j = last_word_idx + 1
                    found = True
                    break
            if found:
                if last != i:
                    outlst.append(doc[last:i])
                    iswlst.append(False)
                    infolst.append(None)
                    ilegal_postaglst.append(set())
                    ilegal_deptaglst.append(set())
                outlst.append(doc[i:j])
                iswlst.append(True)
                infolst.append(info)
                ilegal_postaglst.append(ilegal_pos)
                ilegal_deptaglst.append(ilegal_dep)
                last = j
                i = j
            else:
                i += 1
        if last < l:
            outlst.append(doc[last:l])
            iswlst.append(False)
            infolst.append(None)
            ilegal_postaglst.append(set())
            ilegal_deptaglst.append(set())
        return (outlst, iswlst, ilegal_postaglst, ilegal_deptaglst) if not rinfo else \
            (outlst, iswlst, ilegal_postaglst, ilegal_deptaglst, infolst)


