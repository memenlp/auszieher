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

class TrieNode:
    """建立词典的Trie树节点"""

    def __init__(self, isword):
        self.isword = isword
        self.usertag = set()
        self.norm_v  = ''
        self.children = {}


class TrieTree:
    """预处理器，在用户词典中的词强制分割"""

    def __init__(self, dict_file=None):
        """初始化建立Trie树"""
        if dict_file is None:
            dict_file = []
        self.dict_data = dict_file
        if isinstance(dict_file, str):
            with open(dict_file, encoding="utf-8") as f:
                lines = f.readlines()
            self.trie = TrieNode(False)
            for line in lines:
                fields = line.strip().split('\t')
                word = fields[0].strip()
                usertag = fields[1].strip() if len(fields) > 1 else ''
                norm_v  = fields[2].strip() if len(fields) > 2 else word
                self.insert(word, usertag, norm_v)
        else:
            self.trie = TrieNode(False)
            for w_t in dict_file:
                if isinstance(w_t, str):
                    w = w_t.strip()
                    t = ''
                else:
                    assert isinstance(w_t, tuple)
                    assert len(w_t)==2
                    w, t, nv = map(lambda x:x.strip(), w_t)
                self.insert(w, t, nv)

    def insert(self, word, usertag=None, norm_v=None):
        """Trie树中插入单词"""
        l = len(word)
        now = self.trie
        for i in range(l):
            c = word[i]
            if not c in now.children:
                now.children[c] = TrieNode(False)
            now = now.children[c]
        now.isword = True
        now.usertag.add(usertag)
        now.norm_v  = norm_v

    def solve(self, txt):
        """对文本进行预处理"""
        outlst = []
        iswlst = []
        taglst = []
        norm_vlst = []
        l = len(txt)
        last = 0
        i = 0
        while i < l:
            now = self.trie
            j = i
            found = False
            usertag = set()
            norm_v  = ''
            last_word_idx = -1 # 表示从当前位置i往后匹配，最长匹配词词尾的idx
            while True:
                c = txt[j]
                if not c in now.children and last_word_idx != -1:
                    found = True
                    break
                if not c in now.children and last_word_idx == -1:
                    break
                now = now.children[c]
                if now.isword:
                    last_word_idx = j
                    usertag = now.usertag
                    norm_v  = now.norm_v
                j += 1
                if j == l and last_word_idx == -1:
                    break
                if j == l and last_word_idx != -1 :
                    j = last_word_idx + 1
                    found = True
                    break
            if found:
                if last != i:
                    outlst.append(txt[last:i])
                    iswlst.append(False)
                    taglst.append('')
                    norm_vlst.append('')
                outlst.append(txt[i:j])
                iswlst.append(True)
                taglst.append(usertag)
                norm_vlst.append(norm_v)
                last = j
                i = j
            else:
                i += 1
        if last < l:
            outlst.append(txt[last:l])
            iswlst.append(False)
            taglst.append('')
            norm_vlst.append('')
        return outlst, iswlst, taglst, norm_vlst

