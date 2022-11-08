#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2022 Fisher. All rights reserved.
#   
#   文件名称：token_parser.py
#   创 建 者：YuLianghua
#   创建日期：2022年04月21日
#   描    述：定义 Token extension 特征值
#
#================================================================

import logging
from collections import defaultdict

import spacy
from spacy.tokens import Token, Doc

from ..utils.trie import TrieTree
from ..utils.utils import get_subroot
from ..utils.utils import timeit

class Register:
  """Module register"""

  def __init__(self, registry_name="token_parser_register"):
    self._dict = defaultdict(list)
    self._name = registry_name

  def __setitem__(self, key, value):
    if not callable(value):
      raise Exception("Value of a Registry must be a callable.")
    if key is None:
      key = value.__name__
    self._dict[key].append(value)

  def register(self, param):
    """Decorator to register a function or class."""

    def decorator(key, value):
      self[key] = value
      return value

    if callable(param):
      # @reg.register
      return decorator(None, param)
    # @reg.register('alias')
    return lambda x: decorator(param, x)

  def __getitem__(self, key):
    try:
      return self._dict[key]
    except Exception as e:
      logging.error(f"module {key} not found: {e}")
      raise e

  def __contains__(self, key):
    return key in self._dict

  def keys(self):
    """key"""
    return self._dict.keys()

class TokenParser(Register):
    def __init__(self, extension_map=dict(), model_name='en_core_web_sm'):
        super(TokenParser, self).__init__()

        Token.set_extension("anchor", default=False, force=True)
        Doc.set_extension("anchor_type", default="common", force=True)
        for extension, default in extension_map.items():
            Token.set_extension(extension, default=default)

        from ..src import dataset
        self.dataset = dataset

        self.nlp= spacy.load(model_name)
        self.anchor_badcase_trie = TrieTree()
        self.joint_sentiment_word_trie = TrieTree()
        self._init_trie()
        self._add_oov()
    
    def _init_trie(self):
        for case in self.dataset.anchor_badcase:
            words, poss, deps = case.split('\t')
            ilegal_poss = set(poss.strip().split('|')) if poss else set()
            ilegal_deps = set(deps.strip().split('|')) if deps else set()
            self.anchor_badcase_trie.insert(words, pos=ilegal_poss, dep=ilegal_deps)

        for joint_word,score in self.dataset.joint_sentiment_words_with_score.items():
            self.joint_sentiment_word_trie.insert(joint_word, info=score)

    def _add_oov(self):
        for ow in self.dataset.oov_vocab:
            self.nlp.tokenizer.add_special_case(ow, [{"ORTH": ow}])
    
    @property
    def model(self):
        return self.nlp

    def extension_ancher_getter(self, token):
        r'''
        定位锚点(anchor)
        '''
        anchor = False
        #badcase: Really like the design and the color I chose.
        if token.pos_ == "ADP" and token.dep_=="ROOT":
            anchor = True
        elif ((token.lemma_ in self.dataset.sent_dict) or (token.text in self.dataset.sent_dict)): 
            anchor = True

        return anchor

    def joint_sentiment_anchor_setter(self, doc):
        doc_lemma = [t.lemma_ for t in doc]
        outlst, iswlst, _, _, scorelst = \
          self.joint_sentiment_word_trie.solve(doc_lemma, rinfo=True)

        start = 0
        joint_anchor_sent = dict()
        joint_anchor_flag = False
        joint_tuples = []
        for inx, _iswlst in enumerate(iswlst):
            if not _iswlst:
                start += len(outlst[inx])
                continue
            end = start + len(outlst[inx])
            inxs = set([i for i in range(start, end)])
            joint_tuples.append(inxs)
            deps = []
            for i in inxs:
                head_i = doc[i].head.i
                if (head_i not in inxs) or (i==head_i):
                    continue
                deps.append((i, head_i))
            subroot = get_subroot(deps)
            if len(subroot) == 1:
                doc[subroot[0]]._.anchor = True
                joint_anchor_sent[subroot[0]] = scorelst[inx]
                joint_anchor_flag = True

            start = end

        return doc_lemma, joint_anchor_sent, joint_anchor_flag, joint_tuples
    
    def update_extension(self, doc, callback=None):
        r'''
        使用lambda(callback)策略对自定义的 extension 进行更新;
        '''
        for extension_name, callbacks in self._dict.items():
            for callback in callbacks:
                assert callable(callback), "extension update callback function must callable!"
                flag = callback(doc, extension_name)
                if flag:
                    doc._.anchor_type = callback.__name__

    def anchor_badcase_postprocess(self, doc, doc_lemma):
        r'''
        处理anchor badcase 问题:
        e.g. 
          xx as well (well 不应该为anchor)
        '''
        outlst, iswlst, ilegal_postaglst, ilegal_deptaglst = \
                            self.anchor_badcase_trie.solve(doc_lemma)
        start = 0
        _pos  = []
        for inx, _iswlst in enumerate(iswlst):
            if not _iswlst:
                start += len(outlst[inx])
                continue

            end = start + len(outlst[inx])
            if not ilegal_postaglst[inx] and not ilegal_deptaglst[inx]:
                for i in range(start, end):
                    _pos.append(i)
            elif ilegal_postaglst:
                for i in range(start, end):
                    if doc[i].pos_ in ilegal_postaglst[inx]:
                       _pos.append(i)
            elif ilegal_deptaglst:
                for i in range(start, end):
                    if doc[i].pos_ in ilegal_postaglst[inx]:
                       _pos.append(i)

            start = end
                
        for _p in _pos:
            doc[_p]._.anchor = False

    @timeit
    def __call__(self, text):
        doc = self.nlp(text, disable=['ner'])
        
        for token in doc:
            token._.anchor = self.extension_ancher_getter(token)
        
        return doc

if __name__ == "__main__":
    token_parser = TokenParser()
    doc = token_parser("it can not walk")
