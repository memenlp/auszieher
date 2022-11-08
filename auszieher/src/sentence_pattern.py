#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2022 Fisher. All rights reserved.
#   
#   文件名称：sentence_pattern.py
#   创 建 者：YuLianghua
#   创建日期：2022年07月14日
#   描    述：判断句子类型：虚拟语句、疑问语句、其它
#
#================================================================

from typing import List
from abc import abstractmethod
import numpy as np
import xgboost as xgb
from spacy.tokens import Doc

from auszieher.sentence_pattern_detect.src.server import Server as HypoServer

from ..utils.logger import logger
from ..utils.utils import timeit


QUESTION_WORDS2ID = {
    w:i for i,w in \
       enumerate("what how who whom whose which when where why".split(' '))
    }

class Model(object):
    def __init__(self, model_path, threshold=0.88, do_preprocess=True):
        self.xgb_model = xgb.Booster({'nthread':4})
        self.xgb_model.load_model(model_path)
        self._threshold = threshold
        self.do_preprocess = do_preprocess

    @property
    def threshold(self):
        return self._threshold

    @abstractmethod
    def preprocess(self, doc):
        raise NotImplemented
    
    def predict(self, input):
        prob = self.xgb_model.predict(input)
        return prob

    def __call__(self, input):
        if self.do_preprocess:
            input = self.preprocess(input)
        return self.predict(input)

class Interrogative(Model):
    def __init__(self, config):
        model_path = config['interrogative']
        super(Interrogative, self).__init__(model_path)

    @timeit
    def preprocess(self, docs):
        r''' 预处理，抽取人工特征
        # 1. 末尾是否包含问号
        # 2. 前五是否包含疑问词 what how who whom whose which when where why
        # 3. 疑问词离句首相对位置
        # 4. 疑问词后面词的词性[名称，动词，助动词, 其他]
        # 5. 是否包含助动词
        # 6. 疑问词与助动词间隔(最大5)
        # 7. 助动词与句首位置偏移
        # 8. 是否包含aux_NOUN|PRON|PROPN 是否倒装
        '''
        data = []
        for doc in docs:
            is_question_mark = 0
            suf = len(doc)-1
            while suf>0:
                if doc[suf].lemma_ == ' ': 
                    suf -= 1
                    continue
                else:
                    if doc[suf].lemma_=='?':
                        is_question_mark = 1
                    break

            # 2,3
            is_question_word = 0
            dist = -1
            word_id = -1
            for i,w in enumerate(doc[:5]):
                if doc[i].lemma_ in QUESTION_WORDS2ID:
                    dist = i
                    is_question_word = 1
                    word_id = QUESTION_WORDS2ID[doc[i].lemma_]
                    break

            # 4
            qw_bw_pos = -1
            if (dist+1) < len(doc):
                if doc[dist+1].pos_ == 'NOUN':
                    qw_bw_pos = 0
                elif doc[dist+1].pos_ == 'AUX':
                    qw_bw_pos = 1
                elif doc[dist+1].pos_ == "VERB":
                    qw_bw_pos = 2
                else:
                    qw_bw_pos = 3

            is_aux = 0
            aux_dist = -1
            aux_qw_dist = -1
            for i,w in enumerate(doc[:15]):
                if w.pos_ == "AUX":
                    is_aux = 1
                    aux_dist = min(i, 10)
                    if dist!=-1 and i>dist:
                        aux_qw_dist = min((i-dist), 5)
                    break
                    
            has_reverse = 0
            reverse_dist = -1
            for i,w in enumerate(doc[:15]):
                if w.pos_ in {"NOUN","PRON","PROPN"} and \
                    doc[w.head.i].pos_ in {"AUX","VERB"}:
                    if i > w.head.i:
                        has_reverse = 1
                        reverse_dist = min(w.head.i, 10)
                        break

            data.append([is_question_mark, is_question_word, dist, word_id, \
                qw_bw_pos, is_aux, aux_dist, aux_qw_dist, has_reverse, reverse_dist])

            data = np.array(data)

            return xgb.DMatrix(data)

class HypoSentence(object):
    def __init__(self):
        self.hypothetical_model = HypoServer(logger=logger)

    @timeit
    def __call__(self, texts:List[str]):
        # hypothetical check
        hypots = []
        results = self.hypothetical_model.process(texts, lang='en')
        for i,(label, prob) in enumerate(results):
            if str(label)=='1': 
                logger.info(f"[{texts[i]}] is HYPOTHETICAL sentence pattern, prob: {round(prob,3)} !!!")
                hypots.append(True)
            else:
                hypots.append(False)

        return hypots

class InteSentence(object):
    def __init__(self, config):
        self.interrogative_model= Interrogative(config)

    @timeit
    def __call__(self, docs:List[Doc]):
        # interrogative check
        inters = []
        probs = self.interrogative_model(docs)
        for i,prob in enumerate(probs):
            end = len(docs[i])-1
            while end>0:
                if docs[i][end].lemma_ == ' ':
                    end -= 1
                    continue
                else:break
            last_char = docs[i][end].lemma_
            logger.info(f"[{docs[i].text}] interrogative prob: {prob}")
            if prob > self.interrogative_model.threshold:
                logger.info(f"[{docs[i].text}] is INTERROGATIVE sentence pattern, prob: {round(prob,3)} !!!")
                inters.append(True)
            elif prob>0.55 and last_char=="?":
                logger.info(f"[{docs[i].text}] is INTERROGATIVE sentence pattern, prob: {round(prob,3)} + '?' !!!")
                inters.append(True)
            else:
                inters.append(False)
        
        return inters
