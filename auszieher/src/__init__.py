# coding=utf-8
#================================================================
#   Copyright (C) 2022 Fisher. All rights reserved.
#   
#   文件名称：__init__.py
#   创 建 者：YuLianghua
#   创建日期：2022年04月21日
#   描    述：目标提取器
#
#================================================================

import json

class Element(object):
    def __init__(self):
        self.text = ''
        self.normtag = ''
        self.tokenids = []
        self.index = ''

    def to_dict(self, fields=['text', 'tokenids', 'index', 'normtag'], token_offset=0):
        unit_dict = {}
        for field in fields:
            if getattr(self, field) is not None:
                value = getattr(self, field)
                if isinstance(value, list):
                    value = [_+token_offset for _ in value]
                unit_dict[field] = value
        return unit_dict

    @property
    def token_span_length(self):
        return (max(self.tokenids)-min(self.tokenids)+1) if self.tokenids else 0
    
    @property
    def prob(self):
        return len(self.tokenids)/(self.token_span_length+1e-4)
    
class ExtractResult(object):
    def __init__(self, token_offset=0, char_offset=0):
        self.token_offset = token_offset
        self.char_offset = char_offset

        self.clause = ''
        self.clause_index = ""        
        self.clause_tokens= []
        self.clause_lemmas= []
        self.match_pattern   = -1
        self.all_match_tokenids = []
        self.holder = Element()
        self.emotion= Element()
        self.object = Element()
        self.reason = Element()
        self.holder_type = ''
        self.object_type = ''    # aspect 类型：PRON(指示代词) | TACL(指示名词修饰语)
        self.anchor_id = None
        self.anchor_text = ''
        self.anchor_lemma= ''
        self.e_isneg = False
        self.r_isneg = False
        self.e_neg_tokenids= []
        self.r_neg_tokenids= []
        self.model_sent = 0.0
        self.anchor_sent= 0.0
        self.anchor_type_sent= 0.0
        self.sent_score  = 0.0
        self.sent_intensity = 0

    def to_dict(self, fields=['holder', 'emotion', 'object', 'reason', \
                             'anchor_id', 'anchor_text', 'anchor_lemma', 'e_isneg', 'e_neg_tokenids', \
                             'anchor_type_sent', 'model_sent', 'anchor_sent', 'sent_score', 'match_pattern', \
                             'object_type', 'sent_intensity', 'all_match_tokenids', 'clause', 'clause_index',\
                             'r_isneg', 'r_neg_tokenids']):
        unit_dict = {}
        for field in fields:
            if getattr(self, field) is not None:
                value = getattr(self, field)
                if isinstance(value, list):
                    value = [_+self.token_offset for _ in value]
                elif isinstance(value, Element):
                    value = value.to_dict(token_offset=self.token_offset)

                unit_dict[field] = value
        return unit_dict

    @property
    def prob(self):
        prob = 0.0
        i = 0
        for ele in [self.holder, self.emotion, self.object, self.reason]:
            if ele.tokenids:
                prob += ele.prob
                i += 1
        return prob/i

    def similar(self, other):
        intersection = set(self.all_match_tokenids).intersection(set(other.all_match_tokenids))
        
        return len(intersection)/len(self.all_match_tokenids)

    def diff(self, other):
        return set(self.all_match_tokenids).symmetric_difference(set(other.all_match_tokenids))

    def __eq__(self, other):

        if self.prob == other.prob and self.all_match_tokenids==other.all_match_tokenids:
            return True
        elif (abs(self.prob-other.prob)<=0.15 and self.similar(other) >= 0.75):
            return True
        else:
            return False

    def __lt__(self, other):
        return self.prob < other.prob or \
            len(self.all_match_tokenids) < len(other.all_match_tokenids)

    def __gt__(self, other):
        return self.prob > other.prob and \
            len(self.all_match_tokenids) > len(other.all_match_tokenids)

    def __repr__(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

class DataSet(object):
    def __init__(self):
        self._pos_vocab = set([w.strip() for w in \
            open('./data/positive.txt', 'r').readlines() if w.strip() and w[0]!='#'])
        self._neg_vocab = set([w.strip() for w in \
            open('./data/negative.txt', 'r').readlines() if w.strip() and w[0]!='#'])
        self._emotion_vocab = set([w.strip() for w in \
            open('./data/emotion.txt', 'r').readlines() if w.strip() and w[0]!='#'])

        self._sent_dict = {w:1.0 for w in self._pos_vocab}
        self._sent_dict.update({w:-1.0 for w in self._neg_vocab})

        intensity_items = [l.strip().split('\t') for l in \
            open('./data/sent_intensity.txt', 'r').readlines() if l.strip() and l[0]!='#']
        self._sent_intensity_dict = {item[0]:float(item[1]) for item in intensity_items}


        degree_items = [l.strip().split('\t') for l in \
            open('./data/degree_adverbs.txt', 'r').readlines() if l.strip() and l[0]!='#']
        self._degree_intensity_dict= {item[0]:float(item[1]) for item in degree_items}

        oov_vocab = [l.strip() for l in \
            open('./data/oov_vocab.txt', 'r').readlines() if l.strip() and l[0]!='#']
        self._oov_vocab = oov_vocab

        phrase_vocab = [l.strip() for l in \
            open('./data/phrase_vocab.txt', 'r').readlines() if l.strip() and l[0]!='#']
        self._phrase_vocab = phrase_vocab

        anchor_badcase = [l.strip('\n') for l in \
            open('./data/anchor_badcase.txt', 'r').readlines() if l.strip() and l[0]!='#']
        self._anchor_badcase = anchor_badcase

        filted_case = set([l.strip(' \n\r').lower() for l in \
            open('./data/filter.txt', 'r').readlines() if l.strip() and l[0]!='#'])
        self._filted_case = filted_case

        modify_case= dict()
        for l in open('./data/modify.txt', 'r').readlines():
            if l.strip() and l[0]!='#':
                items = l.strip(' \n\r').rsplit('\t',1)
                modify_case[items[0].lower()] = float(items[-1])
        self._modify_case = modify_case

        joint_sentiment_words = [l.strip().split('\t') for l in \
            open('./data/joint_sentiment_words.txt', 'r').readlines() if l.strip() and l[0]!='#']
        self.joint_sentiment_words_with_score = \
            dict([(_[0].strip(), float(_[1].strip())) for _ in joint_sentiment_words])

    @property
    def pos_vocab(self):
        return self._pos_vocab

    @property
    def neg_vocab(self):
        return self._neg_vocab

    @property
    def emotion_vocab(self):
        return self._emotion_vocab

    @property
    def sent_dict(self):
        return self._sent_dict

    @property
    def sent_intensity_dict(self):
        return self._sent_intensity_dict

    @property
    def degree_intensity_dict(self):
        return self._degree_intensity_dict

    @property
    def phrase_vocab(self):
        return self._phrase_vocab

    @property
    def oov_vocab(self):
        return self._oov_vocab

    @property
    def anchor_badcase(self):
        return self._anchor_badcase

    @property
    def filted_case(self):
        return self._filted_case
    
    @property
    def modify_case(self):
        return self._modify_case

    @property
    def joint_sentiment_words(self):
        return list(self.joint_sentiment_words_with_score.keys())

    def sent_score(self, token):
        return self._sent_dict.get(token, 0.0)

    def sent_intensity(self, token):
        return self._sent_intensity_dict.get(token, 1.0)

    def degree_intensity(self, token):
        return self._degree_intensity_dict.get(token, 1.0)
    
    def joint_sentiment_word_score(self, joint_word):
        return self.joint_sentiment_words_with_score.get(joint_word, 0.0)

    def update(self):
        return self.__init__()

global dataset
dataset = DataSet()
