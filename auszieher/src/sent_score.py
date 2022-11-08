#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2022 Fisher. All rights reserved.
#   
#   文件名称：sent_score.py
#   创 建 者：YuLianghua
#   创建日期：2022年04月21日
#   描    述：计算情感极性值及强度值
#================================================================

import random
from typing import List,Dict

# import stanza
import pickle
import onnxruntime
import numpy as np
from textblob import TextBlob
from scipy.special import softmax
from stanza.models.common.vocab import PAD_ID, UNK_ID

from ..src import dataset
from ..utils.logger import logger
from ..utils.utils import timeit
from .extractor import ExtractResult
from .phrase_parser import Groups

SUNK_ID = 250000

with open('./data/model/sentiment/vocab_map.pkl', 'rb') as handle:
    vocab_map = pickle.load(handle) 
with open('./data/model/sentiment/extra_vocab_map.pkl', 'rb') as handle:
    extra_vocab_map = pickle.load(handle) 

def preprocess(inputs, max_window=5, training=False, max_phrase_len=10):
    if max_window > max_phrase_len:
        max_phrase_len = max_window

    batch_indices = []
    extra_batch_indices = []
    for phrase in inputs:
        if training:
            begin_pad_width = random.randint(0, max_phrase_len - len(phrase))
        else:
            begin_pad_width = 0
        end_pad_width = max_phrase_len - begin_pad_width - len(phrase)

        sentence_indices = [PAD_ID] * begin_pad_width
        sentence_unknowns = []

        for word in phrase:
            if word in vocab_map:
                sentence_indices.append(vocab_map[word])
                continue
            new_word = word.replace("-", "")
            # google vectors have words which are all dashes
            if len(new_word) == 0:
                new_word = word
            if new_word in vocab_map:
                sentence_indices.append(vocab_map[new_word])
                continue

            if new_word[-1] == "'":
                new_word = new_word[:-1]
                if new_word in vocab_map:
                    sentence_indices.append(vocab_map[new_word])
                    continue

            sentence_unknowns.append(len(sentence_indices))
            sentence_indices.append(SUNK_ID)

        sentence_indices.extend([PAD_ID] * end_pad_width)
        batch_indices.append(sentence_indices)

        extra_sentence_indices = [PAD_ID] * begin_pad_width
        for word in phrase:
            if word in extra_vocab_map:
                if training and random.random() < 0.01:
                    extra_sentence_indices.append(UNK_ID)
                else:
                    extra_sentence_indices.append(extra_vocab_map[word])
            else:
                extra_sentence_indices.append(UNK_ID)
        extra_sentence_indices.extend([PAD_ID] * end_pad_width)
        extra_batch_indices.append(extra_sentence_indices)

    batch_indices = np.array(batch_indices)
    extra_batch_indices = np.array(extra_batch_indices)

    return batch_indices, extra_batch_indices

class SentScore(object):
    def __init__(self, nlp, anchor_type_sent={}):
        # self.nlp = nlp
        # self.stanza_nlp = stanza.Pipeline(lang='en', processors='tokenize,sentiment')
        self.textblob = TextBlob
        self.anchor_type_sent = anchor_type_sent
        export_onnx_file = "./data/model/sentiment/stanza_sstplus.onnx"
        self.stanza_sentiment_onnx_session = onnxruntime.InferenceSession(export_onnx_file)

    def _stanza_model_score(self, text):
        doc = self.stanza_nlp(text)
        mlabel = -1
        mscore = 0
        for sentence in doc.sentences:
            sent_label, sent_value = sentence.sentiment 
            score = softmax(sent_value)[sent_label]
            if score>mscore:
                mscore = score
                mlabel = sent_label

        return mlabel, mscore

    def _gen_token_batch(self, doc, extract_results):
        comment_tokens_batch = []
        for extract_result in extract_results:
            all_match_tokenids = extract_result.all_match_tokenids
            comment_tokens = [t.text.lower() for t in \
                doc[ all_match_tokenids[0]:all_match_tokenids[-1]+1 ] if not t.is_punct]

            if len(comment_tokens)>10:
                if len(extract_result.reason.tokenids) >= 10:
                    comment_tokens = [doc[i].text.lower() for i in extract_result.reason.tokenids[:10]]
                else:
                    valid_tokenids = extract_result.emotion.tokenids + extract_result.reason.tokenids
                    if len(valid_tokenids)>= 10:
                        comment_tokens = [doc[i].text.lower() for i in valid_tokenids[:10]]
                    else:
                        comment_tokens = [doc[i].text.lower() for i in valid_tokenids]
            comment_tokens_batch.append(comment_tokens)

        return comment_tokens_batch

    def _gen_text_batch(self, extract_results):
        if not extract_results:
            return ''

        text_batch = [extract_result.clause.replace('.', ' ') for extract_result in extract_results]
        combine_string = '!!! '.join(text_batch)

        return combine_string

    @timeit
    def _stanza_model_onnx_score(self, tokens_batch):
        result = []
        if not tokens_batch:
            return result

        batch_indices, extra_batch_indices = preprocess(tokens_batch)
        outputs = self.stanza_sentiment_onnx_session.run(None,
            { "batch_indices":batch_indices, 
              "extra_batch_indices":extra_batch_indices})[0] 
        prob_dists = softmax(outputs, axis=1)
        labels= np.argmax(outputs, axis=1)
        for i, prob_dist in enumerate(prob_dists):
            label = labels[i]
            prob = prob_dist[label]
            result.append( (label, prob) )
        
        return result

    @timeit
    def _textblob_model_score(self, combine_string, batch_size):
        result = []
        if combine_string=='' or not combine_string :
            return result

        textblob_result = self.textblob(combine_string)

        assert len(textblob_result.sentences)>=batch_size, \
          f"textblob result size:[{len(textblob_result.sentences)}] not equal batch_size:[{batch_size}]"

        for sentence in textblob_result.sentences[:batch_size]:
            polarity = sentence.sentiment.polarity
            sentiment_assessments = sentence.sentiment_assessments.assessments
            result.append((polarity, sentiment_assessments))
        
        return result

    @timeit
    def _score(self, doc, extract_results:List[ExtractResult], anchor_type:str, joint_anchor_sent):
        r''' 计算情感得分
        '''
        comment_tokens_batch = self._gen_token_batch(doc, extract_results)
        stanza_sent_batch_result = self._stanza_model_onnx_score(comment_tokens_batch)

        comment_text_batch = self._gen_text_batch(extract_results)
        textblob_sent_batch_result=self._textblob_model_score(comment_text_batch, len(extract_results))

        for inx, extract_result in enumerate(extract_results):
            anchor_sent = dataset.sent_score(extract_result.anchor_lemma) or \
                                dataset.sent_score(extract_result.anchor_text)
            if extract_result.anchor_id in joint_anchor_sent:
                anchor_sent = joint_anchor_sent[extract_result.anchor_id]

            extract_result.anchor_sent = anchor_sent
            comment_text = extract_result.clause
            ## old 
            # stanza_label, stanza_polarity = self._stanza_model_score(comment_text)
            ## new
            stanza_label, stanza_polarity = stanza_sent_batch_result[inx]

            ## old
            # comment_doc = self.nlp(comment_text, disable=['ner'])
            # # a float within the range [-1.0, 1.0]
            # spacy_polarity = comment_doc._.blob.polarity
            # # a list of polarity and subjectivity scores for the assessed tokens
            # # e.g.: [(['really', 'horrible'], -1.0, 1.0, None), (['worst', '!'], -1.0, 1.0, None)]
            # model_assessments = comment_doc._.blob.sentiment_assessments.assessments
            ## new
            spacy_polarity, model_assessments = textblob_sent_batch_result[inx]

            stanza_score= (stanza_label-1)*stanza_polarity 
            model_score = spacy_polarity if \
                            abs(spacy_polarity)>abs(stanza_score) else stanza_score
            extract_result.model_sent = round(model_score,3)

            logger.info(f"anchor_sent:{anchor_sent}, anchor_type:{anchor_type}, spacy_polarity:{spacy_polarity}, " 
                f"stanza_label:{stanza_label}, stanza_polarity:{round(stanza_polarity,3)}, model_score:{round(model_score,3)}")
            # anchor_sent:1.0, anchor_type:common, spacy_polarity:0.0, stanza_label:1, stanza_polarity:0.8349999785423279, model_score:0.0

            # 表示中性
            if abs(spacy_polarity)<0.1 and \
              (stanza_label==1 and stanza_polarity>=0.85) and \
                not extract_result.emotion.tokenids:  # badcase: i like the screen.
                logger.info(f"filted by neural: {comment_text}")
                continue

            # 如果模型置信度很高直接出结果；
            sent_score = 0.0
            if abs(model_score)>=0.9:
                sent_score = model_score
            else:
                _sent_score = 0.0
                if anchor_sent!=0:
                    neg_tokenids = extract_result.e_neg_tokenids if extract_result.e_isneg \
                        else extract_result.r_neg_tokenids
                    _sent_score = (-1) ** len(neg_tokenids) * anchor_sent
                elif anchor_type in self.anchor_type_sent:
                    extract_result.anchor_type_sent = self.anchor_type_sent[anchor_type]
                    _sent_score = self.anchor_type_sent[anchor_type]
                ass_score = 0.0
                for assessment in model_assessments:
                    ass_score += assessment[1]
                sent_score = 0.7 * _sent_score + 0.1 * ass_score/(len(model_assessments) + 1e-3) + \
                             0.2 * model_score 
            
            extract_result.sent_score = round(sent_score, 3)

    def _norm_intensity(self, intensity):
        if intensity <= 1.0:
            return 0.2
        elif 1.0 < intensity <= 2.0:
            return 0.4 
        elif 2.0 < intensity <= 3.0:
            return 0.6
        elif 3.0 < intensity <= 4.0:
            return 0.8
        else:
            return 1.0

    def _intensity(self, extract_results:List[ExtractResult], degree_head_dict:Dict):
        r''' 计算情感强度
        '''
        degree_adverbs_heads = set(degree_head_dict.keys())
        for extract_result in extract_results:
            # anchor intensity
            anchor_intensity = dataset.sent_intensity(extract_result.anchor_lemma) or \
                                    dataset.sent_intensity(extract_result.anchor_text)

            # degree intensity
            degree_intensity = 1.0
            all_match_tokenids = set(extract_result.all_match_tokenids)
            matched_degree_heads = all_match_tokenids.intersection(degree_adverbs_heads)
            for mdh in matched_degree_heads:
                head = mdh
                done_set = set()
                while (head in degree_head_dict) and (head not in done_set):
                    degree_intensity *= degree_head_dict[head]['intensity']
                    done_set.add(head)
                    head = degree_head_dict[head]['tail']

            sent_intensity = round(abs(anchor_intensity * degree_intensity), 3)
            sent_intensity = self._norm_intensity(sent_intensity)

            logger.info(f"anchor_intensity: {anchor_intensity}, degree_intensity: {round(degree_intensity, 3)} "
                        f"sent_intensity: {sent_intensity}")
            extract_result.sent_intensity = sent_intensity

    def __call__(self, doc, groups:Groups, extract_results: List[ExtractResult], joint_anchor_sent, anchor_type="common"):
        self._score(doc, extract_results, anchor_type, joint_anchor_sent)
        self._intensity(extract_results, groups.degree_head_dict)
