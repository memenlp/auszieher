#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2022 Fisher. All rights reserved.
#   
#   文件名称：__init__.py
#   创 建 者：YuLianghua
#   创建日期：2022年05月20日
#   描    述：
#
#================================================================

from .utils import split_sentence

from .src.extractor import Extractor
from .src.sent_score import SentScore
from .src.token_parser import TokenParser
from .src.phrase_parser import PhraseParser
from .src.pattern_parser import PatternParser
from .src.pattern_matcher import PatternMatcher
from .src.postprocess import PostProcessor
from .src.coref_parser import CorefParser
from .src.sentence_pattern import HypoSentence
from .src.sentence_pattern import InteSentence

from .utils.logger import logger


class Executor(object):
    def __new__(cls, *args, **kwargs):
        instance = super(Executor, cls).__new__(cls)
        instance.token_parser = TokenParser(model_name="en_core_web_md")
        return instance

    def __init__(self, config=None, anchor_type_sent={"not_verb":-1.0, "no_adv":-1.0}):
        self._register()
        self.phrase_parser = PhraseParser()
        self.pattern_object = PatternParser(pattern_file_path='./data/rule/rule.en')()
        self.pattern_matcher = PatternMatcher(pattern_map=self.pattern_object.pattern_map,
                                              vocab = self.token_parser.model.vocab)
        self.extractor = Extractor(self.pattern_object)
        self.sent_score= SentScore(self.token_parser.nlp, anchor_type_sent=anchor_type_sent)
        self.postprocessor= PostProcessor()
        self.coref_parser = CorefParser(config['neuralcoref_hosts'], timeout=3)
        self.hypo_detector = HypoSentence()
        self.inte_detector = InteSentence(config['sentence_pattern'])

    def _register(self):
        @self.token_parser.register("anchor")
        def no_adv(doc, extension):
            # 处理 no longer (no <- ADV)
            flag = False
            for token in doc:
                if (token.dep_ == "neg" or token.lemma_=='no' ) and \
                    token.head.pos_ == "ADV":
                    doc[token.head.i]._.set(extension, True)
                    flag = True
            return flag

        @self.token_parser.register("anchor")
        def not_verb(doc, extension):
            # 处理 not link (not <- VERB)
            flag = False
            for token in doc:
                if (token.dep_ == "neg" or token.lemma_=="not") and \
                    token.head.pos_ == "VERB":
                    doc[token.head.i]._.set(extension, True)
                    flag = True
            return flag

    def extract(self, text, coref=False):
        corefmap = self.coref_parser(text) if coref else {}
        logger.debug(f"coref map: {corefmap}")
        _texts = split_sentence(text)

        # 句型判断
        hypots =  self.hypo_detector(_texts)

        results = []
        token_offset, char_offset = 0, 0
        for inx, _text in enumerate(_texts):
            prefix_offset = 0
            for char in _text:
                if char!=' ':
                    break
                prefix_offset += 1

            if prefix_offset>0:
                char_offset  += prefix_offset

            _text = _text[prefix_offset:]

            if not _text:
                continue
            
            logger.info(f"parse clause: {_text} ")
            doc = self.token_parser(_text)

            if hypots[inx]:
                token_offset += len(doc)
                if len(doc)>0:
                    char_offset += doc[-1].idx+len(doc[-1].text)     # 避免末尾是空格情况
                logger.info(f"[FILTER]: filted by HYPOTHETICAL !!!")
                continue
            if self.inte_detector([doc])[0]:
                token_offset += len(doc)
                if len(doc)>0:
                    char_offset += doc[-1].idx+len(doc[-1].text)     # 避免末尾是空格情况
                logger.info(f"[FILTER]: filted by INTERROGATIVE !!!")
                continue

            logger.debug(f"anchor tokens: {[t.text for t in doc if t._.anchor]}")
            self.token_parser.update_extension(doc)
            doc_lemma, joint_anchor_sent, joint_anchor_flag, joint_tuples = \
                self.token_parser.joint_sentiment_anchor_setter(doc)
            self.token_parser.anchor_badcase_postprocess(doc, doc_lemma)
            anchor_tokens = [t.text for t in doc if t._.anchor] 
            logger.debug(f"processed anchor tokens: {anchor_tokens}")
            if not anchor_tokens:
                logger.info(f"NO anchor tokens!!!")
                token_offset += len(doc)
                if len(doc)>0:
                    char_offset  += doc[-1].idx+len(doc[-1].text)     # 避免末尾是空格情况
                continue

            groups = self.phrase_parser(doc)

            logger.debug(f"phrase_groups: {groups.phrase_groups}")
            logger.debug(f"cc_groups: {groups.cc_groups}")
            logger.debug(f"poss_groups: {groups.poss_group}")
            logger.debug(f"prt_groups: {groups.prt_group}")
            logger.debug(f"xcomp_groups: {groups.xcomp_group}")
            logger.debug(f"aux_groups: {groups.aux_group}")
            logger.debug(f"joint_anchor_sent: {joint_anchor_sent}")

            matches, groups.default_match_group = self.pattern_matcher(doc)
            logger.info(f"pattern matches: {matches}")
            logger.debug(f"default match group: {groups.default_match_group}")
            _results = self.extractor(doc, matches, joint_anchor_flag, joint_tuples, \
                    groups=groups, token_offset=token_offset, char_offset=char_offset)
            self.sent_score(doc, groups, _results, joint_anchor_sent, anchor_type=doc._.anchor_type)

            results += _results
            token_offset += len(doc)
            if len(doc)>0:
                char_offset  += doc[-1].idx+len(doc[-1].text)     # 避免末尾是空格情况

        results = self.postprocessor(results)

        if not coref or not corefmap:
            return results

        for result in results:
            token_offset = result.token_offset
            if result.holder_type=="PRON" and len(result.holder.tokenids)==1:
                token_id = result.holder.tokenids[0] + token_offset
                corefunit = corefmap[token_id]
                if corefunit.coref_main:
                    result.holder.text = corefunit.coref_main
            if result.object_type=="PRON" and len(result.object.tokenids)==1:
                token_id = result.object.tokenids[0] + token_offset
                corefunit = corefmap[token_id]
                if corefunit.coref_main:
                    result.object.text = corefunit.coref_main

        return results
        
