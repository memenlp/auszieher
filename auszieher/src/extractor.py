#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2022 Fisher. All rights reserved.
#   
#   文件名称：extractor.py
#   创 建 者：YuLianghua
#   创建日期：2022年04月21日
#   描    述：目标提取器
#
#================================================================

from copy import deepcopy
from collections import defaultdict
from typing import Dict, Set, List, Tuple

from .phrase_parser import Groups
from ..src import dataset
from ..src import ExtractResult
from ..utils.utils import DocTree
from ..utils.utils import graph_root2leaf_paths
from ..utils.utils import select
from ..utils.utils import timeit
from ..utils.logger import logger

VALID_CONJ_ROOT_DEP = {
    "dobj", "advcl", "poss", "advmod", 
    "root", "nsubj", "acomp", "amod"
    }

class Extractor(object):
    def __init__(self, pattern_object):
        # 保存每个 pattern 位置匹配的抽取目标定义,用于后期反解析抽取目标值
        self.token_match_etype = dict()
        self.pattern_match_trace  = pattern_object.pattern_trace
        self.pattern_special_unit = pattern_object.pattern_special_unit
        self.pattern_token_length = defaultdict(lambda:0)
        self.pattern_unit_etype_dict_map = pattern_object.pattern_unit_etype_dict
        self.pattern_context_cond = pattern_object.pattern_context_cond
        for pattern_id, unit_etype_dict in pattern_object.pattern_unit_etype_dict.items():
            assert isinstance(pattern_id, int)
            assert isinstance(unit_etype_dict, dict)
            etype_indexs = defaultdict(set)
            for inx, etype in unit_etype_dict.items():
                self.pattern_token_length[pattern_id] += 1
                if etype=="H":           # holder
                    etype_indexs['H'].add(inx)
                elif etype=="E":         # emotion
                    etype_indexs['E'].add(inx)
                elif etype=="O":         # object
                    etype_indexs['O'].add(inx)
                elif etype=="R":         # reason
                    etype_indexs['R'].add(inx)
                elif etype=="P":         # placeholder
                    etype_indexs['P'].add(inx)
                else: continue

            self.token_match_etype[pattern_id] = etype_indexs

    @staticmethod
    def complete_phrase(tokenids: Set[int], id_group_map:Dict[int, set], existed_tokenids: Set[int]) -> Set[int]:
        r""" 自动补全处理: 对于 compoud, prep 这样的结果进行缺省值补充;
        """
        for _id, _group in id_group_map.items():
            group = deepcopy(_group)
            if _id in group:
                group.remove(_id)
            if _id in tokenids and \
              not group.intersection(existed_tokenids):
                tokenids.update(group)
                existed_tokenids.update(group)
        
        return tokenids

    @staticmethod
    def complete_default_pattern(tokenids, default_match_group, existed_tokenids: Set[int]):
        for _, unit_groups in default_match_group.items():
            for unit_root, _unit_group in unit_groups.items():
                unit_group = deepcopy(_unit_group)
                if unit_root in unit_group:
                    unit_group.remove(unit_root)
                if unit_root in tokenids and \
                  not unit_group.intersection(existed_tokenids) :
                    tokenids.update(unit_group)
                    existed_tokenids.update(unit_group)
                
        return tokenids

    @staticmethod
    def complete_fixed(tokenids, fixed_group, existed_tokenids: Set[int]):
        for _head, _tail in fixed_group.items():
            if _head in tokenids and \
              _tail not in existed_tokenids:
                tokenids.add(_tail)
                existed_tokenids.add(_tail)

        return tokenids

    @staticmethod
    def complete_aux(tokenids, aux_group, existed_tokenids=set()):
        return Extractor.complete_fixed(tokenids, aux_group, existed_tokenids)
    
    @staticmethod
    def complete_prt(tokenids, prt_group, existed_tokenids: Set[int]):
        return Extractor.complete_fixed(tokenids, prt_group, existed_tokenids)

    @staticmethod
    def complete_xcomp(tokenids, xcomp_group, existed_tokenids=set()):
        return Extractor.complete_fixed(tokenids, xcomp_group, existed_tokenids)

    @staticmethod
    def complete_poss(tokenids, poss_group, existed_tokenids=set()):
        return Extractor.complete_fixed(tokenids, poss_group, existed_tokenids)

    @staticmethod
    def postprocess(lists):
        r""" 后置处理: 对于作为子集的结果将其过滤,
        e.g.: A:{1, 2, 3, 4}, B:{1, 2, 3}
              过滤掉 B
        Args:
            lists: [(holder, emotion, object, reason, total, ...), ... ]
        Returns:
            lists: [(holder, emotion, object, reason, total, ...), ... ]
        """
        if len(lists)<=1:
            return (lists, [_ for _ in range(len(lists))])

        sorted_index = sorted(range(len(lists)), key=lambda x:len(lists[x][4]), reverse=True)
        results = [lists[sorted_index[0]]]
        indexes = [sorted_index[0]]
        for index in sorted_index[1:]:
            sl = lists[index]
            flag = False
            for _ in results:
                if len(sl[4])<=len(_[4]) and set(sl[4]).issubset(_[4]):
                    flag = True
                    break
            if not flag:
                results.append(sl)
                indexes.append(index)

        return (results, indexes)
    
    @staticmethod
    def postprocess2(_results, cc_groups):
        r'''
        进一步对结果进行后处理;
        '''
        if len(_results)<=1:
            return _results

        cc_groups_set = set().union(*cc_groups) 
        
        results = [_results[0]]
        for result in _results[1:]:
            flag = False
            for i,_ in enumerate(results):

                #IDEA: 这里要考虑一个问题：是否是因为并列关系导致的相似度过高问题
                if cc_groups_set and \
                  len(cc_groups_set.intersection(result.diff(_)))>=2:
                    break 

                if result==_:
                    logger.info(f"[FILTER]: {result.all_match_tokenids} by compare: same!!!")
                    flag = True
                    break
                if result.similar(_)>0.7:  
                    logger.info(f"[FILTER]: {result.all_match_tokenids} by compare: similar!!!")
                    s = select(result, _)
                    results[i] = s
                    flag = True
                    break
            if not flag:
                results.append(result)

        return results


    @staticmethod
    def neg_process(results:Set[int], anchor_neg:Dict[int,int]):
        r""" 处理否定问题
        如果anchor有否定词修饰则将其补充到结果中;
        """
        isneg = False
        intersection = results.intersection(anchor_neg.keys())
        if intersection:
            isneg = True
            for _idx in intersection:
                results.add(anchor_neg[_idx])
        return isneg, intersection

    def _pattern_combine(self, matches, anchor_set=set()):
        r""" 同一个pattern 匹配的结果进行合并
        """
        _matches = defaultdict(list)
        for pattern_id, tokenids in matches:
            if pattern_id not in _matches:
                # 这里要保持tokenids 元素原始顺序, 因此不能使用set
                _matches[pattern_id] = [tokenids]
            else:
                _tokenids_list = _matches[pattern_id]
                flag = False
                for i, _tokenids in enumerate(_tokenids_list):
                    #IDEA: 当命中同一个pattern 且有公共交集时进行结果合并;
                    ##note: 合并过程不仅要注意token的合并还有对应的etype合并;
                    ## 距离条件badcase: My wife loves the speed of the processor, huge storage
                    union = set(tokenids).union(set(_tokenids))
                    intersection = set(tokenids).intersection(set(_tokenids))
                    if intersection and \
                      abs(max(union-intersection)-min(union-intersection)<=3):
                        _tokenids_list[i].extend(tokenids)
                        flag = True
                if not flag:
                    _matches[pattern_id].append(tokenids)

        etype_ids_map_list = list()
        for patternid, tokenids_list in _matches.items():
            for i,tokenids in enumerate(tokenids_list):
                etype_ids_map = {
                    'pattern_id': patternid,
                    'anchor_id' : tokenids[0],
                    'etype_tokenids': { 'H': set(), 
                                        'E': set(),
                                        'O': set(), 
                                        'R': set(),
                                        'P': set() }
                    }
                assert len(tokenids) % self.pattern_token_length[patternid]==0
                for inx, tokenid in enumerate(tokenids):
                    _inx = inx % self.pattern_token_length[patternid]
                    # holder
                    if _inx in self.token_match_etype[patternid]['H']:
                        etype_ids_map['etype_tokenids']['H'].add(tokenid)
                    # emotion
                    elif _inx in self.token_match_etype[patternid]['E']:
                        etype_ids_map['etype_tokenids']['E'].add(tokenid)
                    # object
                    elif _inx in self.token_match_etype[patternid]['O']:
                        etype_ids_map['etype_tokenids']['O'].add(tokenid)
                    # reason
                    elif _inx in self.token_match_etype[patternid]['R']:
                        etype_ids_map['etype_tokenids']['R'].add(tokenid)
                    # elif tokenid in anchor_set:
                    #     etype_ids_map['etype_tokenids']['R'].add(tokenid)
                    else:
                        etype_ids_map['etype_tokenids']['P'].add(tokenid)
                etype_ids_map_list.append(etype_ids_map)

        return etype_ids_map_list

    def generate_replace_pair(self, conj_child_traces, conj_root_id, conj_unit, groups):
        r'''
        pattern_conj_child_traces: {1: [{ 5:{8,4}, 8:{9} }] }
        pattern_id: 1
        conj_root_id: 5
        groups:
        doc: doc
        '''
        replace_pair = []
        for conj_child_trace in conj_child_traces:
            if conj_root_id not in conj_child_trace:
                continue
            leaf_paths = graph_root2leaf_paths(conj_child_trace, conj_root_id)
            for leaf_path in leaf_paths:
                conj_node_id = conj_unit.i

                _result = []
                current_node = conj_node_id
                for p in leaf_path[1:]:
                    dep = groups.dep_map[p]
                    if dep in groups.dep_set_map[current_node]:
                        #OPTIMIZE 这里有可能不止一个,现在为了简单起见默认假设只有一个 
                        current_node = groups.dep_set_map[current_node][dep][0]
                        _result.append(current_node)
                    else:
                        break
                result = []
                if _result:
                    result = [conj_node_id] + _result
                    replace_pair.append((leaf_path[:len(_result)+1], result))

        return replace_pair

    def cc_expand_v2(self, idsets, groups:Groups, conj_child_traces):
        expand_results = [idsets]
        for _id in idsets:
            if _id not in groups.conj_root_unit_map:
                continue

            for conj_unit in groups.conj_root_unit_map[_id].conjs:
                _idset = deepcopy(idsets)
                if conj_unit.distance<=3 or \
                   groups.dep_map.get(_id, '') in VALID_CONJ_ROOT_DEP:
                    _idset.remove(_id)
                    _idset.add(conj_unit.i)
                    expand_results.append(_idset)

                else:
                    if not conj_child_traces:
                        continue
                    replace_pairs = self.generate_replace_pair(conj_child_traces, \
                                                            _id, conj_unit, groups)
                    for raw, rep in replace_pairs:
                        _idset = _idset - set(raw)
                        _idset.update(set(rep))
                        expand_results.append(_idset)

        return expand_results

    def cc_expand_v3(self, holder, emotion, fobject, reason, groups:Groups, conj_child_traces):
        def bgroup(ele_group, ele, name):
            for i in ele:
                ele_group[i] = {"name":name, "value":ele}
        def rever(ele_group):
            revers = defaultdict(set)
            for _, r in ele_group.items():
                revers[r['name']].update(r['value'])
            holder, emotion, fobject, reason = set(), set(), set(), set()
            for name, values in revers.items():
                if name == 'holder':
                    holder = values
                elif name== 'emotion':
                    emotion = values
                elif name== 'object':
                    fobject = values
                elif name=='reason':
                    reason = values
                else:
                    raise ValueError(f"name: {name} not valid!!!") 
            return (holder, emotion, fobject, reason)

        ele_group = dict()
        _holder, _emotion, _object, _reason = \
            deepcopy(holder), deepcopy(emotion), deepcopy(fobject), deepcopy(reason)
        bgroup(ele_group, _holder, "holder")
        bgroup(ele_group, _emotion,"emotion")
        bgroup(ele_group, _object, "object")
        bgroup(ele_group, _reason, "reason")
        _ids = set().union(*[holder, emotion, fobject, reason])

        records = [(holder, emotion, fobject, reason)]
        replace_map_pairs = dict()

        for _id in _ids:
            if _id not in groups.conj_root_unit_map:
                continue
            for conj_unit in groups.conj_root_unit_map[_id].conjs:
                _ele_group = deepcopy(ele_group)
                if conj_unit.distance<=3 or \
                   groups.dep_map.get(_id, '') in VALID_CONJ_ROOT_DEP:
                     ele_g = _ele_group[_id]
                     ele_g['value'].remove(_id)
                     ele_g['value'].add(conj_unit.i)
                     replace_map_pairs[_id] = conj_unit.i

                else:
                    if not conj_child_traces:
                        continue
                    replace_pairs = self.generate_replace_pair(conj_child_traces, \
                                                            _id, conj_unit, groups)
                    for raw, rep in replace_pairs:
                        raw, rep = list(raw), list(rep)
                        assert len(raw) == len(rep)
                        for inx, d in enumerate(raw):
                            ele_g = _ele_group.get(d, None)
                            if not ele_g: continue
                            if d in ele_g['value']:
                                ele_g['value'].remove(d)
                            ele_g['value'].add(rep[inx])
                            replace_map_pairs[d] = rep[inx]
                r = rever(_ele_group)
                records.append(r)
        
        return records, replace_map_pairs

    def _extract_v2(self, doc, etype_tokenids:Dict[str, List[int]], groups:Groups, \
                            conj_child_traces, special_unit):
        r'''
        '''
        holder_tokenids = etype_tokenids['H']
        emotion_tokenids= etype_tokenids['E']
        object_tokenids = etype_tokenids['O']
        reason_tokenids = etype_tokenids['R']
        placeholder_tokenids = etype_tokenids['P']
        
        # 处理special unit功能
        if special_unit:
            doc_tree = DocTree(doc)
            special_tokenid = list(special_unit.keys())[0]    # pattern中只能包含一个special unit
            if special_tokenid in set(reason_tokenids):
                if special_unit[special_tokenid] =="#TREEROOT#":
                    subtree_nodes = doc_tree.subtree_nodes(special_tokenid)
                    # 通过subtree获取的数据可能包含palceholder结果
                    _reason_tokenids = set(subtree_nodes)-set(placeholder_tokenids)
                    reason_tokenids.update(_reason_tokenids)

            elif special_tokenid in set(object_tokenids):
                if special_unit[special_tokenid] =="#TREEROOT#":
                    subtree_nodes = doc_tree.subtree_nodes(special_tokenid)
                    object_tokenids = set(subtree_nodes)-set(placeholder_tokenids)

        records, replace_pairs = self.cc_expand_v3(holder_tokenids, emotion_tokenids, \
            object_tokenids, reason_tokenids, groups, conj_child_traces)
        results = []
        for record in records:
            _result = self.__extract_v2(groups, record, placeholder_tokenids)
            results += _result
        
        # holder_expands = self.cc_expand_v2(holder_tokenids, groups, conj_child_traces)
        # emotion_expands= self.cc_expand_v2(emotion_tokenids, groups, conj_child_traces)
        # object_expands = self.cc_expand_v2(object_tokenids, groups, conj_child_traces)
        # reason_expands = self.cc_expand_v2(reason_tokenids, groups, conj_child_traces)
        # holder_expands, emotion_expands, object_expands, reason_expands = zip(*records)
        return results, replace_pairs

    def __extract_v2(self, groups, record, placeholder_tokenids):
        holder_expands, emotion_expands, object_expands, reason_expands = [[record[i]] for i in range(len(record))]
        # IDEA: complete过程中应该避免重复expand
        existed_tokenids = set().union(*holder_expands)
        existed_tokenids = existed_tokenids.union(*emotion_expands)
        existed_tokenids = existed_tokenids.union(*object_expands)
        existed_tokenids = existed_tokenids.union(*reason_expands)

        # 补全预设匹配
        reason_expands = [self.complete_default_pattern(set(r_tokenids), \
            groups.default_match_group, existed_tokenids) for r_tokenids in reason_expands]

        object_expands = [self.complete_default_pattern(set(o_tokenids), \
            groups.default_match_group, existed_tokenids) for o_tokenids in object_expands]

        # 补全处理(compound, 自定义词组等)
        holder_complete_tokenids = [self.complete_phrase(set(h_tokenids), groups.phrase_groups, \
                                        existed_tokenids) for h_tokenids in holder_expands]
        emotion_complete_tokenids= [self.complete_phrase(set(e_tokenids), groups.phrase_groups, \
                                        existed_tokenids) for e_tokenids in emotion_expands]
        object_complete_tokenids = [self.complete_phrase(set(o_tokenids), groups.phrase_groups, \
                                        existed_tokenids) for o_tokenids in object_expands]
        reason_complete_tokenids = [self.complete_phrase(set(r_tokenids), groups.phrase_groups, \
                                        existed_tokenids) for r_tokenids in reason_expands]

        # 补全自定义词典(defined vocab)
        holder_complete_tokenids = [self.complete_phrase(set(h_tokenids), groups.defined_vocab_group, \
                                        existed_tokenids) for h_tokenids in holder_complete_tokenids]
        # emotion_complete_tokenids= [self.complete_phrase(set(e_tokenids), groups.defined_vocab_group, \
        #                                 existed_tokenids) for e_tokenids in emotion_complete_tokenids]
        object_complete_tokenids = [self.complete_phrase(set(o_tokenids), groups.defined_vocab_group, \
                                        existed_tokenids) for o_tokenids in object_complete_tokenids]
        reason_complete_tokenids = [self.complete_phrase(set(r_tokenids), groups.defined_vocab_group, \
                                        existed_tokenids) for r_tokenids in reason_complete_tokenids]

        # 补全xcomp
        reason_complete_tokenids = [self.complete_xcomp(set(r_tokenids), groups.xcomp_group, \
                                        existed_tokenids) for r_tokenids in reason_complete_tokenids]
        
        # 补全aux
        reason_complete_tokenids = [self.complete_aux(set(r_tokenids), groups.aux_group, \
                                        existed_tokenids) for r_tokenids in reason_complete_tokenids]

        # 补全prt
        reason_complete_tokenids = [self.complete_prt(set(r_tokenids), groups.prt_group, \
                                        existed_tokenids) for r_tokenids in reason_complete_tokenids]

        # 补全poss
        holder_complete_tokenids = [self.complete_poss(set(h_tokenids), groups.poss_group, \
                                        existed_tokenids) for h_tokenids in holder_complete_tokenids]
        object_complete_tokenids = [self.complete_poss(set(o_tokenids), groups.poss_group, \
                                        existed_tokenids) for o_tokenids in object_complete_tokenids]
        reason_complete_tokenids = [self.complete_poss(set(r_tokenids), groups.poss_group, \
                                        existed_tokenids) for r_tokenids in reason_complete_tokenids]

        base_result = [ holder_complete_tokenids[0],
                        emotion_complete_tokenids[0],
                        object_complete_tokenids[0],
                        reason_complete_tokenids[0]]
        permutations = [ base_result ]
        for h in holder_complete_tokenids[1:]:
            expand_r = deepcopy(base_result)
            expand_r[0] = h
            permutations.append(expand_r)
        for e in emotion_complete_tokenids[1:]:
            expand_r = deepcopy(base_result)
            expand_r[1] = e
            permutations.append(expand_r)
        for o in object_complete_tokenids[1:]:
            expand_r = deepcopy(base_result)
            expand_r[2] = o
            permutations.append(expand_r)
        for r in reason_complete_tokenids[1:]:
            expand_r = deepcopy(base_result)
            expand_r[3] = r
            permutations.append(expand_r)

        _results = []
        for h, e, o, r in permutations:
            # 否定处理，只处理emotion, reason
            ## emotion
            e_isneg, e_neg_tokenids = self.neg_process(e, groups.neg_group)
            # e_isneg, e_neg_tokenids = self.neg_process(e, groups.anchor_neg)
            ## reason
            r_isneg, r_neg_tokenids = self.neg_process(r, groups.neg_group)
            # r_isneg, r_neg_tokenids = self.neg_process(r, groups.anchor_neg)
            comment_tokenids = set.union(h, e, o, r, set(placeholder_tokenids))
            h_sorted = sorted(h)
            e_sorted = sorted(e)
            o_sorted = sorted(o)
            r_sorted = sorted(r)
            c_sorted = sorted(comment_tokenids)
            _results.append(
                (h_sorted, e_sorted, o_sorted, r_sorted, c_sorted, placeholder_tokenids, 
                 e_isneg, e_neg_tokenids, r_isneg, r_neg_tokenids)
            )
        
        return _results
        
    def conj_match_trace(self, matches, conj_root_unit_map):
        r'''从matches中获取特定conj_root节点的子依存路径
        matches: [(5, [1,5,8,4])]
        conj_root_unit_map: {5: ConjRootUnit[<conjs:List>]}
        pattern_match_trace: {5:[(0,1), (1,2), (1,3)]}
        '''
        # 过滤掉无需处理的conj_root
        filted_conj_root = dict()
        for root_id, conj_root_unit in conj_root_unit_map.items():
            if any(conj_unit.distance>3 for conj_unit in conj_root_unit.conjs):
                filted_conj_root[root_id] = conj_root_unit
        
        pattern_conj_child_trace = defaultdict(list)
        for pattern_id, tokenids in matches:
            conj_root_id_sets = set(tokenids).intersection(filted_conj_root.keys())
            # 如果匹配结果中没有待处理的conj_root
            if len(conj_root_id_sets)<=0:
                continue

            for conj_root_id in conj_root_id_sets:
                conj_child_units = filted_conj_root[conj_root_id].conjs
                traces = defaultdict(set)
                for conj_child_unit in conj_child_units:
                    if conj_child_unit.distance<=3:continue
                    start_trace_id = -1
                    for i,tokenid in enumerate(tokenids):
                        if tokenid == conj_root_id:
                            start_trace_id = i
                            # 这里假设在一个匹配结果中这种长距离的conj只有一个;
                            break
                    if start_trace_id==-1:continue
                    for s,e in self.pattern_match_trace[pattern_id][start_trace_id:]:
                        s_id, e_id = tokenids[s], tokenids[e]
                        traces[s_id].add(e_id)
                pattern_conj_child_trace[pattern_id].append(traces)
                    
        return pattern_conj_child_trace

    def _special_unit_extract(self, matches):

        # 当命中special_unit 对应的位置同时被其他pattern匹配，而且该位置不为末尾节点时
        # 对应的TREEROOT匹配不生效;
        # for badcase: I love that I can see my oxygen levels, heartbeat , temperature , track my steps , workout , accept calls and notify me of emails , texts , calendar reminders , etc, etc ...
        unspecial_tokenids_set = set()
        for pattern_id, tokenids in matches:
            if self.pattern_special_unit.get(pattern_id,{}):
                continue
            match_traces = self.pattern_match_trace[pattern_id]
            sd = dict()
            for s, d in match_traces:
                detype = self.pattern_unit_etype_dict_map[pattern_id][d]
                if detype=='P':
                    continue
                sd[tokenids[s]] = tokenids[d]
            for t in tokenids:
                if t not in sd: continue
                unspecial_tokenids_set.add(t)

        special_unit = dict()
        for pattern_id, tokenids in matches:
            if not self.pattern_special_unit.get(pattern_id, {}):
                continue
            for idx, special_type in self.pattern_special_unit[pattern_id].items():
                tokenid = tokenids[idx]
                if tokenid in unspecial_tokenids_set:
                    info_msg = f"[FILTER]: pattern:[ {pattern_id} ] special token:[ {tokenid} ]!!!"
                    logger.info(info_msg)
                    continue
                special_unit[pattern_id] = {tokenid:special_type}

        return special_unit

    def internal_distance_check(self, _matches):
        r''' 检查元素内相邻位置距离
        '''
        matches = []
        for patternid, tokenids in _matches:
            match_traces = self.pattern_match_trace[patternid]
            valid_flag = True
            for src, dst in match_traces:
                src_etype = self.pattern_unit_etype_dict_map[patternid][src]
                dst_etype = self.pattern_unit_etype_dict_map[patternid][dst]
                if abs(tokenids[dst]-tokenids[src])>5 and src_etype==dst_etype:
                    valid_flag = False
                    logger.info(f"[FILTER]: patternid:[{patternid}], tokenids:[{tokenids}] "\
                                f"by internal distance check")
                    break
            if valid_flag:
                matches.append((patternid, tokenids))

        return matches

    def context_cond_check(self, doc, _matches):
        r''' 上下文条件检查;
        '''
        matches = []
        for patternid, tokenids in _matches:
            if patternid not in self.pattern_context_cond: continue
            valid_flag = True
            for inx, context_cond in self.pattern_context_cond[patternid].items():
                token_id = tokenids[inx]
                left_in = context_cond['left']['in']
                if left_in and doc[token_id-1].lemma_ not in left_in:
                    valid_flag = False
                    break
                left_not_in = context_cond['left']['not_in']
                if left_not_in and doc[token_id-1].lemma_ in left_not_in:
                    valid_flag = False
                    break
                right_in= context_cond['right']['in']
                if right_in and ((token_id+1)<len(doc)-1) and \
                  doc[token_id+1].lemma_ not in right_in:
                    valid_flag = False
                    break 
                right_not_in= context_cond['right']['not_in']
                if right_not_in and ((token_id+1)<len(doc)-1) and \
                  doc[token_id+1].lemma_ in right_not_in:
                    valid_flag = False
                    break

            if valid_flag:
                matches.append((patternid, tokenids))

        return matches

    def _remove_prefix_punct_space(self, tokenids, doc):
        r''' 删除tokenids 前缀的空格和标点符号(一般是因为TREEROOT方式匹配到的)
        '''
        start = 0
        for i,_id in enumerate(tokenids):
            if doc[_id].pos_ in ('PUNCT', 'SPACE'):
                continue
            else: 
                start = i
                break
        return tokenids[start:]
    
    def _span_matched_ratio(self, tokenids):
        return len(tokenids)/(max(tokenids)-min(tokenids) + 1)

    def _combine_str(self, tokenids, doc):
        if len(tokenids)<=0:
            return ''

        qstr = doc[tokenids[0]].lemma_
        for tid in tokenids[1:]:
            if doc[tid].idx == doc[tid-1].idx + len(doc[tid-1].text) and\
              (doc[tid].is_punct or doc[tid-1].is_punct):
                qstr += doc[tid].lemma_
            else:
                qstr += ' '+doc[tid].lemma_
                
        return qstr

    @timeit
    def __call__(self, doc, matches, joint_anchor_flag, joint_tuples, 
                 groups: Groups=None, token_offset=0, char_offset=0):

        # TODO 依据context condition 进行结果过滤;
        matches = self.context_cond_check(doc, matches)

        # IDEA: 为了减少spacy依存关系解析错误导致误召回：匹配结果中同一元素内的相邻匹配位置距离间隔不能太大：如不能超过5
        matches = self.internal_distance_check(matches)
            
        pattern_conj_child_traces = self.conj_match_trace(matches, groups.conj_root_unit_map)

        special_unit_inx = self._special_unit_extract(matches)
        etype_ids_map_list = self._pattern_combine(matches, groups.anchor_set)

        results, result_patternids, result_anchor_ids = [], [], []
        for etype_ids_map in etype_ids_map_list:
            pattern_id = etype_ids_map['pattern_id']
            special_unit = special_unit_inx.get(pattern_id, {})
            anchor_id  = etype_ids_map['anchor_id']
            etype_tokenids = etype_ids_map['etype_tokenids']
            conj_child_traces = pattern_conj_child_traces.get(pattern_id, [])
            _results, replace_pairs = self._extract_v2(doc, etype_tokenids, groups, conj_child_traces, special_unit)
            results += _results
            result_patternids += [pattern_id]*len(_results)
            anchor_ids = []
            for inx in range(len(_results)):
                _valid_ids = set(_results[inx][1]+_results[inx][3])
                if anchor_id not in _valid_ids and anchor_id in replace_pairs and \
                  replace_pairs[anchor_id] in _valid_ids:
                    anchor_ids.append(replace_pairs[anchor_id])
                else:
                    anchor_ids.append(anchor_id)
            result_anchor_ids += anchor_ids
            # result_anchor_ids += [anchor_id] *len(_results)
        
        results, sorted_index = self.postprocess(results)
        extract_results = []
        for i,result in enumerate(results):
            extract_result = ExtractResult(token_offset=token_offset,
                                           char_offset =char_offset)
            # aspect, opinion, comment, meanless, isneg, neg_tokenids = result
            holder, emotion, fobject, reason, comment, meanless, e_isneg, e_neg_tokenids, \
                r_isneg, r_neg_tokenids = result

            match_pattern = result_patternids[sorted_index[i]]

            # 如果是有联合情感词匹配的情况下，联合词应该在同一个元素内
            if joint_anchor_flag:
                _flag = False
                joint_token_total = set().union(*joint_tuples)
                if joint_token_total.intersection(set(comment)):
                    for joint_token_set in joint_tuples:
                        if not joint_token_set.intersection(set(comment)):
                            continue
                        if any(joint_token_set.issubset(_) for _ in [emotion, reason]):
                            _flag = True
                            break
                    if not _flag:
                        logger.info(f"[FILTER]: [{match_pattern}, [holder:{holder}, emotion:{emotion}, "   \
                            f"object:{fobject}, reason:{reason} by joint sentiment token not in same element!!!")
                        continue

            if len(fobject)>0 and self._span_matched_ratio(fobject)<0.25:
                logger.info(f"[FILTER]: [{match_pattern}, holder:{holder}, emotion:{emotion}, "   \
                            f"object:{fobject}, reason:{reason} by object span match ratio!!!")
                continue
            if len(reason)>0 and self._span_matched_ratio(reason)<0.25:
                logger.info(f"[FILTER]: [{match_pattern}, holder:{holder}, emotion:{emotion}, "   \
                            f"object:{fobject}, reason:{reason} by reason span match ratio!!!")
                continue

            all_match_tokenids = sorted(
                list(
                    set(holder+emotion+fobject+reason+\
                        list(meanless)+list(e_neg_tokenids)+list(r_neg_tokenids))))
            anchor_id  = result_anchor_ids[sorted_index[i]]
            extract_result.match_pattern = match_pattern
            extract_result.all_match_tokenids = all_match_tokenids
            _s, _e = all_match_tokenids[0], all_match_tokenids[-1]
            extract_result.tokens = [ t.text for t in doc[ all_match_tokenids[0]:all_match_tokenids[-1]+1 ] ]
            # extract_result.clause = ' '.join(extract_result.tokens)
            extract_result.clause = doc.text
            # clause_start = char_offset + doc[_s].idx
            clause_end   = char_offset + doc[_e].idx+len(doc[_e].text)
            # extract_result.clause_index = f"{clause_start},{clause_end}"
            extract_result.clause_index = f"{char_offset},{clause_end}"
            extract_result.anchor_id = anchor_id
            extract_result.anchor_lemma = doc[anchor_id].lemma_
            extract_result.anchor_text  = doc[anchor_id].text
            extract_result.holder.tokenids = holder
            extract_result.emotion.tokenids= emotion
            extract_result.object.tokenids = fobject
            extract_result.reason.tokenids = self._remove_prefix_punct_space(reason, doc)

            # element offset
            if holder:
                holder_index_start = char_offset + doc[holder[0]].idx
                holder_index_end   = char_offset + doc[holder[-1]].idx+len(doc[holder[-1]].text)
                extract_result.holder.index = f"{holder_index_start},{holder_index_end}"
            if emotion:
                emotion_index_start= char_offset + doc[emotion[0]].idx
                emotion_index_end  = char_offset + doc[emotion[-1]].idx+len(doc[emotion[-1]].text)
                extract_result.emotion.index= f"{emotion_index_start},{emotion_index_end}"
            if fobject:
                object_index_start = char_offset + doc[fobject[0]].idx
                object_index_end   = char_offset + doc[fobject[-1]].idx + len(doc[fobject[-1]].text)
                extract_result.object.index = f"{object_index_start},{object_index_end}"
            if reason:
                reason_index_start = char_offset + doc[reason[0]].idx
                reason_index_end   = char_offset + doc[reason[-1]].idx + len(doc[reason[-1]].text)
                extract_result.reason.index = f"{reason_index_start},{reason_index_end}"

            # extract_result.holder.text = ' '.join([doc[i].lemma_ for i in holder])
            # extract_result.emotion.text= ' '.join([doc[i].lemma_ for i in emotion])
            # extract_result.object.text = ' '.join([doc[i].lemma_ for i in fobject])
            # extract_result.reason.text = ' '.join([doc[i].lemma_ for i in reason])
            extract_result.holder.text = self._combine_str(holder, doc)
            extract_result.emotion.text= self._combine_str(emotion, doc)
            extract_result.object.text = self._combine_str(fobject, doc)
            extract_result.reason.text = self._combine_str(reason, doc)

            # 处理因为并列关系导致的误召回emotion
            if extract_result.emotion.text and \
                extract_result.emotion.text not in dataset.emotion_vocab:
                continue

            if len(fobject)==1:
                if fobject[0] in groups.pron_map:
                    # if "PERSONAL" in groups.pron_map[fobject[0]]:
                    if doc[fobject[0]].lemma_ in ( "it", "this", "these", "that", "something", "all", "everything", "they" ):
                        extract_result.object_type = "PRON"
                    else:
                        extract_result.object_type = "invalid_PRON"

                elif fobject[0] in groups.tacl_set:
                    extract_result.object_type = "TACL"

            if len(holder)==1:
                if holder[0] in groups.pron_map:
                    if "PERSONAL" in groups.pron_map[holder[0]]:
                        extract_result.holder_type = "PRON"
                    else:
                        extract_result.holder_type = "invalid_PRON"
                elif holder[0] in groups.tacl_set:
                    extract_result.holder_type = "TACL"


            # 过滤无意义结果
            if extract_result.object_type in ("invalid_PRON", "TACL"):
                logger.info(f"[FILTER]: by object INVALID PRON: {extract_result.object_type}")
                continue
            elif extract_result.holder_type in ("invalid_PRON", "TACL"):
                logger.info(f"[FILTER]: by holder INVALID PRON: {extract_result.holder_type}")
                continue

            extract_result.e_isneg = e_isneg
            extract_result.e_neg_tokenids = list(e_neg_tokenids)
            extract_result.r_isneg = r_isneg
            extract_result.r_neg_tokenids = list(r_neg_tokenids)
            extract_results.append(extract_result)

        extract_results = self.postprocess2(extract_results, cc_groups=groups.cc_groups)
        
        return extract_results
