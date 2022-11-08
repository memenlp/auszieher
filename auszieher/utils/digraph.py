#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2022 Fisher. All rights reserved.
#   
#   文件名称：digraph.py
#   创 建 者：YuLianghua
#   创建日期：2022年03月02日
#   描    述：有向图数据结构及 path 判断
#
#================================================================

import traceback
from collections.abc import Mapping

def _bidirectional_pred_succ(G, source, target):
    """Bidirectional shortest path helper.
    Returns (pred, succ, w) where
    pred is a dictionary of predecessors from w to the source, and
    succ is a dictionary of successors from w to the target.
    """
    # does BFS from both source and target and meets in the middle
    if target == source:
        return ({target: None}, {source: None}, source)

    # handle either directed or undirected
    if G.is_directed():
        Gpred = G.pred
        Gsucc = G.succ
    else:
        Gpred = G.adj
        Gsucc = G.adj

    # predecesssor and successors in search
    pred = {source: None}
    succ = {target: None}

    # initialize fringes, start with forward
    forward_fringe = [source]
    reverse_fringe = [target]

    while forward_fringe and reverse_fringe:
        if len(forward_fringe) <= len(reverse_fringe):
            this_level = forward_fringe
            forward_fringe = []
            for v in this_level:
                for w in Gsucc[v]:
                    if w not in pred:
                        forward_fringe.append(w)
                        pred[w] = v
                    if w in succ:  # path found
                        return pred, succ, w
        else:
            this_level = reverse_fringe
            reverse_fringe = []
            for v in this_level:
                for w in Gpred[v]:
                    if w not in succ:
                        succ[w] = v
                        reverse_fringe.append(w)
                    if w in pred:  # found path
                        return pred, succ, w

    raise ValueError(f"No path between {source} and {target}.")

class AtlasView(Mapping):
    r"""
    展示 dict-of-dict 格式的数据结构;
    """

    __slots__ = ("_atlas",)

    def __getstate__(self):
        return {"_atlas": self._atlas}

    def __setstate__(self, state):
        self._atlas = state["_atlas"]

    def __init__(self, d):
        self._atlas = d

    def __len__(self):
        return len(self._atlas)

    def __iter__(self):
        return iter(self._atlas)

    def __getitem__(self, key):
        return self._atlas[key]

    def copy(self):
        return {n: self[n].copy() for n in self._atlas}

    def __str__(self):
        return str(self._atlas)  # {nbr: self[nbr] for nbr in self})

    def __repr__(self):
        return f"{self.__class__.__name__}({self._atlas!r})"

class AdjacencyView(AtlasView):
    r"""
    展示 dict-of-dict-of-dict 格式的数据结构: 
    e.g.:
      {0: {1: {}}, 1: {2: {}}, 2: {3: {}}, 3: {}}
    """
    def __getitem__(self, name):
        return AtlasView(self._atlas[name])

    def copy(self):
        return {n: self[n].copy() for n in self._atlas}

class DiGraph(object):
    def __init__(self):
        self.graph = dict()
        self._node = dict() 
        self._adj  = dict()
        self._pred = dict()
        self._succ = self._adj

    @property
    def adj(self):
        return AdjacencyView(self._adj)

    @property
    def pred(self):
        return AdjacencyView(self._pred)

    @property
    def succ(self):
        return AdjacencyView(self._succ)

    def is_directed(self):
        return True

    def add_edge(self, u, v, **attr):
        if u not in self._succ:
            if u is None:
                raise ValueError("None cannot be a node")
            self._succ[u] = dict()
            self._pred[u] = dict()
            self._node[u] = dict()
        if v not in self._succ:
            if v is None:
                raise ValueError("None cannot be a node")
            self._succ[v] = dict()
            self._pred[v] = dict()
            self._node[v] = dict()
        # add the edge
        datadict = self._adj[u].get(v, dict())
        datadict.update(attr)
        self._succ[u][v] = datadict
        self._pred[v][u] = datadict

    def has_path(self, source, target):
        try:
            self.shortest_path(source, target)
        except Exception as e:
            # print(traceback.format_exc())
            return False
        return True

    def shortest_path(self, source, target):
        paths = self.bidirectional_shortest_path(source, target)
        return paths

    def bidirectional_shortest_path(self, source, target):
        if source not in self._node or target not in self._node:
            msg = f"Either source {source} or target {target} is not in G"
            raise ValueError(msg)

        # call helper to do the real work
        results = _bidirectional_pred_succ(self, source, target)
        pred, succ, w = results

        # build path from pred+w+succ
        path = []
        # from source to w
        while w is not None:
            path.append(w)
            w = pred[w]
        path.reverse()
        # from w to target
        w = succ[path[-1]]
        while w is not None:
            path.append(w)
            w = succ[w]

        return path

if __name__ == "__main__":
    DG = DiGraph()
    DG.add_edge(5,8)
    DG.add_edge(8,10)
    DG.add_edge(10,12)
    DG.add_edge(5,14)
    DG.add_edge(14,18)
    print(DG.pred)
    print(DG.succ)
    r = DG.has_path(14,8)
    print(r)
