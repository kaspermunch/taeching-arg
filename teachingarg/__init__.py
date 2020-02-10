
 
"""
# Poppen Dashboard

Make ARG

ARG class:
- All nodes in a list
- Each node has attributes: id, x, y, ancestral_intervals
- Copy interval functions from genominterv to manage intervals and add function to split a set of intervals at a break point.
- leaves of class Leaf
- Coalescence nodes of class Coalescence
	- Attributes: id, x, y, ancestral_intervals, children.
- Recombination nodes of class Recombination.
	- Attributes is, x, y, break_point, child.
	- Upon recombination, duplicate node and split ancestral intervals. 

What class should nodes created by recombination be?

Generate ARG

Do constrained spring layout and populate x and y attributes of each node.

Get tree sequence bottom up. Start with one tree and add coalescence events 
Upon recombination duplicate tree that has the breakpoint


"""
from math import exp, log
import networkx.algorithms.non_randomness

import random
print("Setting random seed")
random.seed(3)

from random import random, shuffle, choices
from functools import partial
import networkx as nx
from functools import reduce

from numpy.random import exponential

import numpy as np

from time import sleep


# Lineages are the edges of our graph

class Lineage(object):

    def __init__(self, node, intervals):
        self.node = node
        self.intervals = intervals

    def __repr__(self):
        return f'{self.node}:{self.intervals}'

# Leaf, Coalscent and Recombination are the nodes of the graph

class Leaf(object):

    def __init__(self, nodeid, height):
        self.nodeid = nodeid
        self.height = height
        self.intervals = [(0, 1)]
        # self.parents = []
        self.parent = None
        self.xpos = None

    def __repr__(self):
        return f'{self.nodeid}'

class Coalescent(object):

    def __init__(self, nodeid, height, children):
        self.nodeid = nodeid
        self.height = height
        self.children = children
        # self.parents = []
        self.parent = None
        self.xpos = None

    def __repr__(self):
        return f'{self.nodeid}'

class Recombination(object):

    def __init__(self, nodeid, height, child, recomb_point):
        self.nodeid = nodeid
        self.height = height
        self.child = child
        # self.parents = []        
        self.left_parent = None
        self.right_parent = None
        self.recomb_point = recomb_point
        self.xpos = None

    def __repr__(self):
        return f'{self.nodeid}'


# In all of the following, the list of intervals must be sorted and 
# non-overlapping. We also assume that the intervals are half-open, so
# that x is in tp(start, end) iff start <= x and x < end.

def flatten(list_of_tps):
    """
    Convert a list of sorted intervals to a list of endpoints.
    :param query: Sorted list of (start, end) tuples.
    :type query: list
    :returns: A list of interval ends
    :rtype: list
    """
    return reduce(lambda ls, ival: ls + list(ival), list_of_tps, [])


def unflatten(list_of_endpoints):
    """
    Convert a list of sorted endpoints into a list of intervals.
    :param query: Sorted list of ends.
    :type query: list
    :returns: A list of intervals.
    :rtype: list
    """
    return [ [list_of_endpoints[i], list_of_endpoints[i + 1]]
          for i in range(0, len(list_of_endpoints) - 1, 2)]


def merge(query, annot, op):
    """
    Merge two lists of sorted intervals according to the boolean function op.
    :param query: List of (start, end) tuples.
    :type query: list
    :param query: List of (start, end) tuples.
    :type query: list
    :param op: Boolean function taking two a bolean arguments.
    :type op: function
    :returns: A list of interval merged according to op
    :rtype: list
    """
    a_endpoints = flatten(query)
    b_endpoints = flatten(annot)

    assert a_endpoints == sorted(a_endpoints), "not sorted or non-overlaping"
    assert b_endpoints == sorted(b_endpoints), "not sorted or non-overlaping"


    sentinel = max(a_endpoints[-1], b_endpoints[-1]) + 1
    a_endpoints += [sentinel]
    b_endpoints += [sentinel]

    a_index = 0
    b_index = 0

    res = []

    scan = min(a_endpoints[0], b_endpoints[0])
    while scan < sentinel:
        in_a = not ((scan < a_endpoints[a_index]) ^ (a_index % 2))
        in_b = not ((scan < b_endpoints[b_index]) ^ (b_index % 2))
        in_res = op(in_a, in_b)

        if in_res ^ (len(res) % 2):
            res += [scan]
        if scan == a_endpoints[a_index]: 
            a_index += 1
        if scan == b_endpoints[b_index]: 
            b_index += 1
        scan = min(a_endpoints[a_index], b_endpoints[b_index])

    return unflatten(res)

def interval_diff(a, b):
    if not (a and b):
        return a and a or b
    return merge(a, b, lambda in_a, in_b: in_a and not in_b)

def interval_union(a, b):
    if not (a and b):
        return []
    return merge(a, b, lambda in_a, in_b: in_a or in_b)

def interval_intersect(a, b):
    if not (a and b):
        return []
    return merge(a, b, lambda in_a, in_b: in_a and in_b)
    
def interval_sum(intervals):
    return sum(e - s for (s, e) in intervals)

def interval_split(intervals, pos):
    left, right = list(), list()
    for s, e in intervals:
        if pos >= e:
            left.append((s, e))
        elif pos >= s and pos < e:
            left.append((s, pos))
            right.append((pos, e))
        else:
            right.append((s, e))
    return left, right

def get_arg_nodes(n=5, N=10000, r=1e-8, L=2e3):

    # because we use the sequence interval from 0 to 1
    r = r * L

    nodes = list()
    live = list() # live lineages

    for i in range(n):
        n = Leaf(i, height=0)
        nodes.append(n)
        lin = Lineage(n, [(0, 1)])
        live.append(lin)
        last_node = i # max node number used so far

    while len(live) > 1:
        shuffle(live)

        coal_prob = (len(live) * len(live)-1) / 2 / (2*N) 
        tot_ancestral = sum(interval_sum(x.intervals) for x in live)
        rec_prob = r * tot_ancestral
        wating_time = exponential(1 / (coal_prob + rec_prob))
        height = wating_time + nodes[-1].height
        print(coal_prob, rec_prob, tot_ancestral)
        if random() < coal_prob / (coal_prob + rec_prob):
            # coalescence
            lin_a, lin_b = live.pop(), live.pop()
            intervals = interval_union(lin_a.intervals, lin_b.intervals)

            # new node
            node_c = Coalescent(nodeid=last_node+1, height=height, 
                        children=[lin_a.node, lin_b.node])
            last_node += 1
            nodes.append(node_c)

            # new lineage
            lin_c = Lineage(node_c, intervals) # fixme intervals
            live.append(lin_c)

            # add parent for children
            lin_a.node.parent = node_c
            lin_b.node.parent = node_c

        else:
            # recombination
            rec_lin = choices(live, weights=[interval_sum(x.intervals) for x in live], k=1)[0]
            live.remove(rec_lin)

            # total ancestral material
            total_anc = interval_sum(rec_lin.intervals)

            # recombination point in ancestral material
            recomb_point_anc = random() * total_anc

            # recombination point in full sequence
            cum = 0
            for s, e in rec_lin.intervals:
                cum += e - s
                if cum > recomb_point_anc:
                    recomb_point = e - (cum - recomb_point_anc)
                    break

            # recombination node
            rec_node = Recombination(nodeid=last_node+1, height=height, 
                child=rec_lin.node, recomb_point=recomb_point)
            last_node += 1
            nodes.append(rec_node)

            # two new lineages both refering back to recombination node
            intervals_a, intervals_b = interval_split(rec_lin.intervals, recomb_point)
            assert interval_sum(intervals_a) and interval_sum(intervals_b) 
            lin_a = Lineage(rec_node, intervals_a)
            lin_b = Lineage(rec_node, intervals_b)
            print('r', intervals_a, intervals_b)
            live.append(lin_a)
            live.append(lin_b)

            # add parents of node
            rec_node.left_parent = lin_a.node
            rec_node.right_parent = lin_b.node

            # add parent for child
            rec_lin.node.parent = rec_node

    return nodes

def _rescale_layout(pos, scale=1):
    # rescale to (0,pscale) in all axes

    # shift origin to (0,0)
    lim = 0  # max coordinate for all axes
    for i in range(pos.shape[1]):
        pos[:, i] -= pos[:, i].min()
        lim = max(pos[:, i].max(), lim)
    # rescale to (0,scale) in all directions, preserves aspect
    for i in range(pos.shape[1]):
        pos[:, i] *= scale / lim
    return pos

def add_node_x_positions(nodes):

    for _ in range(len(nodes)*10):
        try:
            nodes[-1].xpos = 0

            for node in reversed(nodes):
                if node.xpos is None:
                    # if type(node.parents[0]) is Recombination:
                    #     node.xpos = node.parents[0].xpos
                    if type(node.parent) is Recombination:
                        node.xpos = node.parent.xpos

            for i, node in enumerate(reversed(nodes)):
                if type(node) is Coalescent:
                    left, right = node.children
                    left.xpos = node.xpos - (len(nodes)-1-i)
                    right.xpos = node.xpos + (len(nodes)-1-i)

            for node in reversed(nodes):
                if type(node) is Recombination:
                    # left, right = sorted([n.xpos for n in node.parents])
                    left, right = sorted([node.left_parent.xpos, node.right_parent.xpos])
                    x = left + (right - left) / 2 
                    node.xpos = x
        except:
            pass

def get_positions(nodes):

    positions = list()
    for node in nodes:
        positions.append((node.xpos, node.height))
    return positions

def get_breakpoints(nodes):
    # get list of break points
    breakpoints = list()
    for n in nodes:
        if type(n) is Recombination:
            breakpoints.append(n.recomb_point)
    return sorted(breakpoints)

# def condition(node, child, leftbreak=None, rightbreak=None):

#     if type(child) is not Recombination:
#          return True
#     else:
#         return child.left_parent == node and child.recomb_point == leftbreak \
#            or child.right_parent == node and child.recomb_point == rightbreak

# # what about break points and diamond recombinations?

# def tree_nodes(node, visit):

#     nodes = [node]
#     if type(node) is Leaf:
#         return nodes
#     elif type(node) is Recombination:
#         nodes[:0] = tree_nodes(node.child, visit)
#         return nodes
#     else:
#         for child in node.children:  # could there be a sorting problem if coalescence children are not sorted?
#             if visit(node, child):    # I guess not if we always use them in the same order.
#                 nodes[:0] = tree_nodes(child, visit)
#         return nodes
#         # return node with children given by build_tree()

# def marginal_tree(node, leftbreak, rightbreak):
    
#     should_visit = partial(condition, 
#         leftbreak=leftbreak, rightbreak=rightbreak)

#     return tree_nodes(node, should_visit)

# def marginal_tree(nodes, interval):
#     tree_nodes = list()
#     for node in nodes:
#         if [interval] == interval_intersect([interval], node.intervals):
#             tree_nodes.append(node)
#     return tree_nodes



if __name__ == '__main__':

        # produce nodes in arg
    nodes = get_arg_nodes()
    print(nodes)
    add_node_x_positions(nodes)

    positions = get_positions(nodes)

    breakpoints = get_breakpoints(nodes)

    print(breakpoints)

    # print(marginal_tree(nodes, (breakpoints[1], breakpoints[2])))
    # print([n.intervals for n in nodes])

    if True:
        ###############################################
        # draw a networkx graph
        import matplotlib.pyplot as plt

        # make a graph and positions list
        positions = list()
        arg = nx.Graph()
        for node in nodes:
            arg.add_node(node.nodeid)
            positions.append((node.xpos, node.height))
            if isinstance(node, Recombination):
                arg.add_edge(node.child.nodeid, node.nodeid, length=node.height-node.child.height)
            elif isinstance(node, Coalescent):
                for child in node.children:
                    arg.add_edge(child.nodeid, node.nodeid, length=node.height-child.height)

        positions = np.array(positions)

        #positions = dict(zip(arg.nodes(), positions))

        #pos = nx.spring_layout(arg)
        nx.draw(arg, positions, alpha=0.5, node_size=20)
        #with_labels=True, 
        #connectionstyle='arc3, rad = 0.1', 
        #arrowstyle='-')

        # G = nx.DiGraph() #or G = nx.MultiDiGraph()
        # G.add_node('A')
        # G.add_node('B')
        # G.add_edge('A', 'B', length = 2)
        # G.add_edge('B', 'A', length = 3)
        # pos = nx.spring_layout(G)
        # nx.draw(G, pos, with_labels=True, connectionstyle='arc3, rad = 0.1')
        # edge_labels=dict([((u,v,),d['length'])
        #              for u,v,d in G.edges(data=True)])

        plt.show()
        ###############################################







