
from math import exp, log
import networkx.algorithms.non_randomness

import random
print("Setting random seed")
random.seed(3)

from random import random, shuffle, choices
from functools import partial
from copy import deepcopy
import networkx as nx
from functools import reduce
from numpy.random import exponential
import numpy as np
from time import sleep
import matplotlib.pyplot as plt


class Lineage(object):
    """
    Lineages are the edges of our graph
    """
    def __init__(self, down, intervals):
        self.down = down # node at bottom of edte
        self.up = None # node at top of edge
        self.intervals = intervals

    def __repr__(self):
        return f'{self.down}:{self.intervals}'

class Node():
    """
    Leaf, Coalscent and Recombination are the nodes of the graph
    """
    def __hash__(self):
        return self.nodeid

    def __eq__(self, other):
        return self.nodeid == other.nodeid  

    def __repr__(self):
        return f'{self.nodeid}'

class Leaf(Node):

    def __init__(self, nodeid, height):
        self.nodeid = nodeid
        self.height = height
        self.intervals = [(0, 1)]
        self.parent = None
        self.xpos = None

class Coalescent(Node):

    def __init__(self, nodeid, height, children):
        self.nodeid = nodeid
        self.height = height
        self.children = children
        self.parent = None
        self.xpos = None

class Recombination(Node):

    def __init__(self, nodeid, height, child, recomb_point):
        self.nodeid = nodeid
        self.height = height
        self.child = child
        self.left_parent = None
        self.right_parent = None
        self.recomb_point = recomb_point
        self.xpos = None

def flatten(list_of_tps):
 
    return reduce(lambda ls, ival: ls + list(ival), list_of_tps, [])


def unflatten(list_of_endpoints):

    return [ [list_of_endpoints[i], list_of_endpoints[i + 1]]
          for i in range(0, len(list_of_endpoints) - 1, 2)]

def merge(query, annot, op):

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

def x_positions_traverse(node, offset):
    """
    Recursive function for adding x positions
    """
    offset -= 1
    if type(node) is Leaf:
        return 1 # offset
        # do nothing
    if type(node) is Recombination:
        # add xpos to children
        node.child.down.xpos = node.xpos
        offset = x_positions_traverse(node.child.down, offset)
        # add xpos to node if both parents are positioned
        if node.left_parent.up.xpos is not None and node.right_parent.up.xpos is not None:
            left, right = sorted([node.left_parent.up.xpos, node.right_parent.up.xpos])
            x = left + (right - left) / 2 
            node.xpos = x
    if type(node) is Coalescent:
        # add xpos to children
        node.children[0].down.xpos = node.xpos - offset
        node.children[1].down.xpos = node.xpos + offset
        for child in node.children:
            x_positions_traverse(child.down, offset)

def add_node_x_positions(nodes):
    """
    Adds x positions in place
    """
    offset = len(nodes)-1
    nodes[-1].xpos = offset
    x_positions_traverse(nodes[-1], offset)

def get_arg_nodes(n=5, N=10000, r=1e-8, L=2e3):
    """
    Simulates an ARG
    """
    # because we use the sequence interval from 0 to 1
    r = r * L

    nodes = list()
    live = list() # live lineages

    for i in range(n):
        leaf = Leaf(i, height=0)
        nodes.append(leaf)
        lin = Lineage(down=leaf, intervals=[(0, 1)])
        live.append(lin)
        last_node = i # max node number used so far

    while len(live) > 1:
        shuffle(live)

        coal_prob = (len(live) * len(live)-1) / 2 / (2*N) 
        tot_ancestral = sum(interval_sum(x.intervals) for x in live)
        rec_prob = r * tot_ancestral
        wating_time = exponential(1 / (coal_prob + rec_prob))
        height = wating_time + nodes[-1].height
        if random() < coal_prob / (coal_prob + rec_prob):
            # coalescence
            lin_a, lin_b = live.pop(), live.pop()
            intervals = interval_union(lin_a.intervals, lin_b.intervals)
            # new node
            node_c = Coalescent(nodeid=last_node+1, height=height, 
                        children=[lin_a, lin_b])
            last_node += 1
            nodes.append(node_c)

            # add node to top of coalescing lineages
            lin_a.up = node_c
            lin_b.up = node_c

            # new lineage
            lin_c = Lineage(down=node_c, intervals=intervals) # fixme intervals
            live.append(lin_c)
            node_c.parent = lin_c
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
                child=rec_lin, recomb_point=recomb_point)
            last_node += 1
            nodes.append(rec_node)

            # two new lineages both refering back to recombination node
            intervals_a, intervals_b = interval_split(rec_lin.intervals, recomb_point)
            assert interval_sum(intervals_a) and interval_sum(intervals_b) 
            lin_a = Lineage(down=rec_node, intervals=intervals_a)
            lin_b = Lineage(down=rec_node, intervals=intervals_b)
            live.append(lin_a)
            live.append(lin_b)

            # add parents of node
            rec_node.left_parent = lin_a
            rec_node.right_parent = lin_b

            # add parent for child
            rec_lin.up = rec_node

    add_node_x_positions(nodes)

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

def get_positions(nodes):
    """
    Gets list of x,y positions for nodes
    """
    positions = list()
    for node in nodes:
        positions.append((node.xpos, node.height))
    return positions

def get_breakpoints(nodes):
    """
    Gets list of recombination break points
    """    
    # get list of break points
    breakpoints = list()
    for n in nodes:
        if type(n) is Recombination:
            breakpoints.append(n.recomb_point)
    return sorted(breakpoints)

def traverse_marginal(node, interval):
    """
    Recursive function for getting marginal tree/ARG
    """    
    node = deepcopy(node)
    tree_nodes = set()
    if type(node) is Leaf:
        tree_nodes.add(node)
    if type(node) is Recombination:
        if interval_intersect([interval], node.child.intervals):
            tree_nodes.add(node)
            tree_nodes.update(traverse_marginal(node.child.down, interval))
    elif type(node) is Coalescent:
        if node.parent is None or interval_intersect([interval], node.parent.intervals):
            tree_nodes.add(node)
        del_child = None
        for i, child in enumerate(node.children):
            if interval_intersect([interval], child.intervals):
                tree_nodes.update(traverse_marginal(child.down, interval))
            else:
                del_child = i
        if del_child is not None:
            del node.children[del_child]
    return tree_nodes

def remove_dangling_root(tree_nodes):
    """
    Remove the nodes of a marginal tree or ARG 
    from root to first coalescence
    """    
    for_del = list()
    for i in range(len(tree_nodes)-1, -1, -1):
        if type(tree_nodes[i]) is Coalescent and len(tree_nodes[i].children) == 2:
            break
        for_del.append(i)
    for i in for_del:
        del tree_nodes[i]

def marginal_arg(nodes, interval):
    """
    Gets the marginal ARG given a sequene interval
    """    
    # get nodes for marginal
    marg_nodes = traverse_marginal(nodes[-1], list(interval))

    # set to list
    marg_nodes = list(marg_nodes)

    # sort on height
    marg_nodes.sort(key=lambda x: x.height)
    # prune top path above last coalescence
    remove_dangling_root(marg_nodes)

    return marg_nodes

def marginal_trees(nodes):
    """
    Gets list of marginal trees
    """
    tree_list = list()
    breakpoints = get_breakpoints(nodes)
    borders = [0] + breakpoints + [1]
    for interval in zip(borders[:-1], borders[1:]):
        marg_nodes = marginal_arg(nodes, interval)

        tree_list.append(marg_nodes)
    return tree_list

def draw_graph(nodes):
    """
    Draws graph using matplotlib
    """    
    # make a graph and positions list
    positions = list()
    arg = nx.Graph()
    for node in nodes:
        arg.add_node(node.nodeid)
        positions.append((node.xpos, node.height))
        if isinstance(node, Recombination):
            arg.add_edge(node.child.down.nodeid, node.nodeid)
        elif isinstance(node, Coalescent):
            for child in node.children:
                arg.add_edge(child.down.nodeid, node.nodeid)

    positions = np.array(positions)
    positions = dict(zip(arg.nodes(), positions))

    #pos = nx.spring_layout(arg)
    nx.draw(arg, positions, alpha=0.5, node_size=200, with_labels=True)
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

if __name__ == '__main__':

    # get arg and add positions
    nodes = get_arg_nodes()

    # get breakpoints
    breakpoints = get_breakpoints(nodes)

    # get marginal trees
    trees = marginal_trees(nodes)

    # marginal arg for some consequtive intervals
    marg_arg = marginal_arg(nodes, [0, breakpoints[1]])


    # draw graphs for testing
    draw_graph(nodes)
    draw_graph(marg_arg)

    draw_graph(nodes)
    for tree in marginal_trees(nodes):
        print([n.xpos for n in tree])
        draw_graph(tree)


