# -*- coding: utf-8 -*-

import networkx as nx

# file is a list of links (pred-prey pairs)
def foodwebFromFile(filename):
    return nx.read_edgelist(filename, create_using=nx.DiGraph()) 