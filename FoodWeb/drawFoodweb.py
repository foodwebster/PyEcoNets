# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 16:08:25 2016

@author: rich
"""

import numpy as np
from nxDrawTapered import draw_networkx_tapered
from collections import defaultdict

# node_color is a single color or list of colors
# highlight is an array of boolean to highlight node with a circle
# szAttr is map of nodeId:sizeAttr
def drawFoodweb(nw, tl, ax=None, title=None, szAttr=None, node_color='r', highlight=None, labels=None, minSz = 10, maxSz = 200):
    # build and order group of nodes at each trophic level
    posx = {}
    degree = nw.degree()
    tlGroups = defaultdict(list)
    for nid, tlval in tl.iteritems():
        tlval = int(round(tlval))
        tlGroups[tlval].append(nid)
    maxGroup = 0
    for nodes in tlGroups.values():
        nodes.sort(key=lambda x: degree[x])
        maxGroup = max(maxGroup, len(nodes))
        for i in range(len(nodes)):
            posx[nodes[i]] = i
    nNodes = nw.number_of_nodes()
    nodes = nw.nodes()
    tlMax = max(tl.values())
    tlMin = min(tl.values())
    yRange = tlMax - tlMin
    pos = {node:(tlMin + yRange*posx[node]/(maxGroup-1), tl[node]) for node in nodes}
    if szAttr is not None:
        szmax = max(szAttr.values())
        szmin = min(szAttr.values())
        szrange = float(szmax - szmin)
        sz = np.array([minSz + (maxSz - minSz) * (szAttr[node] - szmin)/szrange for node in nodes])
    else:
        sz = np.full(nNodes, maxSz/2)
        
    # normalize positions so edge widths are consistent
#    vals = [p[0] for p in pos.values()]
#    xMin = min(vals)
#    xMax = max(vals)
#    vals = [p[1] for p in pos.values()]
#    yMin = min(vals)
#    yMax = max(vals)
#    xRng = xMax - xMin
#    yRng = yMax - yMin
#    if xRng > 0 and yRng > 0:
#        pos = {idx:((p[0] - xMin)/xRng, (p[1] - yMin)/yRng) for idx, p in pos.iteritems()}   

    draw_networkx_tapered(nw, pos, ax=ax, node_size=sz, node_color=node_color, highlight=highlight, edge_color='0.9', labels=labels, horizontalalignment='left')
    ax.set_title(title)
    ax.get_xaxis().set_visible(False)


