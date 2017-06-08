# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 12:05:58 2016

@author: rich
"""

from drawFoodweb import drawFoodweb
import numpy as np
from NicheModel import nicheModel

np.random.seed(1)
nw, nn, cc, rr, tl = nicheModel(5, 0.25)
#nw.add_edge(0,1)
#tl = {0:1.0, 1:2.0}
node_color=['r', '0.7', '0.7', 'y', 'c']
highlight=[True, False, False, False, False]
drawFoodweb(nw, tl, node_color=node_color, highlight=highlight)
