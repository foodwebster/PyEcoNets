# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import networkx as nx
from FoodWeb.TrophicLevel import computeSWTL
from FoodWeb.TrophicLevel import computeShortPathTL

def resultsSummary(results, resultspath):
    summary = []
    for idx, (modelInfo, resultsRel, nBigRel, eigvalRel, eigvecRel, resultsAbs, nBigAbs, eigvalAbs, eigvecAbs) in enumerate(results):
        numLinks = len(modelInfo['network'])
        nSpp = float(modelInfo['S'])
        nw = nx.parse_edgelist(modelInfo['network'], create_using=nx.DiGraph())
        tls = np.array(computeSWTL(nw).values())
        shortpaths = np.array(computeShortPathTL(nw).values())
        summary.append({'idx': modelInfo['modelIdx'],
                        'nSpp': nSpp,
                        'LperS': numLinks/nSpp,
                        'C': numLinks/(nSpp*nSpp),
                        'fracBasal': modelInfo['B']/nSpp,
                        'FracHerbiv': (tls==2).sum()/nSpp,
                        'inDegSD': np.array(nw.in_degree().values()).std(),
                        'outDegSD': np.array(nw.out_degree().values()).std(),
                        'TLMn': tls.mean(),
                        'TLSD': tls.std(),
                        'TLMax': tls.max(),
                        'ShortPathMn': shortpaths.mean(),
                        'ShortPathSD': shortpaths.std(),
                        'ShortPathMax': shortpaths.max(),
                        'wtdMnTL': modelInfo['wtdMnTL'],
                        'bTot': modelInfo['binit'].sum(),
                        'bMax': modelInfo['binit'].max()})
    pd.DataFrame(summary).to_csv(os.path.join(resultspath, "summary.csv"), index=False)