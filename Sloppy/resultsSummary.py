# -*- coding: utf-8 -*-
import pandas as pd

def resultsSummary(results):
    cols = ['nSpp', 'L/S', 'C', 'fracBasal', 'wtdMnTL', 'bTot', 'bMax']
    summary = []
    for idx, (modelInfo, resultsRel, nBigRel, eigvalRel, eigvecRel, resultsAbs, nBigAbs, eigvalAbs, eigvecAbs) in enumerate(results[0]):
        numLinks = len(modelInfo['network'])
        nSpp = float(modelInfo['S'])
        summary.append([nSpp, \
                        numLinks/nSpp, \
                        numLinks/(nSpp*nSpp), \
                        modelInfo['B']/nSpp, \
                        modelInfo['wtdMnTL'], \
                        modelInfo['binit'].sum(), \
                        modelInfo['binit'].max()])
    pd.DataFrame(summary, columns = cols).to_csv("../SloppyResults/summary.csv", index=False)