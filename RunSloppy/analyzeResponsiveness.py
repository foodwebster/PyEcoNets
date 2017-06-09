# -*- coding: utf-8 -*-

import sys
import os
sys.path.append("..")
from config import staticResultsPath
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

df = pd.DataFrame.from_csv(os.path.join(staticResultsPath, "Results_ResponseMag.csv"), index_col=None)
summarydf = pd.DataFrame.from_csv(os.path.join(staticResultsPath, "summary.csv"), index_col=None)

alldf = summarydf.merge(df, on='idx')

attrs = ['C', 'FracHerbiv', 'LperS', 'ShortPathMax', 'ShortPathMn',
         'ShortPathSD', 'TLMax', 'TLMn', 'TLSD','fracBasal', 'inDegSD',
         'nSpp', 'outDegSD', 'wtdMnTL']
y = alldf['logResp']
for attr in attrs:
    X = alldf[attr]
    X = sm.add_constant(X)
    est = sm.OLS(y, X).fit()
    est.summary()

plt.figure()
alldf['logResp'].hist()
# there is a considerable range in overall system responsiveness to parameter perturbation

alldf.plot.scatter('TLMn', 'bTot')
# there is a significant positive relationship between mean TL and total biomass:
# systems with higher biomass tend to also have higher mean trophic level.
alldf.plot.scatter('TLMn', 'wtdMnTL')
# Mean trophic level weighted by biomass is greater than mean TL because biomass tends to
# accumulates in high trophic level species

alldf.plot.scatter('wtdMnTL', 'bTot')
alldf.plot.scatter('wtdMnTL', 'logResp')
alldf.plot.scatter('bTot', 'logResp')

# System biomass-normalized responsiveness decreases with total biomass, and with mean trophic level
# Taller, top heavy food webs are less responsive to parameter perturbations

# and similarly for raw response:
#plt.figure()
#alldf['logRawResp'].hist()
#alldf.plot.scatter('TLMn', 'logRawResp')
#alldf.plot.scatter('bTot', 'logRawResp')
#alldf.plot.scatter('wtdMnTL', 'logRawResp')
