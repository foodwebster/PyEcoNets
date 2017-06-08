# -*- coding: utf-8 -*-

import os
from os.path import expanduser
import pandas as pd

os.chdir(expanduser("~/OneDrive/Rich/Software/PythonFoodWebs/Weddell"))

pred = 'con.taxonomy'
prey = 'res.taxonomy'

linkFile = "WeddellLinks.xls"
spFile = "WeddellSpecies.xls"
sharkLinkFile = "WeddellSharkLinks.xls"

ldf = pd.read_excel(linkFile)
sldf = pd.read_excel(sharkLinkFile)
sdf = pd.read_excel(spFile).drop_duplicates(subset=[pred])

sMap = dict(zip(sdf[pred], xrange(1,1+len(sdf))))
sharks = set(sldf[pred].unique().tolist())

predId = ldf['Consumer'].apply(lambda x: sMap[x])
preyId = ldf['Resource'].apply(lambda x: sMap[x])

fwdf = pd.DataFrame(list(zip(predId,preyId)))
fwdf.to_csv("Weddell.txt", sep=' ', index=False, header=False)

sdf['id'] = xrange(1,1+len(sdf))
sdf['isShark'] = sdf[pred].apply(lambda x: x in sharks)

sdf.to_csv("WeddellSpp.csv", index=False, header=True)

