# -*- coding: utf-8 -*-

def getBasal(fw):
    return [nd for nd, deg in fw.out_degree_iter() if deg == 0]
            
def getTop(fw):
    return [nd for nd, deg in fw.in_degree_iter() if deg == 0]
            