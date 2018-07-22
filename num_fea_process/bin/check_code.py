#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     check_code.py
   Description :
   Author :        wangzhiyuan
   date：          2018/7/21
-------------------------------------------------
   Change Activity:
                   2018/7/21:
-------------------------------------------------
"""

import pandas as pd

data1 = pd.read_csv("./../output/model_data.csv")
data2 = pd.read_csv("./../output/test.csv")

c1 = data1.columns.tolist()
c2 = data2.columns.tolist()

print len(c1),len(c2)

col = list(set(c1)&set(c2))
print set(c1) - set(c2)
print len(col)

feas= []
for c in col:
    tag = False
    res = 0
    a = data1[c].values.tolist()
    b = data2[c].values.tolist()
    for i in range(len(a)):
        if a[i] != b[i]:
            tag = True
            res = a[i],b[i]
            name1 = data1.loc[i,"PERSONID"]
            name2 = data2.loc[i,"PERSONID"]
            break
    if tag :
        feas.append(c)
        print c, res,name1,name2


print "diff nums",len(feas)