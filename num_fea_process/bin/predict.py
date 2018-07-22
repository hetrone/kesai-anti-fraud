#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     predict
   Description :
   Author :        wangzhiyuan
   date：          2018/7/19
-------------------------------------------------
   Change Activity:
                   2018/7/19:
-------------------------------------------------
"""
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_string_dtype
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
from train_data import *

base_fea = ["PERSONID", "LABEL", "APPLYNO"]


def get_bin_idx(val, thresholds):
    bin_value = 0
    for v_bin, i in enumerate(thresholds[1:]):
        # vv = round(v,len(str(i))-2)
        if val <= i:
            bin_value = v_bin
            break
    return bin_value


def get_bound_value(x, ud_list):
    if x > ud_list[1]:
        x = ud_list[1]
    elif x < ud_list[0]:
        x = ud_list[0]
    return x


def get_bound_log(data, ud_dic, numeric_fea):
    for c in numeric_fea:
        if c in base_fea:
            continue
        data[c] = map(lambda x: get_bound_value(x, ud_dic[c]), data[c])
        data[c] = map(lambda x: np.log(x + 1), data[c])
    return data


def get_dic(file_path):
    f = open(file_path)
    data = f.readlines()
    dic = {}
    for line in data:
        dic.update(eval(line))
    return dic


def bin_process(data, cut_dic):
    for k in cut_dic:
        thresholds = cut_dic[k]
        data[k] = map(lambda x: get_bin_idx(x, thresholds), data[k])
    return data


def woe_process(data, woe_dic):
    for k in woe_dic:
        print k
        data[k] = data[k].map(woe_dic[k])
    return data


def process_fea():
    # test
    # test_path = "./../data/train_part.tsv"
    # data = pd.read_csv(test_path)

    file_path = "./../data/test_A.tsv"
    # file_path = "./../data/train.tsv"
    data = pd.read_csv(file_path,sep="\t")
    print "load data size",data.shape

    # 分开字符特征和数值特征
    string_fea, numeric_fea = get_str_num_fea(data)

    # 极大值/极小值处理 log 处理
    ud_dic = get_dic("./../conf/ud_dic.dict")
    data = get_bound_log(data, ud_dic, numeric_fea)

    # 特征组合
    # 抽象成一个函数，返回 选取的特征列 及 特征数据
    data = fea_combine(data, numeric_fea)

    # 过滤特征
    select_fea = get_dic("./../conf/save_fea.dict")
    data = data[select_fea["feature"]]

    # 按平均值填充
    data = fill_none(data)

    # 分箱编码
    cut_dic = get_dic("./../conf/cut_bin.dict")
    data = bin_process(data, cut_dic)

    # woe编码
    woe_dic = get_dic("./../conf/woe_encode.dict")
    data = woe_process(data, woe_dic)

    data.to_csv("./../output/test.csv", index=None)

    return data


def model_pre():
    data = pd.read_csv("./../output/test.csv")
    # data_id = pd.read_csv("./../data/train_id.tsv",sep="\t")

    # 添加标签
    # data = pd.merge(data, data_id, on="PERSONID")

    fea_col = list(set(data.columns.tolist()) - set(["LABEL", "PERSONID"]))
    data_x = data[fea_col]

    model = joblib.load("./../model/model")
    y_prob = model.predict_proba(data_x)

    # check
    # data_y = data["LABEL"]
    # auc = round(roc_auc_score(data_y, y_prob[:, 1]), 6)
    # print auc

    # keep result
    data["label"] = y_prob[:,1]
    res = data[["PERSONID", "label"]]
    print "保存文件大小： ",res.shape
    res.to_csv("./../output/predict_A.csv", sep="\t",index=None,header=None)




if __name__ == "__main__":
    # data = process_fea()
    model_pre()

    # check
    data = pd.read_csv("./../output/predict_A.csv",header=None)
    print data.shape
    print data.head()