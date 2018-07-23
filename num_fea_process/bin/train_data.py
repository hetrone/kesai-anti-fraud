# coding: utf-8


# base
import math

import pandas as pd
import numpy as np
import sys
import datetime
import re
# data process
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures, LabelBinarizer
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_string_dtype
import seaborn as sns
# model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegressionCV, LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
# model test
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

base_fea = ["PERSONID", "LABEL"]


def get_mean(df):
    return df.mean()


def get_sum(df):
    return df.sum()


def get_std(df):
    return df.std()

    # 都是一样的值，最后只添加一列就可以


def get_count(df):
    # df.name += get_count.__name__.split("_")[1]
    return len(df)


def fea_combine(data, numeric_fea):
    print "开始特征组合。。。"
    # 数值特征组合
    # 求均值/求和/求方差/求频数,将生成的DF保存到一个列表中
    func_list = [get_mean, get_sum, get_std, get_count]
    df_list = [0] * len(func_list)
    n = 0
    for f in func_list:
        df_list[n] = data[numeric_fea + ["PERSONID"]].groupby("PERSONID").agg(f)
        df_list[n].columns = [col_name + "_" + f.__name__.split("_")[1] for col_name in df_list[n].columns]
        n += 1

    # 将列表中DF连接成一个DATAFRAME
    df_all = df_list[0]
    # print df_all.shape
    for i in range(1, len(df_list) - 1):
        df_all = pd.merge(df_all, df_list[i], how="inner", right_index=True, left_index=True)

    df_count = pd.DataFrame(df_list[3]["FTR0_count"])
    # df_all = pd.merge(df_all, df_count, how="inner", on="PERSONID")
    df_all = pd.merge(df_all, df_count, how="inner", right_index=True, left_index=True)
    df_all.rename(columns={"FTR0_count": "FTR_count"}, inplace=True)
    df_all["PERSONID"] = df_all.index.values
    df_all.index.name = "id"
    return df_all

def string_fea_combine(data,string_fea):
    # TODO
    # 字符特征组合
    # string_fea ['PERSONID', 'APPLYNO', 'FTR51', 'CREATETIME']
    # # 建立一个FTR51 字典，有组合特征
    tmp = data["FTR51"].values.tolist()
    FTR_dic = {}
    for t in tmp:
        for i in t.split(","):
            FTR_dic.setdefault(i,0)
            FTR_dic[i] += 1
    count = 0
    for k,v in sorted(FTR_dic.items(),key=lambda x:x[1],reverse=True):
        print k,v
        count += 1
        if count == 10:
            break
    print data["FTR51"].value_counts()
    return data


def get_train_data():
    # 训练样本 ：(1368146, 55)，总计15000个不同的用户，日期为2015-3-1至2016-2-23，不均匀分布,正负bi：14230:770
    data_path = "./../data/train.tsv"
    data = pd.read_csv(data_path, sep='\t')

    # test_path = "./../data/train_part.tsv"
    # data = pd.read_csv(test_path)
    # data.to_csv(test_path,index=None)

    # label 值
    # data_id = "./data/train_id.tsv"
    # data_id = "./data/train_id.tsv"
    # dat_id = pd.read_csv(data_id,sep="\t")
    # data = pd.merge(dat,dat_id, how="inner", on="PERSONID")

    return data


def get_test_data():
    # 需要预测的样本 ： (232502, 55)，总计2500个用户,日期为2015-3-1至2016-2-23，不均匀分布
    data_path = "./data/test_A.tsv"
    # data_path = "./data/test_A.tsv"
    test_data = pd.read_csv(data_path, sep="\t")
    # print test_data.head()
    print "test_data size", test_data.shape
    print "unique predict PESSONID", len(test_data.PERSONID.unique())
    # test_time = test_data.CREATETIME.value_counts().to_dict()
    # sorted(test_time.keys())
    return test_data


def get_str_num_fea(data):
    cols = data.columns.tolist()
    string_fea = []
    numeric_fea = []
    que_fea = []
    for c in cols:
        if is_string_dtype(data[c]):
            string_fea.append(c)
        elif is_numeric_dtype(data[c]):
            numeric_fea.append(c)
        else:
            que_fea.append(c)
    print "字符特征/数值特征/待定特征数量依次为： ", map(len, [string_fea, numeric_fea, que_fea])
    return string_fea, numeric_fea


def bound_log_process(data, numeric_fea):
    def mm_replace(x):
        if x > maxx:
            x = maxx
        elif x < minn:
            x = minn
        else:
            x = x
        return x

    ud_dic = {}
    for c in numeric_fea:
        # 天花板地板处理
        minn = data[c].quantile(0.0001)
        maxx = data[c].quantile(0.9999)
        # print "processing col: ",c
        data[c] = map(mm_replace, data[c])

        # 保存极大值极小值结果
        ud_dic[c] = [minn, maxx]

        # log处理
        data[c] = map(lambda x: np.log(x + 1), data[c])
    return data, ud_dic


def save_dic(path, dic):
    fo = open(path, "w")
    fo.write(str(dic))
    fo.close()


def get_filter_fea(data):
    cols = data.columns.tolist()
    single_value_fea = []
    nan_fea = []
    same_val_fea = []
    mutil_fea = []

    rows = data.shape[0]
    count = 0
    for c in cols:
        # count += 1
        # if count % 100 == 0:
        #     print "processed ",count

        # 跳过索引值，标签值
        if c in base_fea:
            continue

        kv_dic = data[c].value_counts().to_dict()
        fea_name, fea_count = sorted(kv_dic.items(), key=lambda x: x[1], reverse=True)[0]

        if fea_name == "-99":  # 空值且占比超过80%
            ratio = fea_count / float(rows)
            if ratio > 0.8:
                nan_fea.append(c)
                continue

        if fea_count / float(rows) > 0.9:  # 某一取值超过90%，没有区分度
            same_val_fea.append(c)
            continue

        if len(kv_dic) == 1:  # 单一值特征剔除，没有区分度
            single_value_fea.append(c)
            continue

        if len(kv_dic) == rows:  # 可能是有用的特征，不过概率比较低，一般是索引等字段
            mutil_fea.append(c)

    print "单一值特征，空值特征，相同值特征，‘索引’特征的数量依次为", map(len, [single_value_fea, nan_fea, same_val_fea, mutil_fea])
    cols = list(set(cols) - set(same_val_fea))
    data = data[cols]
    return data, cols


def fill_none(data):
    # 空缺值统计
    # df_all.columns[df_all.isnull().sum() > 0]
    # 填补空缺值
    for column in data.columns:
        if column in base_fea:
            continue
        if data[data[column].isnull()].shape[0] > 0:
            mean_val = data[column].mean()
            if math.isnan(mean_val):
                print column,"no means value"
            data[column].fillna(mean_val, inplace=True)
    # df_all[df_all["FTR5_std"].isnull()]  # check
    return data


def bin_process(data):
    print "开始分箱。。。"
    cols = data.columns.tolist()
    bin_fea = []
    bin_dic = {}
    for col_name in cols:
        if col_name in base_fea:
            continue
        cut_nums, tmp_threshold = pd.qcut(data[col_name], 5, retbins=True, duplicates="drop", labels=False)  # 等频
        data[col_name] = cut_nums
        bin_fea.append(col_name)
        # 保存结果
        bin_dic[col_name] = tmp_threshold.tolist()
    return data, bin_dic


def woe_process(data):
    print "开始编码。。。"

    def woe(df, var, target):
        eps = 0.000001
        gbi = pd.crosstab(df[var], df[target]) + eps
        gb = df[target].value_counts() + eps
        gbri = gbi / gb
        gbri["woe"] = np.log(gbri[1] / gbri[0])
        return gbri["woe"].to_dict()

    woe_dic = {}
    cols = data.columns.tolist()
    for c in cols:
        if c in base_fea:
            continue
        tmp_woe_dic = woe(data, c, "LABEL")
        data[c] = [tmp_woe_dic[i] for i in data[c]]
        woe_dic[c] = tmp_woe_dic

    return data, woe_dic

def get_label(data):
    data_id = "./../data/train_id.tsv"
    dat_id = pd.read_csv(data_id, sep="\t")
    data = pd.merge(data, dat_id, how="inner", on="PERSONID")
    return  data

def process_fea():
    # 读入数据
    data = get_train_data()
    print "load data size: ", data.shape

    # test_data = get_test_data()

    # 特征处理
    cols = data.columns.tolist()
    print "data col name", cols
    # print dat.dtypes
    # 字符型：u'PERSONID', u'APPLYNO'，u'FTR51', u'CREATETIME'
    # 其他均为数值型

    # 分开数值特征和字符特征
    string_fea, numeric_fea = get_str_num_fea(data)

    # 数值型特征处理
    print "数值特征，", numeric_fea
    # 查看各字段的方差
    # data[numeric_fea].describe()

    # 没有空缺值
    # for c in numeric_fea:
    #     print c,data[data[numeric_fea[0]].isnull()].shape

    # 天花板地板处理，log处理（是否正态分布）
    data, ud_dic = bound_log_process(data, numeric_fea)
    save_dic("./../conf/ud_dic.dict", ud_dic)

    # 数值特征组合
    data = fea_combine(data, numeric_fea)
    print "col name after feature combine: ", data.columns.tolist()
    print "合并后数值特征的数据大小", data.shape

    # 字符特征组合
    # data = string_fea_combine(data,string_fea)

    # 特征过滤
    # data  中只有生成的数值特征和personid，没有其他变量
    data, select_fea = get_filter_fea(data)
    print "size after select fea: ", data.shape
    print "select fea: ", select_fea
    save_dic("./../conf/save_fea.dict", {"feature": select_fea})


    # 填补空缺值
    data = fill_none(data)

    # 分箱
    data, bin_dic = bin_process(data)
    save_dic("./../conf/cut_bin.dict", bin_dic)

    # woe 编码, 需要目标值
    # 提前merge label 值
    print "data shape without lable",data.shape
    data = get_label(data)
    print " data shape with label: ", data.shape

    data, woe_dic = woe_process(data)
    save_dic("./../conf/woe_encode.dict", woe_dic)
    print "after process size",data.shape

    # 处理完成，添加标签列，保存数据
    data.to_csv("./../output/model_data.csv",index=None)


# ## 模型训练
def train():
    df_all = pd.read_csv("./../output/model_data.csv")
    encode_fea = df_all.columns.tolist()
    encode_fea.remove("LABEL")
    encode_fea.remove("PERSONID")

    data_y = df_all["LABEL"]
    data_x = df_all[encode_fea]
    print "data_x size",data_x.shape

    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, random_state=2)

    # model = LogisticRegression(C=5.0,penalty='l1',solver='liblinear',multi_class='ovr')  # 0.778739753903  0.778527694177
    # model = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=16, min_samples_split=5,
    #                                min_samples_leaf=3)  # 0.774609330092  0.771532436717
    # model = GBT_model = GradientBoostingClassifier(n_estimators=30)

    dtrain = xgb.DMatrix(train_x, label=train_y)
    dtest = xgb.DMatrix(test_x)
    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'eval_metric': 'auc',
              'max_depth': 4,
              'lambda': 10,
              'subsample': 0.75,
              'colsample_bytree': 0.75,
              'min_child_weight': 2,
              'eta': 0.025,
              'seed': 0,
              'nthread': 8,
              'silent': 1}
    watchlist = [(dtrain, 'train')]
    model = xgb.train(params, dtrain, num_boost_round=100, evals=watchlist)
    y_pred = model.predict(dtest)
    print roc_auc_score(test_y, y_pred)



    # # model.fit(train_x, train_y)
    #
    # # y_prob = model.predict_proba(test_x)
    # # fpr, tpr, thresholds = roc_curve(test_y, y_prob[:, 1])
    #
    # roc_auc = auc(fpr, tpr)
    # plt.plot(fpr, tpr, lw=1, label='ROC  (area = %0.2f)' % roc_auc)
    #
    # plt.show()
    # print roc_auc_score(test_y, y_prob[:, 1])   # 0.862957042957043

    joblib.dump(model, "./../model/model")


if __name__ == "__main__":
    # process_fea()
    train()
