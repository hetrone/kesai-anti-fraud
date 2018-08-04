#!/usr/bin/env python
#-*-coding:utf-8-*-
#from hetroneModel.utils  import *
import copy	
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegressionCV, LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score,f1_score
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
import time
from sklearn import svm 
from xgboost.sklearn import XGBClassifier
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import hyperopt.pyll.stochastic

class BaseModelClass(object):
    def __init__(self):
        pass

    def get_ks(self,y_test,y_proba):
        ks_score = ks_2samp(y_proba[y_test==1], y_proba[y_test!=1]).statistic
        return ks_score

    def ModelPredict(self,xv, fmodel):
        if(type(fmodel) in [LogisticRegression, \
                 XGBClassifier, GradientBoostingClassifier,svm.SVC,RandomForestClassifier]):
            yvp = fmodel.predict_proba(xv)[:,1]
        else:
            yvp = fmodel.predict(xv).flatten()
        return(yvp) 

    def CrossPredict(self, xv, modelL):
        yvpL = []
        for i in range(len(modelL)):
            yvp = self.ModelPredict(xv, modelL[i])
            yvpL.append(yvp)
        return(yvpL)

    def Kfolds(self,x, k = 10, seed = 1):
        np.random.seed(seed)
        xL = np.array_split(np.random.choice(x, len(x), replace = False), k)
        return(xL)

    def GroupSelect(self,xL, i = 0):
        xLc = copy.copy(xL)
        ingrp = list(xLc.pop(i))  #测试
        exgrp = sum([list(x) for x in xLc], []) #训练
        return(ingrp, exgrp)
    
    def TrainSet(self, x, y, irtL, ig = 0):
        irt2, irt1 = self.GroupSelect(irtL, i = ig)
        xt1, xt2 = x.loc[irt1].values, x.loc[irt2].values
        yt1, yt2 = y.loc[irt1].values, y.loc[irt2].values
        return(xt1, xt2, yt1, yt2)
    
    def Score(self,y, yp, f = roc_auc_score):
        score = f(y, yp) 
        #print("Score: {:.4f}".format(score))
        return(score)
    
    def CrossTrain(self,x, y, irtL, fmodel, **kwargs):
        #print 'kwargs: ', kwargs
        modelL = []
        for i in range(len(irtL)):
            xt1, xt2, yt1, yt2 = self.TrainSet(x, y, irtL, ig = i)
            modelOne,auc_score = fmodel(xt1, xt2, yt1, yt2, seed = i, **kwargs) 
            #print 'modelOne: ',modelOne
            modelL.append(modelOne)
        return(modelL)

    def CrossValid(self,x, y, irtL, modelL):
        yt2pL = []
        #print 'modelL: ',modelL
        for i in range(len(irtL)):
            #print 'the i in the crossValid is: ',i
            xt1, xt2, yt1, yt2 = self.TrainSet(x, y, irtL, ig = i)
            yt2p = self.ModelPredict(xt2, modelL[i])
            yt2pL.append(yt2p)
        return(yt2pL)

    def ScoreWeight(self,score):
        w = np.exp((score - np.mean(score))/np.min(np.std(score, axis = 0)))
        w = np.sum(w/np.sum(w), axis = 0)
        return(w)

    def CrossScore(self,y, yt2pM, irtL, w = 1):
        score = []
        for i in range(len(irtL)):
            xt1, xt2, yt1, yt2 = self.TrainSet(y, y, irtL, ig = i)
            yt2L = [yt2pM[j][i] for j in range(len(yt2pM))]
            if(w is 1):
                score.append(list(map(lambda x: self.Score(yt2, x), yt2L)))
            else:
                score.append(self.Score(yt2, np.dot(np.array(yt2L).T , w)))
        return (np.array(score))

    def CrossScoreAnalysis(self,y, yt2pM, irtL, w = [], labels = None):
        scoreL = self.CrossScore(y, yt2pM, irtL)
        if(len(w) == 0):    
            w = self.ScoreWeight(scoreL)
        score = self.CrossScore(y, yt2pM, irtL, w)
        #ScorePlot(np.vstack([score, scoreL.T]).T, labels)
        return(scoreL, score, np.array(w))

    def SklearnClassifier(self,x_train, x_test, y_train, y_test, \
        model_type = LogisticRegression,seed = 0,parmodel = {}):
        '''集成sklearn算法方法
        ，可把sklearn的机器学习方法，提供选项集成

        Parameters
        ----------
        x_train : 训练数据集的自变量

        x_test : 测试数据集的自变量

        y_train : 训练数据集的依变量

        y_test : 测试数据集的依变量

        model_type : 默认LogisticRegression，
        目前支持[GradientBoostingClassifier,LogisticRegression,XGBClassifier]

        parmodel : 参数字典，不同方法对应不同参数字典，默认空缺
        '''
        timestart = time.time()
        par = {}    
        if model_type  in [RandomForestClassifier] : 
            par = {"random_state": seed, 'n_jobs': -1} 
        if model_type == svm.SVC :
            parmodel['probability'] = True  
        if 'max_depth' in parmodel.keys():
            parmodel["max_depth"] = int(parmodel["max_depth"])

        par.update(parmodel)
        model = model_type(**par)
        #model.fit(x_train, y_train.flatten())
        model.fit(x_train, y_train)
        score = self.Score(y_test, model.predict_proba(x_test)[:,1])
        #print("Time: {:.2f} seconds".format(time.time() - timestart))
        return(model,score)


    def DNN(self,xt1, xt2, yt1, yt2, seed = 0, parmodel = {}):
        np.random.seed(seed)
        timestart = time.time()
        par = {"nhidlayer": 2, "rdrop": 0.5, "nhidnode": 500, "outnode": 300,
               'optimizer':'sgd', "batch_size": 64, "earlystop": 3, "maxnorm": 4, "l2": 0}
        par.update(parmodel)

        layerin = Input(shape=(xt1.shape[1],))
        layer = layerin
        for i in range(par["nhidlayer"]):
            layer = Dense(par["nhidnode"], init = 'glorot_normal', \
                activation="relu", W_constraint = maxnorm(par["maxnorm"]))(layer)
            layer = BatchNormalization()(layer)
            layer = Dropout(par["rdrop"])(layer)
        layer = Dense(par["outnode"], init = 'glorot_normal', \
                activation="relu", W_constraint = maxnorm(par["maxnorm"]))(layer)
        layer = BatchNormalization()(layer)
        layer = Dropout(par["rdrop"])(layer)
        layerout = Dense(1, activation='sigmoid')(layer)
        model = Model(input=layerin, output=layerout)
        model.compile(loss='binary_crossentropy', optimizer=par['optimizer'])

        model.fit(xt1.astype("float32"), yt1.astype("float32"), nb_epoch=10, \
                batch_size=par["batch_size"], validation_data = (xt2, yt2),\
                callbacks = [EarlyStopping(monitor='val_loss', patience=par["earlystop"])])

        score = self.Score(yt2, self.ModelPredict(xt2, model))
        #print("Time: {:.2f} seconds".format(time.time() - timestart))
        return(model,score)


class HyperParModelClass(BaseModelClass):
    def __init__(self):
        '''超参数调节类
        '''
        pass
        

    def ParModelScore(self,par, df_X, df_y, index_row, fmodel, k = 5, **kwargs ):
        '''交叉验证

        Parameters
        ----------
        par : 模型参数

        df_X : dataframe格式的训练数据集自变量

        df_y : dataframe格式的训练数据集依变量

        index_row : df_X,df_y 的对应的行索引

        fmodel : [SklearnClassifier,DNN],当设定为SklearnClassifier时，必须设定model_type

        k : 分组数，默认为5

        Returns
        -------
        -score : 交叉验证AUC的负平均值

        '''
        index_list = self.Kfolds(index_row,k)
        model_list = self.CrossTrain(df_X, df_y, index_list, fmodel, parmodel = par,**kwargs)
        y_proba_list = self.CrossValid(df_X, df_y, index_list, model_list)
        score = np.mean(self.CrossScoreAnalysis(df_y, [y_proba_list], index_list)[0])
        return(-score)

    def HpOpt(self,df_X, df_y, index_row, space, fmodel, parainit={}, seed = 0, max_evals = 50 ,verbose = True,**kwargs):
        '''模型超参数搜索

        Parameters
        ----------

        df_X : dataframe格式的训练数据集自变量

        df_y : dataframe格式的训练数据集依变量

        index_row : df_X,df_y 的对应的行索引

        fmodel : [SklearnClassifier,DNN],当设定为SklearnClassifier时，必须设定model_type

        parainit : 已经调整好的参数，原则上不与space有交集

        max_evals : 最大搜索次数，默认50

        Returns
        -------
        op : 每次参数及平均AUC值

        '''
        def Obj(par):
            par.update(parainit)
            loss = self.ParModelScore(par, df_X, df_y, index_row, fmodel, **kwargs)
            if verbose:
                show = pd.DataFrame(par, index = [seed])
                show['auc_score'] = loss
                print('parameter and score:\n ', show)
            return({
                'loss': loss,
                'status': STATUS_OK,
                'loss_variance': 5e-5
                })
        np.random.seed(seed)
        trials = Trials()
        best = fmin(Obj,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials)
        op = pd.concat([pd.DataFrame([sum(list(trials.trials[i]["misc"]["vals"].values()), []) \
             for i in range(len(trials))], columns = list(trials.trials[0]["misc"]["vals"].keys())),\
               pd.DataFrame({"loss": trials.losses()})], axis = 1)
        return(op)


    def ModelWOpt(self,df_y, y_proba_matrix, index_row, seed = 0):
        '''模型组合，最佳权重组合参数搜索

        Parameters
        ----------
        df_y : 训练数据集的标签数据

        y_proba_matrix : 组合模型的交叉预测概率矩阵，格式：
        [[y_proba_model1_fold1,...,y_pro_model1_foldk],
        [y_proba_model2_fold1,...,y_pro_model2_foldk],...]

        index_row : df_X,df_y 的对应的行索引

        Returns
        -------
        op : 不同超参数下的结果

        '''

        len_ypm = len(y_proba_matrix) 
        iw = ["w_{}".format(2+i) for i in range( len_ypm -1)]
        space = dict(zip(*[iw, [hp.uniform(iw[i], 0, 1.0/len_ypm) for i in range(len_ypm -1)]]))
        def Obj(w):
            wtmp = [1-np.sum(list(w.values()))]+list(w.values())
            return({
                'loss': -np.mean(self.CrossScore(df_y, y_proba_matrix, index_row, wtmp)),
                'status': STATUS_OK
                })
        np.random.seed(seed)
        trials = Trials()
        fmin(Obj,space=space,algo=tpe.suggest,max_evals=10,trials=trials)
        op = pd.concat([pd.DataFrame({"loss": trials.losses()}),\
             pd.DataFrame([sum(list(trials.trials[i]["misc"]["vals"].values()), [])\
             for i in range(len(trials))],columns = \
             list(trials.trials[0]["misc"]["vals"].keys()))], axis = 1)
        return(op)
    
    def ModelMPredict(self,df_X, modeldir_list, w_list ):
        '''对测试数据及线上数据预测

        Parameters
        ----------
        df_X : datafram格式的需要预测的数据

        modeldir_list : 模型路径列表

        w_list : 与模型对应的权重

        Returns
        -------
        y_proba : 预测概率 

        '''
        if df_X.ndim == 1:
            df_X = df_X.reshape(1,-1)
        yvpM = []
        for modeldir in modeldir_list:
            modelL = joblib.load(modeldir)
            yvpM.append(self.CrossPredict(df_X,modelL)) 
        if sum(w_list.values())!= 1:
            print('please set the sum of weight to 1.0!')
        y_proba_mean = np.array([np.mean(yvpM[i], axis = 0) for i in range(len(yvpM))])
        y_proba = y_proba_mean.T.dot(w_list.values())
        return (y_proba)


