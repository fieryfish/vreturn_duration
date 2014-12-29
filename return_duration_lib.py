# -*- coding: utf-8 -*-
from collections import Counter
import datetime
import numpy as np
import pandas as pd
from scipy.stats import mstats
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor

# This class is to calculate the Return duration which consist of audit and
# production in each inventory.
# This model use the RandomForestRegressor to train the model. For the
# categorical feature, we use OneHotEncoder techique to do the preprocessing.
# For the audit part, we only the the FirDeptID field to train the model which
# is not so precise because of the low R square value. But the audit duration is not
# quite long, so it doesn't affect too much
# For the production part, we use 'ReturnCompanyID','StockNo' and 'FactNum' to
# train the model which means '配送中心','库房' and '实际退货数量'.
class ReturnDuration:

    def __init__(self):
        self.start_time = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M')

    def load_raw_data(self, path="./data/training.csv", sep=','):
        return pd.read_csv(path, sep=sep)

    # reconstruct the training data on OneHotEncoder
    def load_enc_data(self, path, enc_array, enc_model=None):
        df = self.load_raw_data(path)
        enc_arr = []
        for e in enc_array:
            enc_arr.append(df[e])

        categorical_df  = pd.concat(enc_arr,axis=1)
        categorical_arr = np.array(categorical_df)
        if enc_model is None:
            enc_model   = OneHotEncoder().fit(categorical_arr)
            enc_model.transform(categorical_arr).toarray()
        else:
            enc_model.transform(categorical_arr).toarray()

        return [categorical_arr, df, enc_model]

    def load_data(self, path, enc_array=[], un_enc_array=[], enc_model=None):
        categorical_arr,df,enc_model = self.load_enc_data(path, enc_array, enc_model)
        preprocessed_data            = pd.DataFrame(categorical_arr)
        total_training_data          = [preprocessed_data]
        for e in un_enc_array:
            total_training_data.append(df[e])

        preprocessed_data = pd.concat(total_training_data,axis=1)
        return [preprocessed_data, enc_model]

    def randomize(self, df):
        return df.reindex(np.random.permutation(df.index))

    # remove the data that above the 0.9 quantile which is the outliers
    def slice_09_quantile(self, df, time_field = 'time_diff_produce'):
        mask = []

        if time_field == 'time_diff_aud':
            produce_quantiles    = mstats.mquantiles(df[time_field], prob=[0.1,0.5,0.9])
            produce_higher_bound = produce_quantiles[2]
            for i in range(len(df)):
                if (df[time_field][i] <= produce_higher_bound):
                    mask.append(True)
                else:
                    mask.append(False)
        else:
            produce_quantiles    = mstats.mquantiles(df[time_field], prob=[0.1,0.5,0.9])
            produce_higher_bound = produce_quantiles[2]
            for i in range(len(df)):
                if (df[time_field][i] <= produce_higher_bound):
                    mask.append(True)
                else:
                    mask.append(False)

        df = df[mask]
        return df

    def RFR_develeopment(self, train_df, test_df, n_estimators=100):
        rfr = RandomForestRegressor(n_estimators=n_estimators)

        model = rfr.fit(train_df.iloc[:,:-1], train_df.iloc[:,-1])
        r2 = rfr.score(train_df.iloc[:,:-1], train_df.iloc[:,-1])
        error_square = 0.0

        for i, e in test_df.iterrows():
            error_square += (model.predict(np.array(e[:-1]))[0] - e[-1])**2

        mse = error_square/float(len(test_df))
        return [r2,rfr ,mse]

    def RFR_train(self, train_df,n_estimators=100):
        rfr = RandomForestRegressor(n_estimators=n_estimators)
        model = rfr.fit(train_df.iloc[:,:-1], train_df.iloc[:,-1])
        return model

    def RFR_predict(self, model, test_data):
        return model.predict(test_data)


    def output_result(self, content_list, path=""):
        if path == "":
            path = './data/result_'+self.start_time+'.txt'
        FILE = open(path,"w")
        for item in content_list:
            FILE.write("%s\n" % item)
        FILE.close()

