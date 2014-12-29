# -*- coding: utf-8 -*-
import os, sys
import datetime
import numpy as np
from sklearn.externals import joblib
sys.path.append(os.path.join("./"))
from return_duration_lib import ReturnDuration
# load the path of module
aud_model = joblib.load('./components/rfr_aud.pkl')
aud_enc   = joblib.load('./components/enc_aud.pkl')
pro_model = joblib.load('./components/rfr_production.pkl')
pro_enc   = joblib.load('./components/enc_production.pkl')
#production part
rd = ReturnDuration()
enc_array     = ['ReturnCompanyID','StockNo']
un_enc_array  = ['FactNum']
df, enc_model = rd.load_data('./data/to_predict_data.csv',enc_array, un_enc_array, pro_enc)
pro_result    = rd.RFR_predict(pro_model, np.array(df))
#audit part
enc_array_aud = ['FirDeptID']
df, enc_model = rd.load_data('./data/to_predict_data.csv',enc_array_aud, enc_model=aud_enc)
aud_result    = rd.RFR_predict(aud_model, np.array(df))
# write to file
np.savetxt('./data/prediction_result/pro'+rd.start_time+'.csv', pro_result, delimiter=',' )
np.savetxt('./data/prediction_result/aud'+rd.start_time+'.csv', aud_result, delimiter=',')
