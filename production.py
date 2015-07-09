# -*- coding: utf-8 -*-
import os, sys
import csv
import datetime
sys.path.append(os.path.join("./"))
#sys.path.insert(0,os.getcwd()+"/lib/")
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from return_duration_lib import ReturnDuration
# load the path of module
class Production:
    def __init__(self):
        self.aud_model = joblib.load('./components/rfr_aud.pkl')
        self.aud_enc   = joblib.load('./components/enc_aud.pkl')
        self.pro_model = joblib.load('./components/rfr_production.pkl')
        self.pro_enc   = joblib.load('./components/enc_production.pkl')
        self.rd        = ReturnDuration()

    def save_to_file(self, path, data, delimiter = ','):
        df = pd.DataFrame(data, columns=['result'])
        df.to_csv(path, index=False)
        #with open(path, 'wb') as csvfile:
            #writer = csv.writer(csvfile, delimiter=delimiter)
            #writer.writerow(data)
            ##writer.writerow([e.get_value() for e in data])
            ##for e in data:
                ##print e
                ##writer.writerow([r.get_value() for r in result_list])
        ##np.savetxt(path, data, delimiter)

    def generate_save_result(self, df, path, model):
        result = self.rd.RFR_predict(model, np.array(df))
        result = np.round(result, 2)
        self.save_to_file(path, result)
        return result

#production part
    def production_prediction(self, input_path='./data/to_predict_data.csv', output_path='./data/tmp.csv', sep=','):
        enc_array     = ['ReturnCompanyID','StockNo']
        un_enc_array  = ['FactNum']
        df, enc_model = self.rd.load_data(input_path,enc_array, un_enc_array, self.pro_enc)
        result = self.generate_save_result(df, output_path, model=self.pro_model)
        return result

#audit part
    def audit_prediction(self, input_path='./data/to_predict_data.csv', output_path='./data/tmp.csv',sep =','):
        enc_array_aud = ['FirDeptID']
        df, enc_model = self.rd.load_data(input_path, enc_array_aud, enc_model=self.aud_enc, sep='\t')
        result = self.generate_save_result(df, output_path, model=self.aud_model)
        return result


p = Production()

if len(sys.argv)==2:
    to_predict_data = './data/' + sys.argv[1]
    print "read to predict data from "+ to_predict_data
    aud_result = p.audit_prediction(input_path=to_predict_data, output_path='./data/prediction_result/pro'+p.rd.start_time+'.csv', sep ='\t')
    pro_result = p.production_prediction(input_path=to_predict_data, output_path = './data/prediction_result/aud'+p.rd.start_time+'.csv', sep ='\t')
    print 1111
    print aud_result

else:
    aud_result = p.audit_prediction(output_path = './data/prediction_result/pro'+p.rd.start_time+'.csv')
    pro_result = p.production_prediction(output_path = './data/prediction_result/aud'+p.rd.start_time+'.csv')

total_result = aud_result + pro_result
total_result = np.round(total_result)
p.save_to_file('./data/prediction_result/total' + p.rd.start_time+'.csv', total_result)
