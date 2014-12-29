# -*- coding: utf-8 -*-
import os, sys
import datetime
from sklearn.externals import joblib
# load the path of module
sys.path.append(os.path.join("./"))
from return_duration_lib import ReturnDuration

def cal_write_result(rd, train_df, test_df):
    R_square, rfr, mse = rd.RFR_develeopment(train_df,test_df)
    end_time = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M')
    write_out_content = ["start_time:",rd.start_time, "end_time:",end_time,
                        "mse:",mse, "import_features:",rfr.feature_importances_,"R_square:", R_square]
    return write_out_content

def generate_train_test(df, rd, time_field='time_diff_produce'):
    df = rd.slice_09_quantile(df,time_field)
    df = rd.randomize(df)
    train_num = int(len(df) * 0.9) # use 90% of instance for training, 10% to test
    train_df = df[:train_num]
    test_df = df[train_num+1:]
    return [train_df, test_df, df]

#生产时长
def production_time(input_file='./data/tmp.csv'):
# init
    rd = ReturnDuration()
# features that use OneHotEncoder to do preprocessing
    enc_array     = ['ReturnCompanyID','StockNo']
# features that don't use OneHotEncoder
    un_enc_array  = ['FactNum','time_diff_produce']

    df, enc_model = rd.load_data(input_file,enc_array, un_enc_array)
    train_df, test_df, df = generate_train_test(df,rd)

    write_out_content = cal_write_result(rd, train_df, test_df)
    rd.output_result(write_out_content, path = './data/training_result/result_produce'+rd.start_time+'.txt')
    rfr_model = rd.RFR_train(df)
# make sure you have the components dir first
    joblib.dump(enc_model, './components/enc_production.pkl')
    joblib.dump(rfr_model, './components/rfr_production.pkl')


def audit_time(input_file='./data/tmp.csv'):
    rd = ReturnDuration()
    enc_array     = ['FirDeptID']
    un_enc_array  = ['time_diff_aud']
    df, enc_model = rd.load_data(input_file,enc_array, un_enc_array)

    train_df, test_df, df = generate_train_test(df, rd, 'time_diff_aud')
    write_out_content = cal_write_result(rd, train_df, test_df)
    rd.output_result(write_out_content, path = './data/training_result/result_audit'+rd.start_time+'.txt')
    rfr_model = rd.RFR_train(df)
    joblib.dump(enc_model, './components/enc_aud.pkl')
    joblib.dump(rfr_model, './components/rfr_aud.pkl')

def run(input_file=""):
    production_time(input_file)
    audit_time(input_file)
#run("./data/tmp.csv")
if len(sys.argv)==2:
    training_file_path = './data/' + sys.argv[1]
    print "read training data from "+ training_file_path
    run(training_file_path)
else:
    print "read training data from ./data/training_data.csv"
    run("./data/training_data.csv")
