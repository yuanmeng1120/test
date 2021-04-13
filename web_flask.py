# from numpy import loadtxt
# from xgboost import XGBClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
# import numpy as np
# import xgboost as xgb

import pickle
import pandas as pd
import xgboost as xgb
# new_model = xgb.Booster(model_file='xgb.h5')  # 加载模型
#加载模型

new_model=pickle.load(open('bst.pkl','rb'))

app = Flask(__name__)


# @app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['GET', 'POST'])
def xg_predict():
    # if request.method == 'POST':
    #     get_input = request.get_json()['input']  # 从请求中获取数据
    #     get_input = xgb.DMatrix(get_input)  # 转换成xg的数据格式
    #     predicted = new_model.predict(get_input)  # 预测
    #     print(predicted)
    #     return jsonify(predicted)  # 将预测结果返回给请求者

    # if request.method == 'POST':
        # get_input = request.get_json()['input']  # 从请求中获取数据
        #读取数据：
        data = pd.read_excel('一行.xlsx')

        columns = ['fund_name', 'fund_type', 'user_id']
        data.drop(columns, axis=1, inplace=True)
        # 设置变量名：
        data = data.rename(
            columns={'title_id_68': 'title_id_00068_item_value', 'title_id_69': 'title_id_00069_item_value',
                     'title_id_70': 'title_id_00070_item_value', 'title_id_71': 'title_id_00071_item_value',
                     'title_id_72': 'title_id_00072_item_value', 'title_id_73': 'title_id_00073_item_value',
                     'title_id_74': 'title_id_00074_item_value',
                     'title_id_75': 'title_id_00075_item_value', 'title_id_76': 'title_id_00076_item_value',
                     'title_id_77': 'title_id_00077_item_value',
                     'title_id_00': 'title_id_00000_item_value', 'sex': 'gender', 'fund_type_id': 'fund_type',
                     'age': 'Age_Level',
                     'rate_rise_fall': 'floating_loss_rate/floating_profit_rate', })

        data['title_id_00000_item_value'].replace(['A', 'B'], [0, 1], inplace=True)
        data['Age_Level'].fillna(5, inplace=True)
        data['fund_first_purchase_day'].fillna(0, inplace=True)
        data['holding_return_rate'].fillna(0, inplace=True)
        data.fillna(1, inplace=True)

        dtest = xgb.DMatrix(data)
        # get_input = xgb.DMatrix(get_input)  # 转换成xg的数据格式
        predicted = new_model.predict(dtest)  # 预测
        print(predicted)
        # # return jsonify(predicted)  # 将预测结果返回给请求者
        # # 数据保存到原始表里
        # pred_result = pd.DataFrame(predicted, columns=['tolerance_level'])
        # result_data = pd.concat([data, pred_result], axis=1)
        # result_data.to_excel('result_data_1.xlsx')
        # return predicted
if __name__ == "__main__":
    # app.run(debug=True, host='0.0.0.0')
    app.run()
# import pickle
# import pandas as pd
# import xgboost as xgb
#
# # load data and data transform
# # data = pd.read_excel('基金赎回23日样本.xlsx')
# data = pd.read_excel('一行.xlsx')
#
# columns=['fund_name','fund_type','user_id']
# data.drop(columns,axis=1,inplace=True)
# #设置变量名：
# data=data.rename(columns={'title_id_68':'title_id_00068_item_value','title_id_69':'title_id_00069_item_value','title_id_70':'title_id_00070_item_value','title_id_71':'title_id_00071_item_value',
#                          'title_id_72':'title_id_00072_item_value','title_id_73':'title_id_00073_item_value','title_id_74':'title_id_00074_item_value',
#                          'title_id_75':'title_id_00075_item_value','title_id_76':'title_id_00076_item_value','title_id_77':'title_id_00077_item_value',
#                          'title_id_00':'title_id_00000_item_value','sex':'gender','fund_type_id':'fund_type','age':'Age_Level',
#                          'rate_rise_fall':'floating_loss_rate/floating_profit_rate',})
#
# data['title_id_00000_item_value'].replace(['A','B'],[0,1],inplace=True)
# data['Age_Level'].fillna(5,inplace=True)
# data['fund_first_purchase_day'].fillna(0,inplace=True)
# data['holding_return_rate'].fillna(0,inplace=True)
# data.fillna(1,inplace=True)
#
#
# dtest=xgb.DMatrix(data)
#
# # load model
# model_load=pickle.load(open('bst.pkl','rb'))
#
# # model predict
# bst_predict=model_load.predict(dtest)
# print(bst_predict)
# # 数据保存到原始表里
# pred_result=pd.DataFrame(bst_predict,columns=['tolerance_level'])
# result_data=pd.concat([data,pred_result],axis=1)
# result_data.to_excel('result_data.xlsx')