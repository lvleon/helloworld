# -*- coding: UTF-8 -*-
import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
from xgboost.sklearn import XGBClassifier
import matplotlib.pyplot as plt
import numpy as np


# 加载数据
def load_data():
    white_sample = pd.read_csv('../../data/white_train_feature_label.csv')
    black_sample = pd.read_csv('../../data/black_train_feature_label.csv')

    # white_train = white_sample.sample(n=500, random_state=np.random.seed())
    # black_train = black_sample.sample(n=350, random_state=np.random.seed())

    frames = [white_sample, black_sample]
    train_data = pd.DataFrame(pd.concat(frames))
    label = train_data['label'].copy()

    del train_data['id']
    del train_data['label']
    # del train_data['unrepeated_x_count']
    # del train_data['x_count_ratio']
    # del train_data['unrepeated_y_count']
    # del train_data['y_count_ratio']
    # del train_data['unrepeated_xy_count']
    # del train_data['xy_count_ratio']
    # del train_data['sample_time']

    return train_data, label


# 自定义评价函数
def customed_score(preds, dtrain):
    label = dtrain.get_label()
    pred = [int(i >= 0.5) for i in preds]
    confusion_matrixs = confusion_matrix(label, pred)
    recall = float(confusion_matrixs[1][1]) / float(confusion_matrixs[1][1]+confusion_matrixs[1][0])
    precision = float(confusion_matrixs[1][1]) / float(confusion_matrixs[1][1]+confusion_matrixs[0][1])
    F = -5*precision*recall/(2*precision+3*recall)
    return 'FSCORE', float(F)


# 交叉验证，返回训练好的模型
def xgboost_cv(alg, train_data, label):

    dtrain = xgb.DMatrix(train_data.values, label=label.values)
    param = alg.get_xgb_params()
    cvresult = xgb.cv(param, dtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=5,
                      feval=customed_score, early_stopping_rounds=50, show_progress=False)
    print(cvresult)
    print('树的个数为：%d'%(cvresult.shape[0]))
    alg.set_params(n_estimators=cvresult.shape[0])

    alg.fit(train_data, label, eval_metric=customed_score)

    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    alg.booster().save_model('../../xgb.model')
    plt.show()
    return alg


# 使用训练好的模型进行预测
def predict(alg):
    predict_sample = pd.read_csv('../../data/predict_feature.csv')
    predict_id = predict_sample['id'].copy()
    predict_id = pd.DataFrame(predict_id)
    del predict_sample['id']
    # del predict_sample['unrepeated_x_count']
    # del predict_sample['x_count_ratio']
    # del predict_sample['unrepeated_y_count']
    # del predict_sample['y_count_ratio']
    # del predict_sample['unrepeated_xy_count']
    # del predict_sample['xy_count_ratio']
    # del predict_sample['sample_time']
    y = alg.predict_proba(predict_sample)
    predict_id['prob'] = y[:, 1]
    res = predict_id.sort_values(by='prob')
    res.iloc[0:20000].id.to_csv('../../submission.txt', header=None, index=False)


    # ress = res[res['prob'] < 0.5]
    # ress = ress.sort_values(by='id')
    # ress.id.to_csv('../../submission.txt', index=False, index_label=False)

    ress = pd.read_csv('../../submission.txt', names=['id'])
    ress = ress.sort_values(by='id')
    ress.id.to_csv('../../submission.txt', header=None, index=False)


# 主函数
def run():
    xgb1 = XGBClassifier(
        learning_rate=0.01,
        n_estimators=1000,
        max_depth=6,
        min_child_weight=1,
        gamma=0,
        reg_lambda=1,
        subsample=0.78,
        colsample_bytree=0.52,
        objective='binary:logistic',
        nthread=8,
        scale_pos_weight=1,
        seed=0)

    train_data, label = load_data()
    xgb1 = xgboost_cv(xgb1, train_data, label)
    predict(xgb1)

if __name__ == '__main__':
    run()