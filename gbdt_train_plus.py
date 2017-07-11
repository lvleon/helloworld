# -*- coding: UTF-8 -*-
import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
from xgboost.sklearn import XGBClassifier
import matplotlib.pyplot as plt
import numpy as np
from xgboost import plot_tree
from xgboost import plot_importance
# from graphviz import Digraph
# import pydot
# 加载数据
def load_data():
    white_sample = pd.read_csv('../../data/white_train_feature_plus_label.csv')
    black_sample = pd.read_csv('../../data/black_train_feature_plus_label.csv')

    # white_train = white_sample.sample(n=500, random_state=np.random.seed())
    # black_train = black_sample.sample(n=350, random_state=np.random.seed())

    black_speed_var_feature_file = pd.read_csv('../../data/added_feature/black_speed_var_feature.csv')

    white_speed_var_feature_file = pd.read_csv('../../data/added_feature/white_speed_var_feature.csv')

    white_sample = pd.merge(white_sample, white_speed_var_feature_file, how='left', on='id')
    black_sample = pd.merge(black_sample, black_speed_var_feature_file, how='left', on='id')

    # white_sample.to_csv('../../data/white_train_feature_plus_label.csv', index=False)
    # black_sample.to_csv('../../data/black_train_feature_plus_label.csv', index=False)

    frames = [white_sample, black_sample]
    train_data = pd.DataFrame(pd.concat(frames))
    # train_data = train_data.sample(len(train_data['id']))
    label = train_data['label'].copy()

    # del train_data['']
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

    dtrain = xgb.DMatrix(train_data.values, label=label.values, missing=np.NAN)
    param = alg.get_xgb_params()
    # cvresult = xgb.cv(param, dtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=5,
    #                   feval=customed_score, early_stopping_rounds=50, show_progress=False)
    cvresult = xgb.cv(param, dtrain, num_boost_round=alg.get_params()['n_estimators'], metrics=['auc'], nfold=5, early_stopping_rounds=50, show_progress=False)
    print(cvresult)
    print('树的个数为：%d'%(cvresult.shape[0]))
    alg.set_params(n_estimators=cvresult.shape[0])

    # alg.fit(train_data, label, eval_metric=customed_score)
    alg.fit(train_data, label, eval_metric='auc')

    # feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    # feat_imp.plot(kind='bar', title='Feature Importances')
    # plt.ylabel('Feature Importance Score')
    alg.booster().save_model('../../xgb_plus.model')
    # plot_tree(alg)
    plot_importance(alg)
    plt.show()
    return alg


# 使用训练好的模型进行预测
def predict(alg):
    predict_sample = pd.read_csv('../../data/predict_feature_plus.csv')
    predict_id = predict_sample['id'].copy()
    predict_id = pd.DataFrame(predict_id)
    predict_speed_var_feature_file = pd.read_csv('../../data/added_feature/predict_speed_var_feature.csv')

    predict_sample = pd.merge(predict_sample, predict_speed_var_feature_file, how='left', on='id')

    del predict_sample['id']
    predict_sample = pd.DataFrame(predict_sample)
    # del predict_sample['']
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

    res.iloc[0:20000].id.to_csv('../../submission_plus.txt', header=None, index=False)

    # ress = res[res['prob'] < 0.5]
    # ress = ress.sort_values(by='id')
    # ress.id.to_csv('../../submission_plus.txt', index=False, index_label=False)

    ress = pd.read_csv('../../submission_plus.txt', names=['id'])
    ress = ress.sort_values(by='id')
    ress.id.to_csv('../../submission_plus.txt', header=None, index=False)


# 主函数
def run():
    xgb1 = XGBClassifier(
        learning_rate=0.01,
        n_estimators=1000,
        max_depth=6,
        min_child_weight=1,
        gamma=0,
        reg_lambda=1,
        subsample=0.84,
        colsample_bytree=0.75,
        objective='binary:logistic',
        nthread=8,
        scale_pos_weight=1,
        seed=0)

    train_data, label = load_data()
    xgb1 = xgboost_cv(xgb1, train_data, label)
    predict(xgb1)

if __name__ == '__main__':
    run()

'''
    xgb1 = XGBClassifier(
        learning_rate=0.01,
        n_estimators=115,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        reg_lambda=1,
        subsample=0.6,
        colsample_bytree=0.4,
        objective='binary:logistic',
        nthread=8,
        scale_pos_weight=1,
        seed=0)
    这个模型全学对了， -1.000000         0.000000          -1.000000          0.000000
    预测负样本14065个,提交前20000，线上得分91.44




'''