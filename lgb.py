# -*- coding: UTF-8 -*-
import lightgbm as lgb
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# 加载数据
def load_data():

    white_train = pd.read_csv('../../data/white_train_feature_plus_label.csv')
    black_train = pd.read_csv('../../data/black_train_feature_plus_label.csv')

    frames = [white_train, black_train]
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
def customed_score(preds, label):
    pred = [int(i >= 0.5) for i in preds]
    confusion_matrixs = confusion_matrix(label, pred)
    recall = float(confusion_matrixs[1][1]) / float(confusion_matrixs[1][1]+confusion_matrixs[1][0])
    precision = float(confusion_matrixs[1][1]) / float(confusion_matrixs[1][1]+confusion_matrixs[0][1])
    F = -5*precision*recall/(2*precision+3*recall)*100
    return float(F)


if __name__ == '__main__':

    train_data, label = load_data()

    train_x, test_x, train_y, test_y = train_test_split(train_data, label, test_size=0.01, random_state=0)

    lgb_train = lgb.Dataset(train_x, train_y)
    lgb_eval = lgb.Dataset(test_x, test_y, reference=lgb_train)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': ['binary_logloss', 'auc'],
        'num_leaves': 12,
        'learning_rate': 0.01,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }

    print('Start training...')
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=400,
                    valid_sets=[lgb_train, lgb_eval],
                    verbose_eval=False)

    print(gbm.best_iteration)
    print('Save model...')
    gbm.save_model('../../lgb.model')

    print('Start test auc...')
    pred_y = gbm.predict(train_data)
    print('The auc of prediction is:', customed_score(pred_y, label))

    print('Load test data...')
    predict_sample = pd.read_csv('../../data/predict_feature_plus.csv')
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

    print('Start predicting...')
    y = gbm.predict(predict_sample, num_iteration=gbm.best_iteration)
    predict_id['prob'] = y
    res = predict_id.sort_values(by='prob')
    ress = res[res['prob'] < 0.5]
    ress = ress.sort_values(by='id')
    ress.id.to_csv('../../submission_lgb.txt', index=False, index_label=False)

    res.iloc[0:20000].to_csv('../../submission_lgb.txt', header=None, index=False)

    # ress = pd.read_csv('../../submission_lgb.txt', names=['id'])
    # ress = ress.sort_values(by='id')
    # ress.id.to_csv('../../submission_lgb.txt', header=None, index=False)


    feat_imp = pd.Series(gbm.feature_importance()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()
