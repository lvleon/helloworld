# -*- coding: UTF-8 -*-
import pandas as pd
import scipy as sp
import numpy as np
import sklearn
import gc
import warnings
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix
from scipy import spatial

warnings.filterwarnings("ignore")

cache = 'cache'
sub = 'sub'
datadir = 'data'

train_path = os.path.join(datadir, 'dsjtzs_txfz_training.txt')
test_path = os.path.join(datadir, 'dsjtzs_txfz_test1.txt')

if not os.path.exists(cache):
    os.mkdir(cache)
if not os.path.exists(sub):
    os.mkdir(sub)


# 自定义评价函数
def customed_score(preds, label):
    pred = [int(i >= 0.5) for i in preds]
    confusion_matrixs = confusion_matrix(label, pred)
    recall = float(confusion_matrixs[1][1]) / float(confusion_matrixs[1][1]+confusion_matrixs[1][0])
    precision = float(confusion_matrixs[1][1]) / float(confusion_matrixs[1][1]+confusion_matrixs[0][1])
    F = -5*precision*recall/(2*precision+3*recall)*100
    return float(F)


def applyParallel(dfGrouped, func):
    with Parallel(n_jobs=8) as parallel:
        retLst = parallel(delayed(func)(pd.Series(value)) for key, value in dfGrouped)
        return pd.concat(retLst, axis=0)


def draw(df):
    import matplotlib.pyplot as plt
    if not os.path.exists('pic'):
        os.mkdir('pic')

    points = []
    for point in df.trajectory[:-1].split(';'):
        point = point.split(',')
        points.append((float(point[0]) / 7, float(point[1]) / 13))

    x, y = zip(*points)
    plt.figure(figsize=(16, 5))
    plt.subplot(121)
    plt.plot(x, y)
    plt.subplot(122)
    plt.plot(x, y)
    aim = df.aim.split(',')
    aim = (float(aim[0]) / 7, float(aim[1]) / 13)
    plt.scatter(aim[0], aim[1])
    plt.title(df.label)
    plt.savefig('pic/%s-label=%s' % (df.idx, df.label))
    plt.clf()
    plt.close()


def get_feature(df):
    points = []

    for point in df.trajectory[:-1].split(';'):
        point = point.split(',')
        points.append(((float(point[0]), float(point[1])), float(point[2])))

    xs = pd.Series([point[0][0] for point in points])
    ys = pd.Series([point[0][1] for point in points])

    aim = df.aim.split(',')
    aim = (float(aim[0]), float(aim[1]))

    distance_deltas = pd.Series(
        [sp.spatial.distance.euclidean(points[i][0], points[i+1][0]) for i in range(len(points) - 1)])

    time_deltas = pd.Series([points[i+1][1] - points[i][1] for i in range(len(points) - 1)])
    xs_deltas = xs.diff(1)
    ys_deltas = ys.diff(1)

    speeds = pd.Series(
        [np.log1p(distance) - np.log1p(delta) for (distance, delta) in zip(distance_deltas, time_deltas)])
    angles = pd.Series(
        [np.log1p((points[i+1][0][1] - points[i][0][1])) - np.log1p((points[i+1][0][0] - points[i][0][0])) for i in
         range(len(points) - 1)])

    speed_diff = speeds.diff(1).dropna()
    angle_diff = angles.diff(1).dropna()

    distance_aim_deltas = pd.Series([sp.spatial.distance.euclidean(points[i][0], aim) for i in range(len(points))])
    distance_aim_deltas_diff = distance_aim_deltas.diff(1).dropna()

    df['speed_diff_median'] = speed_diff.median() # 1
    df['speed_diff_mean'] = speed_diff.mean() # 2
    df['speed_diff_var'] = speed_diff.var() # 3
    df['speed_diff_max'] = speed_diff.max() # 4
    df['angle_diff_var'] = angle_diff.var() # 5

    df['time_delta_min'] = time_deltas.min() # 6
    df['time_delta_max'] = time_deltas.max() # 7
    df['time_delta_var'] = time_deltas.var() # 8

    df['distance_deltas_max'] = distance_deltas.max() # 9
    df['distance_deltas_0_count'] = len(distance_deltas[distance_deltas == 0]) # 10

    df['aim_distance_last'] = distance_aim_deltas.values[-1] # 11
    df['aim_distance_diff_max'] = distance_aim_deltas_diff.max() # 12
    df['aim_distance_diff_var'] = distance_aim_deltas_diff.var() # 13

    df['mean_speed'] = speeds.mean() # 14
    df['median_speed'] = speeds.median() # 15
    df['var_speed'] = speeds.var() # 16

    df['max_angle'] = angles.max() # 17
    df['var_angle'] = angles.var() # 18
    df['kurt_angle'] = angles.kurt() # 19

    df['y_min'] = ys.min() # 20
    df['y_max'] = ys.max() # 21
    df['y_var'] = ys.var() # 22

    df['x_min'] = xs.min() # 23
    df['x_max'] = xs.max() # 24
    df['x_var'] = xs.var() # 25

    df['x_init'] = xs.values[0] # 26
    df['y_init'] = ys.values[0] # 27

    df['x_back_num'] = min((xs.diff(1).dropna() > 0).sum(), (xs.diff(1).dropna() < 0).sum()) # 28
    df['y_back_num'] = min((ys.diff(1).dropna() > 0).sum(), (ys.diff(1).dropna() < 0).sum()) # 29

    df['xs_delta_var'] = xs_deltas.var() # 30
    df['xs_delta_max'] = xs_deltas.max() # 31
    df['xs_delta_min'] = xs_deltas.min() # 32
    df['time_deltas_0_count'] = len(time_deltas[time_deltas == 0]) # 33

    return df.to_frame().T


def get_single_feature(df):
    points = []

    for point in df.trajectory[:-1].split(';'):
        point = point.split(',')
        points.append(((float(point[0]), float(point[1])), float(point[2])))

    aim = df.aim.split(',')
    aim = (float(aim[0]), float(aim[1]))

    aim_angle = pd.Series([np.log1p(point[0][1] - aim[1]) - np.log1p(point[0][0] - aim[0]) for point in points])
    aim_angle_diff = aim_angle.diff(1).dropna()

    df['aim_angle_last'] = aim_angle.values[-1]
    df['aim_angle_diff_max'] = aim_angle_diff.max()
    df['aim_angle_diff_var'] = aim_angle_diff.var()

    if len(aim_angle_diff) > 0:
        df['aim_angle_diff_last'] = aim_angle_diff.values[-1]
    else:
        df['aim_angle_diff_last'] = -1
    return df.to_frame().T


def make_train_set():
    dump_path = os.path.join(cache, 'train.hdf')
    if os.path.exists(dump_path):
        train = pd.read_hdf(dump_path, 'all')
    else:
        train = pd.read_csv('../../data/raw_data/dsjtzs_txfz_training.txt', sep=' ', header=None, names=['id', 'trajectory', 'aim', 'label'])
        train['count'] = train.trajectory.map(lambda x: len(x.split(';')))
        train = applyParallel(train.iterrows(), get_feature).sort_values(by='id')
        train.to_hdf(dump_path, 'all')
    return train


def make_test_set():
    dump_path = os.path.join(cache, 'test.hdf')
    if os.path.exists(dump_path):
        test = pd.read_hdf(dump_path, 'all')
    else:
        test = pd.read_csv('../../data/raw_data/dsjtzs_txfz_test1.txt', sep=' ', header=None, names=['id', 'trajectory', 'aim'])
        test['count'] = test.trajectory.map(lambda x: len(x.split(';'))) # 0
        test = applyParallel(test.iterrows(), get_feature).sort_values(by='id')
        test.to_hdf(dump_path, 'all')
    return test


if __name__ == '__main__':
    draw_if = False
    train, test = make_train_set(), make_test_set()
    if draw_if:
        train.reset_index().rename(columns={'index': 'idx'}).apply(draw, axis=1)

    training_data, label = train.drop(['id', 'trajectory', 'aim', 'label'], axis=1).astype(float), train['label']
    sub_training_data, instanceIDs = test.drop(['id', 'trajectory', 'aim'], axis=1).astype(float), test['id']
    print (training_data.shape)

    train_x, test_x, train_y, test_y = train_test_split(training_data, label, test_size=0.01, random_state=0)

    # lgb
    # create dataset for lightgbm
    lgb_train = lgb.Dataset(train_x, train_y)
    lgb_eval = lgb.Dataset(test_x, test_y, reference=lgb_train)

    print (train_x.shape)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': ['binary_logloss', 'auc'],
        'num_leaves': 7,
        'learning_rate': 0.05,
        'feature_fraction': 0.83,
        'bagging_fraction': 0.85,
        'bagging_freq': 5,
        'verbose': 0
    }

    print('Start training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=280,
                    valid_sets=[lgb_train, lgb_eval],
                    verbose_eval=False)
    print(gbm.best_iteration)


    print('Start test auc...')
    pred_y = gbm.predict(training_data)

    print('The auc of prediction is:', customed_score(pred_y, label.astype(int)))


    y = gbm.predict(sub_training_data)
    res = instanceIDs.to_frame()
    res['prob'] = y
    res['id'] = res['id'].astype(int)
    res = res.sort_values(by='prob')
    res.iloc[0:20000].to_csv(os.path.join(sub, 'BDC20160706.txt'), header=None, index=False)
    feat_imp = pd.Series(gbm.feature_importance()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()