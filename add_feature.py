# -*- coding: UTF-8 -*-
import pandas as pd
import scipy as sp
import numpy as np
from joblib import Parallel, delayed
import warnings
from scipy import spatial
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

def applyParallel(dfGrouped, func):
    with Parallel(n_jobs=8) as parallel:
        retLst = parallel(delayed(func)(pd.Series(value)) for key, value in dfGrouped)
        return pd.concat(retLst, axis=0)


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
        [sp.spatial.distance.euclidean(points[i][0], points[i + 1][0]) for i in range(len(points) - 1)])

    time_deltas = pd.Series([points[i + 1][1] - points[i][1] for i in range(len(points) - 1)])

    speeds = pd.Series(
        [np.log1p(distance) - np.log1p(delta) for (distance, delta) in zip(distance_deltas, time_deltas)])

    angles = pd.Series(
        [np.log1p((points[i + 1][0][1] - points[i][0][1])) - np.log1p((points[i + 1][0][0] - points[i][0][0])) for i in
         range(len(points) - 1)])

    # xy速度的一阶差分
    speed_diff = speeds.diff(1).dropna()
    angle_diff = angles.diff(1).dropna()

    distance_aim_deltas = pd.Series([sp.spatial.distance.euclidean(points[i][0], aim) for i in range(len(points))])
    distance_aim_deltas_diff = distance_aim_deltas.diff(1).dropna()

    df['xy_speed_diff_median'] = speed_diff.median()
    df['xy_speed_diff_mean'] = speed_diff.mean()
    df['xy_speed_diff_var'] = speed_diff.var()
    df['xy_speed_diff_max'] = speed_diff.max()
    df['xy_speed_diff_kurt'] = speed_diff.kurt()
    df['xy_speed_diff_skew'] = speed_diff.skew()

    df['time_delta_min'] = time_deltas.min()
    df['time_delta_max'] = time_deltas.max()
    df['time_delta_var'] = time_deltas.var()

    df['distance_deltas_max'] = distance_deltas.max()

    df['aim_distance_diff_max'] = distance_aim_deltas_diff.max()
    df['aim_distance_diff_var'] = distance_aim_deltas_diff.var()

    df['xy_speed_mean'] = speeds.mean()
    df['xy_speed_median'] = speeds.median()
    df['xy_speed_kurt'] = angles.kurt()
    df['xy_speed_skew'] = angles.skew()

    df['y_dist_x_dist_ratio_var'] = angles.var()
    df['y_dist_x_dist_ratio_kurt'] = angles.kurt()
    df['y_dist_x_dist_ratio_skew'] = angles.skew()

    df['xy_angle_diff_var'] = angle_diff.var()
    df['xy_angle_diff_mean'] = angle_diff.mean()
    df['xy_angle_diff_kurt'] = angle_diff.kurt()
    df['xy_angle_diff_skew'] = angle_diff.skew()

    df['y_var'] = ys.var()
    df['y_kurt'] = ys.kurt()
    df['y_skew'] = ys.skew()

    df['x_var'] = xs.var()
    df['y_kurt'] = ys.kurt()
    df['y_skew'] = ys.skew()

    df['x_init'] = xs.values[0]
    df['y_init'] = ys.values[0]

    df['x_back_num'] = min((xs.diff(1).dropna() > 0).sum(), (xs.diff(1).dropna() < 0).sum())  # 28
    df['y_back_num'] = min((ys.diff(1).dropna() > 0).sum(), (ys.diff(1).dropna() < 0).sum())  # 29

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


def get_speed_var_feature(df):
    points = []
    for point in df.trajectory[:-1].split(';'):
        point = point.split(',')
        points.append(((float(point[0]), float(point[1])), float(point[2])))

    xs = pd.Series([point[0][0] for point in points])
    ys = pd.Series([point[0][1] for point in points])
    time_deltas = pd.Series([points[i + 1][1] - points[i][1] for i in range(len(points) - 1)])

    x_diff = xs.diff(1).dropna()
    x_speed = pd.Series([np.log1p(x_delta) - np.log1p(delta) for (x_delta, delta) in zip(x_diff, time_deltas)])
    x_speed_diff = x_speed.diff(1).dropna()
    speed_diff_time = pd.Series([time_deltas[i]/2 + time_deltas[i+1]/2 for i in range(len(time_deltas)-1)])
    x_acc = pd.Series([np.log1p(speed_delta) - np.log1p(time_delta) for (speed_delta, time_delta) in zip(x_speed_diff, speed_diff_time)])

    y_diff = ys.diff(1).dropna()
    y_speed = pd.Series([np.log1p(y_delta) - np.log1p(delta) for (y_delta, delta) in zip(y_diff, time_deltas)])
    y_speed_diff = y_speed.diff(1).dropna()
    y_acc = pd.Series([np.log1p(speed_delta) - np.log1p(time_delta) for (speed_delta, time_delta) in zip(y_speed_diff, speed_diff_time)])
    df['x_acc_var'] = x_acc.var()
    df['y_acc_var'] = y_acc.var()

    return df.to_frame().T


def make_balck_train_set():

    balck_data = pd.read_csv('../data/raw_data/point_than_2_black_data.txt', sep=' ', header=None,
                        names=['id', 'trajectory', 'aim', 'label'])

    # train = applyParallel(balck_data.iterrows(), get_feature).sort_values(by='id')
    # del train['trajectory']
    # del train['aim']
    # del train['label']
    # train.to_csv('../data/added_feature/black_added_feature.csv', index=None)
    #
    # train_1 = applyParallel(balck_data.iterrows(), get_single_feature).sort_values(by='id')
    # del train_1['trajectory']
    # del train_1['aim']
    # del train_1['label']
    # train_1.to_csv('../data/added_feature/black_added_single_feature.csv', index=None)

    train_2 = applyParallel(balck_data.iterrows(), get_speed_var_feature).sort_values(by='id')
    del train_2['trajectory']
    del train_2['aim']
    del train_2['label']
    train_2['x_acc_var'].plot(kind='bar')
    train_2['y_acc_var'].plot(kind='bar')
    train_2.to_csv('../data/added_feature/black_speed_var_feature.csv', index=None)


def make_white_train_set():
    white_data = pd.read_csv('../data/raw_data/point_than_2_white_data.txt', sep=' ', header=None,
                        names=['id', 'trajectory', 'aim', 'label'])
    # train = applyParallel(white_data.iterrows(), get_feature).sort_values(by='id')
    # del train['trajectory']
    # del train['aim']
    # del train['label']
    # train.to_csv('../data/added_feature/white_added_feature.csv', index=None)
    #
    # train_1 = applyParallel(white_data.iterrows(), get_single_feature).sort_values(by='id')
    # del train_1['trajectory']
    # del train_1['aim']
    # del train_1['label']
    # train_1.to_csv('../data/added_feature/white_added_single_feature.csv', index=None)

    train_2 = applyParallel(white_data.iterrows(), get_speed_var_feature).sort_values(by='id')
    del train_2['trajectory']
    del train_2['aim']
    del train_2['label']
    train_2['x_acc_var'].plot(kind='bar')
    train_2['y_acc_var'].plot(kind='bar')
    train_2.to_csv('../data/added_feature/white_speed_var_feature.csv', index=None)


def make_predict_train_set():
    predict_data = pd.read_csv('../data/raw_data/point_than_2_predict_data.txt', sep=' ', header=None,
                        names=['id', 'trajectory', 'aim', 'label'])
    # train = applyParallel(predict_data.iterrows(), get_feature).sort_values(by='id')
    # del train['trajectory']
    # del train['aim']
    # del train['label']
    # train.to_csv('../data/added_feature/predict_added_feature.csv', index=None)
    #
    # train_1 = applyParallel(predict_data.iterrows(), get_single_feature).sort_values(by='id')
    # del train_1['trajectory']
    # del train_1['aim']
    # del train_1['label']
    # train_1.to_csv('../data/added_feature/predict_added_single_feature.csv', index=None)

    train_2 = applyParallel(predict_data.iterrows(), get_speed_var_feature).sort_values(by='id')
    del train_2['trajectory']
    del train_2['aim']
    del train_2['label']
    train_2.to_csv('../data/added_feature/predict_speed_var_feature.csv', index=None)


if __name__ == '__main__':

    make_balck_train_set()
    make_white_train_set()
    # make_predict_train_set()
