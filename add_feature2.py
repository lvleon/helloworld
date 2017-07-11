# -*- coding: UTF-8 -*-
import pandas as pd
import scipy as sp
import numpy as np
import warnings
from scipy import spatial
from add_feature import applyParallel

warnings.filterwarnings("ignore")


def get_feature(df):

    points = []
    for point in df.trajectory[:-1].split(';'):
        point = point.split(',')
        points.append(((float(point[0]), float(point[1])), float(point[2])))

    ys = pd.Series([point[0][1] for point in points])

    aim = df.aim.split(',')
    aim = (float(aim[0]), float(aim[1]))

    x_deltas = pd.Series([points[i+1][0][0] - points[i][0][0] for i in range(len(points) - 1)])
    y_deltas = pd.Series([points[i+1][0][1] - points[i][0][1] for i in range(len(points) - 1)])
    time_deltas = pd.Series([points[i + 1][1] - points[i][1] for i in range(len(points) - 1)])

    x_speeds = pd.Series([np.log1p(distance) - np.log1p(delta) for (distance, delta) in zip(x_deltas, time_deltas)])
    y_speeds = pd.Series([np.log1p(distance) - np.log1p(delta) for (distance, delta) in zip(y_deltas, time_deltas)])

    df['x_speed_mean'] = x_speeds.mean()
    df['x_speed_median'] = x_speeds.median()
    df['x_last_speed'] = x_speeds.values[-1]

    df['y_speed_mean'] = y_speeds.mean()
    df['y_speed_median'] = y_speeds.median()
    df['y_last_speed'] = y_speeds.values[-1]

    x_aim_deltas = pd.Series([points[i][0][0] - aim[0] for i in range(len(points))])
    y_aim_deltas = pd.Series([points[i][0][1] - aim[1] for i in range(len(points))])
    xy_aim_deltas = pd.Series([sp.spatial.distance.euclidean(points[i][0], aim) for i in range(len(points))])

    df['x_aim_dist_min'] = x_aim_deltas.min()
    df['x_aim_dist_max'] = x_aim_deltas.max()
    df['x_aim_dist_var'] = x_aim_deltas.var()
    df['x_aim_dist_median'] = x_aim_deltas.median()
    df['x_aim_dist_last'] = x_aim_deltas.values[-1]

    df['y_aim_dist_min'] = y_aim_deltas.min()
    df['y_aim_dist_max'] = y_aim_deltas.max()
    df['y_aim_dist_var'] = y_aim_deltas.var()
    df['y_aim_dist_median'] = y_aim_deltas.median()
    df['y_aim_dist_last'] = y_aim_deltas.values[-1]

    df['y_last_y_init_diff'] = ys.values[-1] - ys.values[0]

    df['xy_aim_dist_min'] = xy_aim_deltas.min()
    df['xy_aim_dist_max'] = xy_aim_deltas.max()
    df['xy_aim_dist_var'] = xy_aim_deltas.var()
    df['xy_aim_dist_median'] = xy_aim_deltas.median()

    if len([y_aim_delta for y_aim_delta in y_aim_deltas if y_aim_delta < 0]) > 0:
        df['aim_y_arrive'] = 1.0
    else:
        df['aim_y_arrive'] = 0.0

    return df.to_frame().T


def make_balck_train_set():

    balck_data = pd.read_csv('../data/raw_data/point_than_2_black_data.txt', sep=' ', header=None,
                        names=['id', 'trajectory', 'aim', 'label'])

    train = applyParallel(balck_data.iterrows(), get_feature).sort_values(by='id')
    del train['trajectory']
    del train['aim']
    del train['label']

    train.to_csv('../data/added_feature/black_added_feature_2.csv', index=None)


def make_white_train_set():
    white_data = pd.read_csv('../data/raw_data/point_than_2_white_data.txt', sep=' ', header=None,
                        names=['id', 'trajectory', 'aim', 'label'])

    train = applyParallel(white_data.iterrows(), get_feature).sort_values(by='id')
    del train['trajectory']
    del train['aim']
    del train['label']
    train.to_csv('../data/added_feature/white_added_feature_2.csv', index=None)


def make_predict_train_set():
    predict_data = pd.read_csv('../data/raw_data/point_than_2_predict_data.txt', sep=' ', header=None,
                        names=['id', 'trajectory', 'aim', 'label'])

    train = applyParallel(predict_data.iterrows(), get_feature).sort_values(by='id')
    del train['trajectory']
    del train['aim']
    del train['label']
    train.to_csv('../data/added_feature/predict_added_feature_2.csv', index=None)


if __name__ == '__main__':

    make_balck_train_set()
    make_white_train_set()
    make_predict_train_set()
