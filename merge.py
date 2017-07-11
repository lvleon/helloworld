# -*- coding: UTF-8 -*-
import pandas as pd


def merge_feature():

    # 速度类特征
    white_speed_feature_file = pd.read_csv('../data/speed_feature/white_speed_feature_file.csv')
    black_speed_feature_file = pd.read_csv('../data/speed_feature/black_speed_feature_file.csv')
    predict_speed_feature_file = pd.read_csv('../data/speed_feature/predict_speed_feature_file.csv')

    # 数量类特征
    white_count_feature_file = pd.read_csv('../data/count_feature/white_count_feature_file.csv')
    black_count_feature_file = pd.read_csv('../data/count_feature/black_count_feature_file.csv')
    predict_count_feature_file = pd.read_csv('../data/count_feature/predict_count_feature_file.csv')

    # 时间类特征
    white_time_feature_file = pd.read_csv('../data/time_feature/white_time_feature_file.csv')
    black_time_feature_file = pd.read_csv('../data/time_feature/black_time_feature_file.csv')
    predict_time_feature_file = pd.read_csv('../data/time_feature/predict_time_feature_file.csv')

    # 距离类特征
    white_distance_feature_file = pd.read_csv('../data/distance_feature/white_distance_feature_file.csv')
    black_distance_feature_file = pd.read_csv('../data/distance_feature/black_distance_feature_file.csv')
    predict_distance_feature_file = pd.read_csv('../data/distance_feature/predict_distance_feature_file.csv')

    df_list = [(white_speed_feature_file, white_count_feature_file, white_time_feature_file, white_distance_feature_file, '../data/white_train_feature.csv'),
                 (black_speed_feature_file, black_count_feature_file, black_time_feature_file, black_distance_feature_file,  '../data/black_train_feature.csv'),
                 (predict_speed_feature_file, predict_count_feature_file, predict_time_feature_file, predict_distance_feature_file, '../data/predict_feature.csv')]
    for m_df in df_list:
        cache_df1 = pd.merge(m_df[0], m_df[1], how='left', on='id')
        cache_df2 = pd.merge(cache_df1, m_df[2], how='left', on='id')
        out_df = pd.merge(cache_df2, m_df[3], how='left', on='id')
        out_df.to_csv(m_df[4])


def add_label():
    white_train_feature = open('../data/white_train_feature.csv')
    black_train_feature = open('../data/black_train_feature.csv')

    point_than_2_white_data = open('../data/raw_data/point_than_2_white_data.txt')
    point_than_2_black_data = open('../data/raw_data/point_than_2_black_data.txt')

    white_train_feature_label = open('../data/white_train_feature_label.csv', 'w')
    black_train_feature_label = open('../data/black_train_feature_label.csv', 'w')

    file_list = [(white_train_feature, point_than_2_white_data, white_train_feature_label), (black_train_feature, point_than_2_black_data, black_train_feature_label)]

    for m_file in file_list:
        line = m_file[0].readline()
        line_ = line.split('\n')
        m_file[2].write(line_[0]+',label\n')
        for line in m_file[0].readlines():
            l_line = m_file[1].readline()
            l_line_ = l_line.split(' ')
            line_ = line.split('\n')
            n_line = line_[0]+','+str(float(l_line_[3]))+'\n'
            m_file[2].write(n_line)
        m_file[0].close()
        m_file[1].close()
        m_file[2].close()


def run():
    merge_feature()
    add_label()

run()