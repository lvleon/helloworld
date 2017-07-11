# -*- coding: UTF-8 -*-
import pandas as pd


def merge_feature():

    white_feature_file = pd.read_csv('../data/white_train_feature_plus.csv')
    black_feature_file = pd.read_csv('../data/black_train_feature_plus.csv')
    predict_feature_file = pd.read_csv('../data/predict_feature_plus.csv')

    white_added_feature_file_2 = pd.read_csv('../data/added_feature/white_added_feature_2.csv')
    black_added_feature_file_2 = pd.read_csv('../data/added_feature/black_added_feature_2.csv')
    predict_added_feature_file_2 = pd.read_csv('../data/added_feature/predict_added_feature_2.csv')

    white_added_feature_file_3 = pd.read_csv('../data/added_feature/white_added_feature_3.csv')
    black_added_feature_file_3 = pd.read_csv('../data/added_feature/black_added_feature_3.csv')
    predict_added_feature_file_3 = pd.read_csv('../data/added_feature/predict_added_feature_3.csv')

    df_list = [(white_feature_file,  white_added_feature_file_2, white_added_feature_file_3,  '../data/white_train_feature_plus.csv'),
                 (black_feature_file, black_added_feature_file_2, black_added_feature_file_3,  '../data/black_train_feature_plus.csv'),
                 (predict_feature_file, predict_added_feature_file_2, predict_added_feature_file_3, '../data/predict_feature_plus.csv')]

    for m_df in df_list:
        cache_df = pd.merge(m_df[0], m_df[1], how='left', on='id')
        out_df = pd.merge(cache_df, m_df[2], how='left', on='id')
        out_df.to_csv(m_df[3])


def add_label():
    white_train_feature = open('../data/white_train_feature_plus.csv')
    black_train_feature = open('../data/black_train_feature_plus.csv')

    point_than_2_white_data = open('../data/raw_data/point_than_2_white_data.txt')
    point_than_2_black_data = open('../data/raw_data/point_than_2_black_data.txt')

    white_train_feature_label = open('../data/white_train_feature_plus_label.csv', 'w')
    black_train_feature_label = open('../data/black_train_feature_plus_label.csv', 'w')

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