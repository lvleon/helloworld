# -*- coding: UTF-8 -*-
from collections import defaultdict
import pandas as pd


def generate_repeat_feature():
    x_unrepeated_white_file = open('../../data/raw_data/x_unrepeated_white_data.txt')
    x_unrepeated_black_file = open('../../data/raw_data/x_unrepeated_black_data.txt')
    x_unrepeated_predict_file = open('../../data/raw_data/x_unrepeated_predict_data.txt')

    y_unrepeated_white_file = open('../../data/raw_data/y_unrepeated_white_data.txt')
    y_unrepeated_black_file = open('../../data/raw_data/y_unrepeated_black_data.txt')
    y_unrepeated_predict_file = open('../../data/raw_data/y_unrepeated_predict_data.txt')

    xy_unrepeated_white_file = open('../../data/raw_data/xy_unrepeated_white_data.txt')
    xy_unrepeated_black_file = open('../../data/raw_data/xy_unrepeated_black_data.txt')
    xy_unrepeated_predict_file = open('../../data/raw_data/xy_unrepeated_predict_data.txt')

    raw_white_file = open('../../data/raw_data/raw_white_data.txt')
    raw_black_file = open('../../data/raw_data/raw_black_data.txt')
    raw_predict_file = open('../../data/raw_data/dsjtzs_txfz_test1.txt')

    white_repeat_feature_file = open('../../data/count_feature/white_repeat_feature_file.csv', 'w')
    black_repeat_feature_file = open('../../data/count_feature/black_repeat_feature_file.csv', 'w')
    predict_repeat_feature_file = open('../../data/count_feature/predict_repeat_feature_file.csv', 'w')

    file_list = [(x_unrepeated_white_file, y_unrepeated_white_file, xy_unrepeated_white_file, raw_white_file, white_repeat_feature_file),
                 (x_unrepeated_black_file, y_unrepeated_black_file, xy_unrepeated_black_file, raw_black_file, black_repeat_feature_file),
                 (x_unrepeated_predict_file, y_unrepeated_predict_file, xy_unrepeated_predict_file, raw_predict_file, predict_repeat_feature_file)]
    for file_pair in file_list:
        file_pair[4].write('id,unrepeated_x_count,x_count_ratio,unrepeated_y_count,y_count_ratio,unrepeated_xy_count,xy_count_ratio\n')
        for line in file_pair[3].readlines():
            cols = line.split(' ')
            point_list = cols[1].split(';')
            raw_length = len(point_list) - 1
            x_line = file_pair[0].readline()
            cols = x_line.split(' ')
            point_list = cols[1].split(';')
            unrepeated_x_count = float(len(point_list) - 1)
            x_count_ratio = unrepeated_x_count/raw_length
            y_line = file_pair[1].readline()
            cols = y_line.split(' ')
            point_list = cols[1].split(';')
            unrepeated_y_count = float(len(point_list) - 1)
            y_count_ratio = unrepeated_y_count/raw_length
            xy_line = file_pair[2].readline()
            cols = xy_line.split(' ')
            point_list = cols[1].split(';')
            unrepeated_xy_count = float(len(point_list) - 1)
            xy_count_ratio = unrepeated_xy_count/raw_length
            file_pair[4].write(cols[0]+','+str(unrepeated_x_count)+','+str(x_count_ratio)+','+str(unrepeated_y_count)+','+str(y_count_ratio)+','+str(unrepeated_xy_count)+','+str(xy_count_ratio)+'\n')
        file_pair[0].close()
        file_pair[1].close()
        file_pair[2].close()
        file_pair[3].close()
        file_pair[4].close()


def generate_stop_feature():
    white_xt_file = open('../../data/cache_data/white_xt_data.txt')
    black_xt_file = open('../../data/cache_data/black_xt_data.txt')
    predict_xt_file = open('../../data/cache_data/predict_xt_data.txt')

    white_yt_file = open('../../data/cache_data/white_yt_data.txt')
    black_yt_file = open('../../data/cache_data/black_yt_data.txt')
    predict_yt_file = open('../../data/cache_data/predict_yt_data.txt')

    white_file = open('../../data/raw_data/point_than_2_white_data.txt')
    black_file = open('../../data/raw_data/point_than_2_black_data.txt')
    predict_file = open('../../data/raw_data/point_than_2_predict_data.txt')

    white_stop_feature_file = open('../../data/count_feature/white_stop_feature_file.csv', 'w')
    black_stop_feature_file = open('../../data/count_feature/black_stop_feature_file.csv', 'w')
    predict_stop_feature_file = open('../../data/count_feature/predict_stop_feature_file.csv', 'w')

    file_list = [(white_xt_file, white_yt_file, white_file, white_stop_feature_file),
                 (black_xt_file, black_yt_file, black_file, black_stop_feature_file),
                 (predict_xt_file, predict_yt_file, predict_file, predict_stop_feature_file)]

    for m_file in file_list:
        m_file[3].write('id,x_stop_count,y_stop_count,xy_stop_count\n')
        for line in m_file[0].readlines():
            x_stop_count = 0.0
            cols = line.split(' ')
            point_list = cols[1].split(';')
            length = len(point_list) - 1
            pre = point_list[0].split(',')
            stop = False
            for i in range(1, length):
                now = point_list[i].split(',')
                if pre[0] == now[0]:
                    stop = True
                elif stop:
                    x_stop_count += 1
                    stop = False
                pre = now
            if stop:
                x_stop_count += 1

            yt_line = m_file[1].readline()
            y_stop_count = 0.0
            cols = yt_line.split(' ')
            point_list = cols[1].split(';')
            length = len(point_list) - 1
            pre = point_list[0].split(',')
            stop = False
            for i in range(1, length):
                now = point_list[i].split(',')
                if pre[0] == now[0]:
                    stop = True
                elif stop:
                    y_stop_count += 1
                    stop = False
                pre = now
            if stop:
                y_stop_count += 1

            xyt_line = m_file[2].readline()
            xy_stop_count = 0.0
            cols = xyt_line.split(' ')
            point_list = cols[1].split(';')
            length = len(point_list) - 1
            pre = point_list[0].split(',')
            stop = False
            for i in range(1, length):
                now = point_list[i].split(',')
                if pre[0] == now[0] and pre[1] == now[1]:
                    stop = True
                elif stop:
                    xy_stop_count += 1
                    stop = False
                pre = now
            if stop:
                xy_stop_count += 1
            m_file[3].write(cols[0]+','+str(x_stop_count)+','+str(y_stop_count)+','+str(xy_stop_count)+'\n')
        m_file[0].close()
        m_file[1].close()
        m_file[2].close()
        m_file[3].close()


def generate_wave_feature():
    white_x_speed_file = open('../../data/cache_data/speed/white_x_speed_data.txt')
    black_x_speed_file = open('../../data/cache_data/speed/black_x_speed_data.txt')
    predict_x_speed_file = open('../../data/cache_data/speed/predict_x_speed_data.txt')

    white_y_speed_file = open('../../data/cache_data/speed/white_y_speed_data.txt')
    black_y_speed_file = open('../../data/cache_data/speed/black_y_speed_data.txt')
    predict_y_speed_file = open('../../data/cache_data/speed/predict_y_speed_data.txt')

    white_stop_feature_file = open('../../data/count_feature/white_wave_feature_file.csv', 'w')
    black_stop_feature_file = open('../../data/count_feature/black_wave_feature_file.csv', 'w')
    predict_stop_feature_file = open('../../data/count_feature/predict_wave_feature_file.csv', 'w')

    file_list = [(white_x_speed_file, white_y_speed_file, white_stop_feature_file),
                 (black_x_speed_file, black_y_speed_file, black_stop_feature_file),
                 (predict_x_speed_file, predict_y_speed_file, predict_stop_feature_file)]
    for m_file in file_list:
        m_file[2].write('id,x_wave_count,y_wave_count\n')
        for line in m_file[0].readlines():
            cols = line.split(' ')
            speedts = cols[1].split(';')
            length = len(speedts) - 1
            pre_speedt = speedts[0].split(',')
            pre_speed = float(pre_speedt[0])
            x_wave_count = 0.0
            for i in range(1, length):
                speedt = speedts[i].split(',')
                speed = float(speedt[0])
                if speed*pre_speed < 0:
                    x_wave_count += 1
                pre_speed = speed

            yspeedt_line = m_file[1].readline()
            cols = yspeedt_line.split(' ')
            speedts = cols[1].split(';')
            length = len(speedts) - 1
            pre_speedt = speedts[0].split(',')
            pre_speed = float(pre_speedt[0])
            y_wave_count = 0.0
            for i in range(1, length):
                speedt = speedts[i].split(',')
                speed = float(speedt[0])
                if speed * pre_speed < 0:
                    y_wave_count += 1
                pre_speed = speed
            m_file[2].write(cols[0]+','+str(x_wave_count)+','+str(y_wave_count)+'\n')
        m_file[0].close()
        m_file[1].close()
        m_file[2].close()


def generate_multi_point_feature():
    white_file = open('../../data/raw_data/point_than_2_white_data.txt')
    black_file = open('../../data/raw_data/point_than_2_black_data.txt')
    predict_file = open('../../data/raw_data/point_than_2_predict_data.txt')

    white_multi_point_feature_file = open('../../data/count_feature/white_multi_point_feature_file.csv', 'w')
    black_multi_point_feature_file = open('../../data/count_feature/black_multi_point_feature_file.csv', 'w')
    predict_multi_point_feature_file = open('../../data/count_feature/predict_multi_point_feature_file.csv', 'w')

    file_list = [(white_file, white_multi_point_feature_file),
                 (black_file, black_multi_point_feature_file),
                 (predict_file, predict_multi_point_feature_file)]
    for m_file in file_list:
        m_file[1].write('id,multi_point_one_time\n')
        for line in m_file[0].readlines():
            cols = line.split(' ')
            points = cols[1].split(';')
            t_dict = defaultdict(set)
            length = len(points) - 1
            multi_point = 0.0
            for i in range(0, length):
                point = points[i].split(',')
                t = point[2]
                xy = point[0]+','+point[1]
                t_dict[t].add(xy)
            for item in t_dict.items():
                if len(item[1]) > 1:
                    multi_point = 1.0
                    break
            m_file[1].write(cols[0]+','+str(multi_point)+'\n')
        m_file[0].close()
        m_file[1].close()


def merge():
    white_repeat_feature_file = pd.read_csv('../../data/count_feature/white_repeat_feature_file.csv')
    black_repeat_feature_file = pd.read_csv('../../data/count_feature/black_repeat_feature_file.csv')
    predict_repeat_feature_file = pd.read_csv('../../data/count_feature/predict_repeat_feature_file.csv')

    white_stop_feature_file = pd.read_csv('../../data/count_feature/white_stop_feature_file.csv')
    black_stop_feature_file = pd.read_csv('../../data/count_feature/black_stop_feature_file.csv')
    predict_stop_feature_file = pd.read_csv('../../data/count_feature/predict_stop_feature_file.csv')

    white_wave_feature_file = pd.read_csv('../../data/count_feature/white_wave_feature_file.csv')
    black_wave_feature_file = pd.read_csv('../../data/count_feature/black_wave_feature_file.csv')
    predict_wave_feature_file = pd.read_csv('../../data/count_feature/predict_wave_feature_file.csv')

    white_multi_point_feature_file = pd.read_csv('../../data/count_feature/white_multi_point_feature_file.csv')
    black_multi_point_feature_file = pd.read_csv('../../data/count_feature/black_multi_point_feature_file.csv')
    predict_multi_point_feature_file = pd.read_csv('../../data/count_feature/predict_multi_point_feature_file.csv')

    white_count_feature_file = '../../data/count_feature/white_count_feature_file.csv'
    black_count_feature_file = '../../data/count_feature/black_count_feature_file.csv'
    predict_count_feature_file = '../../data/count_feature/predict_count_feature_file.csv'

    df_list = [(white_repeat_feature_file, white_stop_feature_file, white_wave_feature_file, white_multi_point_feature_file, white_count_feature_file),
                 (black_repeat_feature_file, black_stop_feature_file, black_wave_feature_file, black_multi_point_feature_file, black_count_feature_file),
                 (predict_repeat_feature_file, predict_stop_feature_file, predict_wave_feature_file, predict_multi_point_feature_file, predict_count_feature_file)]
    for m_df in df_list:
        cache_df1 = pd.merge(m_df[0], m_df[1], how='left', on='id')
        cache_df2 = pd.merge(cache_df1, m_df[2], how='left', on='id')
        out_df = pd.merge(cache_df2, m_df[3], how='left', on='id')
        out_df.to_csv(m_df[4], index=False)


def run():
    generate_repeat_feature()
    generate_stop_feature()
    generate_wave_feature()
    generate_multi_point_feature()
    merge()
    pass

run()
