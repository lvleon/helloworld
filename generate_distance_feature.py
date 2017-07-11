# -*- coding: UTF-8 -*-
import pandas as pd

def generate_x_feature():

    white_xt_file = open('../../data/cache_data/white_xt_data.txt')
    black_xt_file = open('../../data/cache_data/black_xt_data.txt')
    predict_xt_file = open('../../data/cache_data/predict_xt_data.txt')

    white_x_feature_file = open('../../data/distance_feature/white_x_feature_file.csv', 'w')
    black_x_feature_file = open('../../data/distance_feature/black_x_feature_file.csv', 'w')
    predict_x_feature_file = open('../../data/distance_feature/predict_x_feature_file.csv', 'w')

    file_list = [(white_xt_file, white_x_feature_file),
                 (black_xt_file, black_x_feature_file),
                 (predict_xt_file, predict_x_feature_file)]
    for m_file in file_list:
        m_file[1].write('id,x_min,x_max,x_range,aim_x_arrive,x_dist_ratio_1,x_dist_ratio_2\n')
        for line in m_file[0].readlines():
            cols = line.split(' ')
            point_list = cols[1].split(';')
            length = len(point_list) - 1
            pre = point_list[0].split(',')
            first_x = float(pre[0])
            x_min = first_x
            x_max = x_min
            x_range = 0.0
            aim_x = float(cols[2])
            aim_x_arrive = 0.0
            initial_distance = abs(aim_x-x_min)
            beyond_distance = 0.0
            x_dist_ratio_1 = 0.0
            x_dist_ratio_2 = 0.0
            arrive_flag = False
            max_x_index = 0
            min_x_after_arrive = 10000.0
            for i in range(1, length):
                now = point_list[i].split(',')
                now_x = float(now[0])
                if now_x < x_min:
                    x_min = now_x
                if now_x > x_max:
                    x_max = now_x
                    max_x_index = i
                if now_x > aim_x and not arrive_flag:
                    aim_x_arrive = 1.0
                    arrive_flag = True

            if arrive_flag:
                for i in range(max_x_index, length):
                    now = point_list[i].split(',')
                    now_x = float(now[0])
                    if now_x < min_x_after_arrive:
                        min_x_after_arrive = now_x
            x_range = abs(x_max - x_min)
            if abs(x_max - first_x) == 0:
                x_dist_ratio_1 = -1.0
            else:
                x_dist_ratio_1 = initial_distance / abs(x_max - first_x)
            if not arrive_flag:
                x_dist_ratio_2 = 0.0
            elif x_max == aim_x:
                x_dist_ratio_2 = 0.0
            else:
                x_dist_ratio_2 = (x_max - min_x_after_arrive) / (x_max - aim_x)
            m_file[1].write(cols[0]+','+str(x_min)+','+str(x_max)+','+str(x_range)+','
                            +str(aim_x_arrive)+','+str(x_dist_ratio_1)+','+str(x_dist_ratio_2)+'\n')
        m_file[0].close()
        m_file[1].close()


def generate_y_feature():
    white_xt_file = open('../../data/cache_data/white_yt_data.txt')
    black_xt_file = open('../../data/cache_data/black_yt_data.txt')
    predict_xt_file = open('../../data/cache_data/predict_yt_data.txt')

    white_y_feature_file = open('../../data/distance_feature/white_y_feature_file.csv', 'w')
    black_y_feature_file = open('../../data/distance_feature/black_y_feature_file.csv', 'w')
    predict_y_feature_file = open('../../data/distance_feature/predict_y_feature_file.csv', 'w')

    file_list = [(white_xt_file, white_y_feature_file),
                 (black_xt_file, black_y_feature_file),
                 (predict_xt_file, predict_y_feature_file)]
    for m_file in file_list:
        m_file[1].write('id,y_min,y_max,y_range\n')
        for line in m_file[0].readlines():
            cols = line.split(' ')
            point_list = cols[1].split(';')
            length = len(point_list) - 1
            pre = point_list[0].split(',')
            first_y = float(pre[0])
            y_min = first_y
            y_max = y_min
            y_range = 0.0
            for i in range(1, length):
                now = point_list[i].split(',')
                now_y = float(now[0])
                if now_y < y_min:
                    y_min = now_y
                if now_y > y_max:
                    y_max = now_y
            y_range = y_max - y_min
            m_file[1].write(cols[0]+','+str(y_min)+','+str(y_max)+','+str(y_range)+'\n')
        m_file[0].close()
        m_file[1].close()


def generate_xy_feature():
    white_file = open('../../data/raw_data/point_than_2_white_data.txt')
    black_file = open('../../data/raw_data/point_than_2_black_data.txt')
    predict_file = open('../../data/raw_data/point_than_2_predict_data.txt')

    white_xy_feature_file = open('../../data/distance_feature/white_xy_feature_file.csv', 'w')
    black_xy_feature_file = open('../../data/distance_feature/black_xy_feature_file.csv', 'w')
    predict_xy_feature_file = open('../../data/distance_feature/predict_xy_feature_file.csv', 'w')

    file_list = [(white_file, white_xy_feature_file),
                 (black_file, black_xy_feature_file),
                 (predict_file, predict_xy_feature_file)]
    for m_file in file_list:
        m_file[1].write('id,x_dist_y_dist_ratio_max,y_dist_x_dist_ratio_max,xy_stop_point_aim_dist\n')
        for line in m_file[0].readlines():
            cols = line.split(' ')
            aim_xy = cols[2].split(',')
            point_list = cols[1].split(';')
            length = len(point_list) - 1
            pre = point_list[0].split(',')
            pre_x = float(pre[0])
            pre_y = float(pre[1])
            last_point = point_list[length-1].split(',')
            d = pow(float(last_point[0])-float(aim_xy[0]),2)+pow(float(last_point[1])-float(aim_xy[1]),2)
            xy_stop_point_aim_dist = pow(d, 0.5)
            x_dist_y_dist_ratio_max = 0.0
            y_dist_x_dist_ratio_max = 0.0

            for i in range(1, length):
                now = point_list[i].split(',')
                now_x = float(now[0])
                now_y = float(now[1])
                if now_y - pre_y:
                    x_dist_y_dist_ratio = abs((now_x-pre_x)/(now_y-pre_y))
                    if x_dist_y_dist_ratio > x_dist_y_dist_ratio_max:
                        x_dist_y_dist_ratio_max = x_dist_y_dist_ratio
                if now_x - pre_x:
                    y_dist_x_dist_ratio = abs((now_y-pre_y)/(now_x-pre_x))
                    if y_dist_x_dist_ratio > y_dist_x_dist_ratio_max:
                        y_dist_x_dist_ratio_max = y_dist_x_dist_ratio
                pre_x = now_x
                pre_y = now_y

            m_file[1].write(cols[0]+','+str(x_dist_y_dist_ratio_max)+','+str(y_dist_x_dist_ratio_max)+','+str(xy_stop_point_aim_dist)+'\n')
        m_file[0].close()
        m_file[1].close()


def merge():
    white_x_feature_file = pd.read_csv('../../data/distance_feature/white_x_feature_file.csv')
    black_x_feature_file = pd.read_csv('../../data/distance_feature/black_x_feature_file.csv')
    predict_x_feature_file = pd.read_csv('../../data/distance_feature/predict_x_feature_file.csv')

    white_y_feature_file = pd.read_csv('../../data/distance_feature/white_y_feature_file.csv')
    black_y_feature_file = pd.read_csv('../../data/distance_feature/black_y_feature_file.csv')
    predict_y_feature_file = pd.read_csv('../../data/distance_feature/predict_y_feature_file.csv')

    white_xy_feature_file = pd.read_csv('../../data/distance_feature/white_xy_feature_file.csv')
    black_xy_feature_file = pd.read_csv('../../data/distance_feature/black_xy_feature_file.csv')
    predict_xy_feature_file = pd.read_csv('../../data/distance_feature/predict_xy_feature_file.csv')

    white_distance_feature_file = '../../data/distance_feature/white_distance_feature_file.csv'
    black_distance_feature_file = '../../data/distance_feature/black_distance_feature_file.csv'
    predict_distance_feature_file = '../../data/distance_feature/predict_distance_feature_file.csv'



    df_list = [(white_x_feature_file, white_y_feature_file, white_xy_feature_file, white_distance_feature_file),
                 (black_x_feature_file, black_y_feature_file, black_xy_feature_file, black_distance_feature_file),
                 (predict_x_feature_file, predict_y_feature_file, predict_xy_feature_file, predict_distance_feature_file)]
    for m_df in df_list:
        cache_df = pd.merge(m_df[0], m_df[1], how='left', on='id')
        out_df = pd.merge(cache_df, m_df[2], how='left', on='id')
        out_df.to_csv(m_df[3], index=False)


def run():
    generate_x_feature()
    generate_y_feature()
    generate_xy_feature()
    merge()

run()