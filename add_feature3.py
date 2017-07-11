# -*- coding: UTF-8 -*-
import numpy as np


def generate_stop_feature():
    white_xt_file = open('../data/cache_data/white_xt_data.txt')
    black_xt_file = open('../data/cache_data/black_xt_data.txt')
    predict_xt_file = open('../data/cache_data/predict_xt_data.txt')

    white_yt_file = open('../data/cache_data/white_yt_data.txt')
    black_yt_file = open('../data/cache_data/black_yt_data.txt')
    predict_yt_file = open('../data/cache_data/predict_yt_data.txt')

    white_stop_feature_file = open('../data/added_feature/white_added_feature_3.csv', 'w')
    black_stop_feature_file = open('../data/added_feature/black_added_feature_3.csv', 'w')
    predict_stop_feature_file = open('../data/added_feature/predict_added_feature_3.csv', 'w')

    file_list = [(white_xt_file, white_yt_file, white_stop_feature_file),
                 (black_xt_file, black_yt_file, black_stop_feature_file),
                 (predict_xt_file, predict_yt_file, predict_stop_feature_file)]

    for m_file in file_list:
        m_file[2].write('id,x_first_stop_moment,x_first_stop_time,x_1st_stop_time_moment_ratio,x_2st_stop_moment,x_2st_stop_time,x_2_1st_stop_moment_ratio,'
                        'x_2_1st_stop_time_ratio,y_first_stop_moment,y_first_stop_time,y_1st_stop_time_moment_ratio,lowst_y_moment\n')
        for line in m_file[0].readlines():
            cols = line.split(' ')
            point_list = cols[1].split(';')
            length = len(point_list) - 1
            pre = point_list[0].split(',')
            last_point = point_list[length-1].split(',')
            last_time = float(last_point[1])

            x_first_stop_moment = last_time
            x_first_stop_time = 0.0
            x_1st_stop_time_moment_ratio = 0.0

            x_2st_stop_moment = last_time
            x_2st_stop_time = 0.0

            x_2_1st_stop_moment_ratio = 0.0
            x_2_1st_stop_time_ratio = 0.0

            find_first_stop_moment = False
            find_second_stop_moment = False

            for i in range(1, length):
                now = point_list[i].split(',')
                time_diff = float(now[1]) - float(pre[1])
                if pre[0] == now[0]:
                    if not find_first_stop_moment:
                        x_first_stop_moment = float(pre[1])
                        find_first_stop_moment = True
                    x_first_stop_time += time_diff
                elif find_first_stop_moment:
                    break
                pre = now
            x_1st_stop_time_moment_ratio = np.log1p(x_first_stop_time) - np.log1p(x_first_stop_moment)

            pre = point_list[0].split(',')
            for i in range(1, length):
                now = point_list[i].split(',')
                time_diff = float(now[1]) - float(pre[1])
                if float(now[1]) > x_first_stop_moment+x_first_stop_time:
                    if pre[0] == now[0]:
                        if not find_second_stop_moment:
                            x_2st_stop_moment = float(pre[1])
                            find_second_stop_moment = True
                        x_2st_stop_time += time_diff
                    elif find_second_stop_moment:
                        break
                pre = now

            x_2_1st_stop_moment_ratio = np.log1p(x_2st_stop_moment) - np.log1p(x_first_stop_moment)
            x_2_1st_stop_time_ratio = np.log1p(x_2st_stop_time) - np.log1p(x_first_stop_time)


            yt_line = m_file[1].readline()
            cols = yt_line.split(' ')
            point_list = cols[1].split(';')
            length = len(point_list) - 1
            pre = point_list[0].split(',')
            last_point = point_list[length-1].split(',')
            last_time = float(last_point[1])
            y_first_stop_moment = last_time
            y_first_stop_time = 0.0
            y_1st_stop_time_moment_ratio = 0.0
            find_first_stop_moment = False
            lowst_y = float(pre[0])
            lowst_y_moment = float(pre[1])

            for i in range(1, length):
                now = point_list[i].split(',')
                time_diff = float(now[1]) - float(pre[1])
                if pre[0] == now[0]:
                    if not find_first_stop_moment:
                        y_first_stop_moment = float(pre[1])
                        find_first_stop_moment = True
                    y_first_stop_time += time_diff
                elif find_first_stop_moment:
                    break
                pre = now
            y_1st_stop_time_moment_ratio = np.log1p(y_first_stop_time) - np.log1p(y_first_stop_moment)

            for i in range(1, length):
                now = point_list[i].split(',')
                y = float(now[0])
                if y < lowst_y:
                    lowst_y = y
                    lowst_y_moment = float(now[1])

            m_file[2].write(cols[0]+','+str(x_first_stop_moment)+','+str(x_first_stop_time)+','+str(x_1st_stop_time_moment_ratio)+','+
                            str(x_2st_stop_moment)+','+str(x_2st_stop_time)+','+str(x_2_1st_stop_moment_ratio)
                            +','+str(x_2_1st_stop_time_ratio)+','+str(y_first_stop_moment)+','+str(y_first_stop_time)+
                            ','+str(y_1st_stop_time_moment_ratio)+','+str(lowst_y_moment)+'\n')
        m_file[0].close()
        m_file[1].close()
        m_file[2].close()


if __name__ == '__main__':
    generate_stop_feature()
