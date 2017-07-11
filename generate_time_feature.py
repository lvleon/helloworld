# -*- coding: UTF-8 -*-


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

    white_stop_feature_file = open('../../data/time_feature/white_time_feature_file.csv', 'w')
    black_stop_feature_file = open('../../data/time_feature/black_time_feature_file.csv', 'w')
    predict_stop_feature_file = open('../../data/time_feature/predict_time_feature_file.csv', 'w')

    file_list = [(white_xt_file, white_yt_file, white_file, white_stop_feature_file),
                 (black_xt_file, black_yt_file, black_file, black_stop_feature_file),
                 (predict_xt_file, predict_yt_file, predict_file, predict_stop_feature_file)]

    for m_file in file_list:
        m_file[3].write('id,sample_time,x_stop_time,x_max_stop_time,x_stop_time_ratio,y_stop_time,y_max_stop_time,'
                        'y_stop_time_ratio,xy_stop_time,xy_max_stop_time,xy_stop_time_ratio,first_stop_time,aimed_and_sample_time_ratio\n')
        for line in m_file[0].readlines():
            cols = line.split(' ')
            point_list = cols[1].split(';')
            length = len(point_list) - 1
            pre = point_list[0].split(',')
            begin_time = float(pre[-1])
            stop = False
            x_max_stop_time = 0.0
            x_stop_time = 0.0
            last_point = point_list[length-1].split(',')
            sample_time = float(last_point[-1]) - begin_time
            stop_time = 0.0
            for i in range(1, length):
                now = point_list[i].split(',')
                time_diff = float(now[1]) - float(pre[1])
                if pre[0] == now[0]:
                    stop_time += time_diff
                    stop = True
                elif stop:
                    x_stop_time += stop_time
                    if stop_time > x_max_stop_time:
                        x_max_stop_time = stop_time
                    stop_time = 0.0
                    stop = False
                pre = now
            if stop:
                x_stop_time += stop_time
                if stop_time > x_max_stop_time:
                    x_max_stop_time = stop_time
            x_stop_time_ratio = x_stop_time/sample_time

            yt_line = m_file[1].readline()
            cols = yt_line.split(' ')
            point_list = cols[1].split(';')
            length = len(point_list) - 1
            pre = point_list[0].split(',')
            stop = False
            y_max_stop_time = 0.0
            y_stop_time = 0.0
            stop_time = 0.0
            for i in range(1, length):
                now = point_list[i].split(',')
                time_diff = float(now[1]) - float(pre[1])
                if pre[0] == now[0]:
                    stop_time += time_diff
                    stop = True
                elif stop:
                    y_stop_time += stop_time
                    if stop_time > y_max_stop_time:
                        y_max_stop_time = stop_time
                    stop_time = 0.0
                    stop = False
                pre = now
            if stop:
                y_stop_time += stop_time
                if stop_time > y_max_stop_time:
                    y_max_stop_time = stop_time
            y_stop_time_ratio = y_stop_time / sample_time

            xyt_line = m_file[2].readline()
            cols = xyt_line.split(' ')
            point_list = cols[1].split(';')
            length = len(point_list) - 1
            pre = point_list[0].split(',')
            stop = False
            xy_max_stop_time = 0.0
            xy_stop_time = 0.0
            stop_time = 0.0
            for i in range(1, length):
                now = point_list[i].split(',')
                time_diff = float(now[2]) - float(pre[2])
                if pre[0] == now[0] and pre[1] == now[1]:
                    stop_time += time_diff
                    stop = True
                elif stop:
                    xy_stop_time += stop_time
                    if stop_time > xy_max_stop_time:
                        xy_max_stop_time = stop_time
                    stop_time = 0.0
                    stop = False
                pre = now
            if stop:
                xy_stop_time += stop_time
                if stop_time > xy_max_stop_time:
                    xy_max_stop_time = stop_time
            xy_stop_time_ratio = xy_stop_time / sample_time

            m_file[3].write(cols[0]+','+str(sample_time)+','+str(x_stop_time)+','+str(x_max_stop_time)+','+
                            str(x_stop_time_ratio)+','+str(y_stop_time)+','+str(y_max_stop_time)
                            +','+str(y_stop_time_ratio)+','+str(xy_stop_time)+','+str(xy_max_stop_time)+
                            ','+str(xy_stop_time_ratio)+'\n')
        m_file[0].close()
        m_file[1].close()
        m_file[2].close()
        m_file[3].close()


def generate_first_stop_time():

    white_file = open('../../data/raw_data/point_than_2_white_data.txt')
    black_file = open('../../data/raw_data/point_than_2_black_data.txt')
    predict_file = open('../../data/raw_data/point_than_2_predict_data.txt')

    white_time_feature_file = '../../data/time_feature/white_time_feature_file.csv'
    black_time_feature_file = '../../data/time_feature/black_time_feature_file.csv'
    predict_time_feature_file = '../../data/time_feature/predict_time_feature_file.csv'

    file_list = [(white_file, white_time_feature_file), (black_file, black_time_feature_file), (predict_file, predict_time_feature_file)]

    for m_file in file_list:
        ss = []
        out_file = open(m_file[1])
        ss.append(out_file.readline())
        for line in m_file[0].readlines():
            first_stop_time = 0.0
            cols = line.split(' ')
            point_list = cols[1].split(';')
            length = len(point_list) - 1
            pre = point_list[0].split(',')
            stop = False
            for i in range(1, length):
                now = point_list[i].split(',')
                time_diff = float(now[2]) - float(pre[2])
                if (pre[0] != now[0] or pre[1] != now[1]) and not stop:
                    break
                if pre[0] == now[0] and pre[1] == now[1]:
                    first_stop_time += time_diff
                    stop = True
                elif stop:
                    break
                pre = now
            sub_ss = out_file.readline()
            sub_ss_list = sub_ss.split('\n')
            new_ss = sub_ss_list[0]+','+str(first_stop_time)+'\n'
            ss.append(new_ss)
        m_file[0].close()
        out_file.close()
        out_file = open(m_file[1], 'w')
        for s in ss:
            out_file.write(s)
        out_file.close()


def generate_aimed_and_sample_time_ratio():
    white_xt_file = open('../../data/cache_data/white_xt_data.txt')
    black_xt_file = open('../../data/cache_data/black_xt_data.txt')
    predict_xt_file = open('../../data/cache_data/predict_xt_data.txt')

    white_time_feature_file = '../../data/time_feature/white_time_feature_file.csv'
    black_time_feature_file = '../../data/time_feature/black_time_feature_file.csv'
    predict_time_feature_file = '../../data/time_feature/predict_time_feature_file.csv'

    file_list = [(white_xt_file, white_time_feature_file), (black_xt_file, black_time_feature_file), (predict_xt_file, predict_time_feature_file)]

    for m_file in file_list:
        ss = []
        out_file = open(m_file[1])
        ss.append(out_file.readline())
        for line in m_file[0].readlines():
            cols = line.split(' ')
            point_list = cols[1].split(';')
            length = len(point_list) - 1
            pre = point_list[0].split(',')
            begin_x = float(pre[0])
            begin_time = float(pre[1])
            aim_x = float(cols[2])
            # 是否正向，即目标位于右侧
            direction = True
            if begin_x > aim_x:
                direction = False
            aimed_and_sample_time_ratio = 0.0
            last_point = point_list[length-1].split(',')
            sample_time = float(last_point[1]) - begin_time
            # 是否到达目标
            aimed = False
            # 到达目标之前那个点与目标的距离
            close_distance_1 = abs(aim_x - float(pre[0]))
            # 到达目标之后那个点与目标的距离
            close_distance_2 = abs(aim_x - float(pre[0]))
            # 到达目标之前那个点所用时间
            close_time_1 = 0.0
            # 到达目标之后那个点所用时间
            close_time_2 = 0.0
            for i in range(1, length):
                now = point_list[i].split(',')
                if not aimed:
                    if direction:
                        if float(now[0]) < aim_x:
                            distance = aim_x - float(now[0])
                            if distance < close_distance_1:
                                close_distance_1 = distance
                                close_time_1 = float(now[1]) - begin_time
                        else:
                            aimed = True
                            distance = float(now[0]) - aim_x
                            if distance < close_distance_2:
                                close_distance_2 = distance
                                close_time_2 = float(now[1]) - begin_time
                            break
                    else:
                        if float(now[0]) > aim_x:
                            distance = float(now[0]) - aim_x
                            if distance < close_distance_1:
                                close_distance_1 = distance
                                close_time_1 = float(now[1]) - begin_time
                        else:
                            aimed =True
                            distance = aim_x - float(now[0])
                            if distance < close_distance_2:
                                close_distance_2 = distance
                                close_time_2 = float(now[1]) - begin_time
                            break
                pre = now
            if close_distance_1 < close_distance_2:
                aimed_and_sample_time_ratio = close_time_1 / sample_time
            else:
                aimed_and_sample_time_ratio = close_time_2 / sample_time
            sub_ss = out_file.readline()
            sub_ss_list = sub_ss.split('\n')
            new_ss = sub_ss_list[0]+','+str(aimed_and_sample_time_ratio)+'\n'
            ss.append(new_ss)
        m_file[0].close()
        out_file.close()
        out_file = open(m_file[1], 'w')
        for s in ss:
            out_file.write(s)
        out_file.close()
    pass


def run():
    generate_stop_feature()
    generate_first_stop_time()
    generate_aimed_and_sample_time_ratio()
    pass

run()