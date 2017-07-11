# -*- coding: UTF-8 -*-


def generate_x_speed(fin_name, fout_name, predict):
    in_file = open(fin_name)
    out_file = open(fout_name, 'w')
    for line in in_file.readlines():
        cols = line.split(' ')
        xts = cols[1].split(';')
        length = len(xts)-1
        pre_xt = xts[0].split(',')
        speeds = str()
        for i in range(1, length):
            xt = xts[i].split(',')
            t_diff = int(xt[1]) - int(pre_xt[1])
            if t_diff:
                speed = (float(xt[0]) - float(pre_xt[0])) / t_diff
                speeds += str(speed)+','+str(t_diff)+';'
            pre_xt = xt
        if predict:
            out_file.write(cols[0] + ' ' + speeds + ' ' + cols[2])
        else:
            out_file.write(cols[0]+' '+speeds+' '+cols[2]+' '+cols[3])
    in_file.close()
    out_file.close()


def generate_y_speed(fin_name, fout_name, predict):
    in_file = open(fin_name)
    out_file = open(fout_name, 'w')
    for line in in_file.readlines():
        cols = line.split(' ')
        yts = cols[1].split(';')
        length = len(yts)-1
        pre_yt = yts[0].split(',')
        speeds = str()
        for i in range(1, length):
            yt = yts[i].split(',')
            t_diff = int(yt[1]) - int(pre_yt[1])
            if t_diff:
                speed = (float(yt[0]) - float(pre_yt[0])) / t_diff
                speeds += str(speed)+','+str(t_diff)+';'
            pre_yt = yt
        if predict:
            out_file.write(cols[0] + ' ' + speeds + ' ' + cols[2])
        else:
            out_file.write(cols[0]+' '+speeds+' '+cols[2]+' '+cols[3])
    in_file.close()
    out_file.close()


def generate_xy_speed(fin_name, fout_name, predict):
    in_file = open(fin_name)
    out_file = open(fout_name, 'w')
    for line in in_file.readlines():
        cols = line.split(' ')
        xyts = cols[1].split(';')
        length = len(xyts)-1
        pre_xyt = xyts[0].split(',')
        speeds = str()
        for i in range(1, length):
            xyt = xyts[i].split(',')
            t_diff = int(xyt[2]) - int(pre_xyt[2])
            x_diff = float(xyt[0]) - float(pre_xyt[0])
            y_diff = float(xyt[1]) - float(pre_xyt[1])
            distance = pow(x_diff*x_diff+y_diff*y_diff, 0.5)
            if t_diff:
                speed = distance / t_diff
                speeds += str(speed)+','+str(t_diff)+';'
            pre_xyt = xyt
        if predict:
            out_file.write(cols[0] + ' ' + speeds + ' ' + cols[2])
        else:
            out_file.write(cols[0]+' '+speeds+' '+cols[2]+' '+cols[3])
    in_file.close()
    out_file.close()


def run():
    generate_x_speed('../../data/cache_data/white_xt_data.txt', '../../data/cache_data/speed/white_x_speed_data.txt', 0)
    generate_y_speed('../../data/cache_data/white_yt_data.txt', '../../data/cache_data/speed/white_y_speed_data.txt', 0)
    generate_xy_speed('../../data/raw_data/point_than_2_white_data.txt', '../../data/cache_data/speed/white_xy_speed_data.txt', 0)

    generate_x_speed('../../data/cache_data/black_xt_data.txt', '../../data/cache_data/speed/black_x_speed_data.txt', 0)
    generate_y_speed('../../data/cache_data/black_yt_data.txt', '../../data/cache_data/speed/black_y_speed_data.txt', 0)
    generate_xy_speed('../../data/raw_data/point_than_2_black_data.txt', '../../data/cache_data/speed/black_xy_speed_data.txt', 0)

    generate_x_speed('../../data/cache_data/predict_xt_data.txt', '../../data/cache_data/speed/predict_x_speed_data.txt', 1)
    generate_y_speed('../../data/cache_data/predict_yt_data.txt', '../../data/cache_data/speed/predict_y_speed_data.txt', 1)
    generate_xy_speed('../../data/raw_data/point_than_2_predict_data.txt', '../../data/cache_data/speed/predict_xy_speed_data.txt', 1)
    pass

run()