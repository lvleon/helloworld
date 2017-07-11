# -*- coding: UTF-8 -*-

import pandas as pd


def generate_x_speed_feature():
    white_x_speed_file = open('../../data/cache_data/speed/white_x_speed_data.txt')
    black_x_speed_file = open('../../data/cache_data/speed/black_x_speed_data.txt')
    predict_x_speed_file = open('../../data/cache_data/speed/predict_x_speed_data.txt')

    white_x_speed_feature_file = open('../../data/speed_feature/white_x_speed_feature_file.csv', 'w')
    black_x_speed_feature_file = open('../../data/speed_feature/black_x_speed_feature_file.csv', 'w')
    predict_x_speed_feature_file = open('../../data/speed_feature/predict_x_speed_feature_file.csv', 'w')

    file_list = [(white_x_speed_file, white_x_speed_feature_file), (black_x_speed_file, black_x_speed_feature_file), (predict_x_speed_file, predict_x_speed_feature_file)]
    for file_pair in file_list:
        file_pair[1].write('id,x_speed_max,x_speed_min,x_speed_range,first_x_speed,'
                                 'x_speed_var,x_acc_max,x_acc_min,x_acc_range\n')
        for line in file_pair[0].readlines():
            cols = line.split(' ')
            speedts = cols[1].split(';')
            length = len(speedts) - 1
            first_speedt = speedts[0].split(',')
            min_speed = float(first_speedt[0])
            max_speed = min_speed
            first_no_zero_speed = min_speed
            if length == 1:
                file_pair[1].write(
                    cols[0] + ',' + str(max_speed) + ',' + str(min_speed) + ',' + str(0.0) + ',' +
                    str(first_no_zero_speed) + ',' + str(0.0) + ',' + str(0.0) + ',' + str(0.0) + ','
                    + str(0.0) + '\n')
                continue
            speed_list = [min_speed]
            speed_sum = min_speed
            count = 1
            second_speedt = speedts[1].split(',')
            min_acc = (float(second_speedt[0])-float(first_speedt[0]))/(int(first_speedt[1])+int(second_speedt[1]))*2
            max_acc = min_acc
            var_sum = 0

            for i in range(1, length):
                speedt_before = speedts[i-1].split(',')
                speedt = speedts[i].split(',')
                if int(speedt[1]) < 0:
                    continue
                speed = float(speedt[0])
                speed_list.append(speed)
                if -0.000000000001 < first_no_zero_speed < 0.000000000001:
                    first_no_zero_speed = speed
                speed_sum += speed
                count += 1
                if speed < min_speed:
                    min_speed = speed
                if speed > max_speed:
                    max_speed = speed
                acc = (speed-float(speedt_before[0]))/(int(speedt_before[1])+int(speedt[1]))*2
                if acc < min_acc:
                    min_acc = acc
                if acc > max_acc:
                    max_acc = acc
            speed_range = max_speed - min_speed
            acc_range = max_acc - min_acc
            average_speed = speed_sum/count

            for s in speed_list:
                var_sum += pow(s-average_speed, 2)
            speed_var = var_sum/count
            file_pair[1].write(cols[0]+','+str(max_speed)+','+str(min_speed)+','+str(speed_range)+','+
                                           str(first_no_zero_speed)+','+str(speed_var)+','+str(max_acc)+','+str(min_acc)+','
                                           +str(acc_range)+'\n')
        file_pair[0].close()
        file_pair[1].close()


def generate_y_speed_feature():
    white_y_speed_file = open('../../data/cache_data/speed/white_y_speed_data.txt')
    black_y_speed_file = open('../../data/cache_data/speed/black_y_speed_data.txt')
    predict_y_speed_file = open('../../data/cache_data/speed/predict_y_speed_data.txt')

    white_y_speed_feature_file = open('../../data/speed_feature/white_y_speed_feature_file.csv', 'w')
    black_y_speed_feature_file = open('../../data/speed_feature/black_y_speed_feature_file.csv', 'w')
    predict_y_speed_feature_file = open('../../data/speed_feature/predict_y_speed_feature_file.csv', 'w')

    file_list = [(white_y_speed_file, white_y_speed_feature_file), (black_y_speed_file, black_y_speed_feature_file), (predict_y_speed_file, predict_y_speed_feature_file)]
    for file_pair in file_list:
        file_pair[1].write('id,y_speed_max,y_speed_min,y_speed_range,first_y_speed,y_speed_var,'
                           'y_acc_max,y_acc_min,y_acc_range\n')
        for line in file_pair[0].readlines():
            cols = line.split(' ')
            speedts = cols[1].split(';')
            length = len(speedts) - 1
            first_speedt = speedts[0].split(',')
            min_speed = float(first_speedt[0])
            max_speed = min_speed
            first_no_zero_speed = min_speed
            if length == 1:
                file_pair[1].write(
                    cols[0] + ',' + str(max_speed) + ',' + str(min_speed) + ',' + str(0.0) + ',' +
                    str(first_no_zero_speed) + ',' + str(0.0) + ',' + str(0.0) + ',' + str(0.0) + ','
                    + str(0.0) + '\n')
                continue
            speed_list = [min_speed]
            speed_sum = min_speed
            count = 1
            second_speedt = speedts[1].split(',')
            min_acc = (float(second_speedt[0])-float(first_speedt[0]))/(int(first_speedt[1])+int(second_speedt[1]))*2
            max_acc = min_acc
            var_sum = 0

            for i in range(1, length):
                speedt_before = speedts[i-1].split(',')
                speedt = speedts[i].split(',')
                if int(speedt[1]) < 0:
                    continue
                speed = float(speedt[0])
                speed_list.append(speed)
                if -0.000000000001 < first_no_zero_speed < 0.000000000001:
                    first_no_zero_speed = speed
                speed_sum += speed
                count += 1
                if speed < min_speed:
                    min_speed = speed
                if speed > max_speed:
                    max_speed = speed
                acc = (speed-float(speedt_before[0]))/(int(speedt_before[1])+int(speedt[1]))*2
                if acc < min_acc:
                    min_acc = acc
                if acc > max_acc:
                    max_acc = acc
            speed_range = max_speed - min_speed
            acc_range = max_acc - min_acc
            average_speed = speed_sum/count

            for s in speed_list:
                var_sum += pow(s-average_speed, 2)
            speed_var = var_sum/count
            file_pair[1].write(cols[0]+','+str(max_speed)+','+str(min_speed)+','+str(speed_range)+','+
                                           str(first_no_zero_speed)+','+str(speed_var)+','+str(max_acc)+','+str(min_acc)+','
                                           +str(acc_range)+'\n')
        file_pair[0].close()
        file_pair[1].close()


def generate_xy_speed_feature():
    white_xy_speed_file = open('../../data/cache_data/speed/white_xy_speed_data.txt')
    black_xy_speed_file = open('../../data/cache_data/speed/black_xy_speed_data.txt')
    predict_xy_speed_file = open('../../data/cache_data/speed/predict_xy_speed_data.txt')

    white_xy_speed_feature_file = open('../../data/speed_feature/white_xy_speed_feature_file.csv', 'w')
    black_xy_speed_feature_file = open('../../data/speed_feature/black_xy_speed_feature_file.csv', 'w')
    predict_xy_speed_feature_file = open('../../data/speed_feature/predict_xy_speed_feature_file.csv', 'w')

    file_list = [(white_xy_speed_file, white_xy_speed_feature_file), (black_xy_speed_file, black_xy_speed_feature_file), (predict_xy_speed_file, predict_xy_speed_feature_file)]
    for file_pair in file_list:
        file_pair[1].write('id,xy_speed_max,xy_speed_min,xy_speed_range,xy_speed_var\n')
        for line in file_pair[0].readlines():
            cols = line.split(' ')
            speedts = cols[1].split(';')
            length = len(speedts) - 1
            first_speedt = speedts[0].split(',')
            min_speed = float(first_speedt[0])
            max_speed = min_speed
            if length == 1:
                file_pair[1].write(
                    cols[0] + ',' + str(max_speed) + ',' + str(min_speed) + ',' + str(0.0) + ',' + str(0.0) + '\n')
                continue
            speed_list = [min_speed]
            speed_sum = min_speed
            count = 1
            var_sum = 0

            for i in range(1, length):
                speedt = speedts[i].split(',')
                if int(speedt[1]) < 0:
                    continue
                speed = float(speedt[0])
                speed_list.append(speed)
                speed_sum += speed
                count += 1
                if speed < min_speed:
                    min_speed = speed
                if speed > max_speed:
                    max_speed = speed
            speed_range = max_speed - min_speed
            average_speed = speed_sum/count

            for s in speed_list:
                var_sum += pow(s-average_speed, 2)
            speed_var = var_sum/count
            file_pair[1].write(cols[0]+','+str(max_speed)+','+str(min_speed)+','+str(speed_range)+','+ str(speed_var)+'\n')
        file_pair[0].close()
        file_pair[1].close()


def merge():
    white_x_speed_feature_file = pd.read_csv('../../data/speed_feature/white_x_speed_feature_file.csv')
    black_x_speed_feature_file = pd.read_csv('../../data/speed_feature/black_x_speed_feature_file.csv')
    predict_x_speed_feature_file = pd.read_csv('../../data/speed_feature/predict_x_speed_feature_file.csv')
    white_y_speed_feature_file = pd.read_csv('../../data/speed_feature/white_y_speed_feature_file.csv')
    black_y_speed_feature_file = pd.read_csv('../../data/speed_feature/black_y_speed_feature_file.csv')
    predict_y_speed_feature_file = pd.read_csv('../../data/speed_feature/predict_y_speed_feature_file.csv')
    white_xy_speed_feature_file = pd.read_csv('../../data/speed_feature/white_xy_speed_feature_file.csv')
    black_xy_speed_feature_file = pd.read_csv('../../data/speed_feature/black_xy_speed_feature_file.csv')
    predict_xy_speed_feature_file = pd.read_csv('../../data/speed_feature/predict_xy_speed_feature_file.csv')

    white_speed_feature_file = '../../data/speed_feature/white_speed_feature_file.csv'
    black_speed_feature_file = '../../data/speed_feature/black_speed_feature_file.csv'
    predict_speed_feature_file = '../../data/speed_feature/predict_speed_feature_file.csv'

    df_list = [(white_x_speed_feature_file, white_y_speed_feature_file, white_xy_speed_feature_file, white_speed_feature_file),
                 (black_x_speed_feature_file, black_y_speed_feature_file, black_xy_speed_feature_file, black_speed_feature_file),
                 (predict_x_speed_feature_file, predict_y_speed_feature_file, predict_xy_speed_feature_file, predict_speed_feature_file)]
    for m_df in df_list:
        cache_df = pd.merge(m_df[0], m_df[1], how='left', on='id')
        out_df = pd.merge(cache_df, m_df[2], how='left', on='id')
        out_df.to_csv(m_df[3], index=False)


def run():
    generate_x_speed_feature()
    generate_y_speed_feature()
    generate_xy_speed_feature()
    merge()
    pass
run()

