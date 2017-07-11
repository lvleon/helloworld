# -*- coding: UTF-8 -*-


# 去掉训练集和预测集中去重后小于等于2个采样点的记录
def delete_too_short_line():
    xy_unrepeated_white_file = open('../data/raw_data/xy_unrepeated_white_data.txt')
    xy_unrepeated_black_file = open('../data/raw_data/xy_unrepeated_black_data.txt')
    xy_unrepeated_predict_file = open('../data/raw_data/xy_unrepeated_predict_data.txt')

    raw_white_file = open('../data/raw_data/raw_white_data.txt')
    raw_black_file = open('../data/raw_data/raw_black_data.txt')
    raw_predict_file = open('../data/raw_data/dsjtzs_txfz_test1.txt')

    point_than_2_white_file = open('../data/raw_data/point_than_2_white_data.txt', 'w')
    point_than_2_black_file = open('../data/raw_data/point_than_2_black_data.txt', 'w')
    point_than_2_predict_file = open('../data/raw_data/point_than_2_predict_data.txt', 'w')

    file_list = [(xy_unrepeated_white_file, raw_white_file, point_than_2_white_file),
                 (xy_unrepeated_black_file, raw_black_file, point_than_2_black_file),
                 (xy_unrepeated_predict_file, raw_predict_file, point_than_2_predict_file)]

    for m_file in file_list:
        for line in m_file[0].readlines():
            cols = line.split(' ')
            cols1 = cols[1].split(';')
            sample_num = len(cols1) - 1
            if sample_num > 2:
                m_file[2].write(m_file[1].readline())
            else:
                m_file[1].readline()
        m_file[0].close()
        m_file[1].close()
        m_file[2].close()


# 将记录按时间排序
def sort_time(file_name, predict):
    ss = []
    original = open(file_name)
    for line in original.readlines():
        cols = line.split(' ')
        point_list = cols[1].split(';')
        length = len(point_list) - 1
        my_list = []
        for i in range(0, length):
            xyt = point_list[i].split(',')
            t = int(xyt[2])
            my_list.append((xyt[0], xyt[1], t))
        my_list = sorted(my_list, key=lambda p:p[2])
        s = cols[0] + ' '
        for point in my_list:
            s += point[0]+','+point[1]+','+str(point[2])+';'
        if predict:
            s += ' '+cols[2]
        else:
            s += ' '+cols[2]+' '+cols[3]
        ss.append(s)
    original.close()
    new = open(file_name, 'w')
    for string in ss:
        new.write(string)
    new.close()


def run():
    delete_too_short_line()

    point_than_2_white_file = '../data/raw_data/point_than_2_white_data.txt'
    point_than_2_black_file = '../data/raw_data/point_than_2_black_data.txt'
    point_than_2_predict_file = '../data/raw_data/point_than_2_predict_data.txt'
    sort_time(point_than_2_white_file, 0)
    sort_time(point_than_2_black_file, 0)
    sort_time(point_than_2_predict_file, 1)

run()