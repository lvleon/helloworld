# -*- coding: UTF-8 -*-
import pandas as pd
import matplotlib.pyplot as plt

white_sample = pd.read_csv('../data/white_train_feature_plus_label.csv')
black_sample = pd.read_csv('../data/black_train_feature_plus_label.csv')

black_speed_var_feature_file = pd.read_csv('../data/added_feature/black_speed_var_feature.csv')

white_speed_var_feature_file = pd.read_csv('../data/added_feature/white_speed_var_feature.csv')

white_sample = pd.merge(white_sample, white_speed_var_feature_file, how='left', on='id')
black_sample = pd.merge(black_sample, black_speed_var_feature_file, how='left', on='id')

del white_sample['id']
del black_sample['id']

for feature in white_sample.columns:
    ax1 = plt.subplot(211)
    white_sample[feature].plot(color='r')
    plt.legend(loc='best')
    plt.title('Positive')
    plt.grid()
    plt.sca(ax1)
    ax2 = plt.subplot(212)
    black_sample[feature].plot(color='b')
    plt.legend(loc='best')
    plt.title('Negative')
    plt.grid()
    plt.sca(ax2)
    plt.savefig('../figtures/feature/%s.png' %(feature))
    plt.clf()
    plt.close()