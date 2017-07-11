# -*- coding: UTF-8 -*-

from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
from gbdt_train import customed_score
from gbdt_train import load_data

# xgb调参
def xgb_param_adjust(train_data, label):

    param_test1 = {
        'max_depth': range(1, 10, 1),
        'min_child_weight': range(1, 6, 1)
    }

    # param_test2 = {
    #     'gamma': [i / 10.0 for i in range(0, 5)]
    # }
    param_test2 = {
        'subsample': [i / 10.0 for i in range(1, 10)],
        'colsample_bytree': [i / 10.0 for i in range(1, 10)]
    }
    # param_test4 = {
    #     'reg_alpha': [0, 1e-5, 1e-2, 0.1, 1, 100]
    # }
    # param_test5 = {
    #     'reg_lambda': [0, 1e-5, 1e-2, 0.1, 1, 2]
    # }
    param_test3 = {
        'subsample': [i / 100.0 for i in range(65, 80, 1)],
        'colsample_bytree': [i / 100.0 for i in range(45, 60, 1)]
    }
    gsearch1 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.01, n_estimators=17, max_depth=6,
                                                    min_child_weight=1, gamma=0, reg_lambda=1, subsample=0.78, colsample_bytree=0.52,
                                                    objective='binary:logistic', nthread=8, scale_pos_weight=1,
                                                    seed=0),
                            param_grid=param_test3, scoring='roc_auc', n_jobs=8, iid=False, cv=5)
    gsearch1.fit(train_data, label)
    print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)


# lgb调参
def lgb_param_adjust(train_data, label):

    param_test1 = {
        'max_depth': range(1, 10, 1),
        'min_child_weight': range(1, 6, 1)
    }

    param_test2 = {
        'gamma': [i / 10.0 for i in range(0, 5)]
    }
    param_test3 = {
        'subsample': [i / 10.0 for i in range(1, 10)],
        'colsample_bytree': [i / 10.0 for i in range(1, 10)]
    }
    param_test4 = {
        'reg_alpha': [0, 1e-5, 1e-2, 0.1, 1, 100]
    }
    param_test5 = {
        'reg_lambda': [0, 1e-5, 1e-2, 0.1, 1, 2]
    }

    gsearch1 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=41, max_depth=4,
                                                    min_child_weight=1, gamma=0, reg_lambda=0, subsample=0.4, colsample_bytree=0.3,
                                                    objective='binary:logistic', nthread=8, scale_pos_weight=1,
                                                    seed=0),
                            param_grid=param_test5, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    gsearch1.fit(train_data, label)
    print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)


def run():
    train_data, label = load_data()
    xgb_param_adjust(train_data, label)

if __name__ == '__main__':
    run()