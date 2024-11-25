'''
Author: sherrywaan sherrywaan@outlook.com
Date: 2024-06-12 10:45:47
LastEditors: sherrywaan sherrywaan@outlook.com
LastEditTime: 2024-09-24 16:00:58
FilePath: /wxy/PD/PD_early/RF.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

from dataset.pd_dataset import PDDataset


def train_ablation(
    x_train_ori, y_train, save_dir,
    seed = 10,
):

    model = RandomForestClassifier(criterion="entropy",oob_score=True, random_state=seed)


    # normalize features
    norm_obj = StandardScaler().fit(x_train_ori)
    x_train = norm_obj.transform(x_train_ori)
    # store the normalization object for future use
    model.norm_obj_ = norm_obj

    # convert labels to integers
    y_train = y_train.astype(np.int32)

    # train model
    model.fit(x_train, y_train)
    print(model.oob_score_)
    y_predprob = model.predict_proba(x_train)[:, 1]
    print("AUC Score (Train): %f"%metrics.roc_auc_score(y_train, y_predprob))
    
    print('--------------starting select n_estimators------------------')
    param_test1 = {'n_estimators':range(200,800,10)}
    gsearch1 = GridSearchCV(estimator = RandomForestClassifier(criterion="entropy", max_features='sqrt' , oob_score=True, random_state=seed), 
                            param_grid = param_test1, scoring='roc_auc',cv=5)
    gsearch1.fit(x_train, y_train)
    print("best_params:", gsearch1.best_params_)
    print("best_score:", gsearch1.best_score_)
    # print("results:", gsearch1.cv_results_)
    
    print('--------------starting select random_state------------------')
    param_test4 = {'random_state':range(0,100,1)}
    gsearch4 = GridSearchCV(estimator = RandomForestClassifier(criterion="entropy", n_estimators= gsearch1.best_params_['n_estimators'],
                                  max_features='sqrt', oob_score=True),
                            param_grid = param_test4, scoring='roc_auc', cv=5)
    gsearch4.fit(x_train, y_train)
    print("best_params:", gsearch4.best_params_)
    print("best_score:", gsearch4.best_score_)
    # print("results:", gsearch4.cv_results_)

    print('--------------starting select max_depth and min_samples_split------------------')
    param_test2 = {'max_depth':range(5,20,2), 'min_samples_split':range(2, 20, 2)}
    gsearch2 = GridSearchCV(estimator = RandomForestClassifier(criterion="entropy", n_estimators= gsearch1.best_params_['n_estimators'], 
                                  max_features='sqrt' ,oob_score=True, random_state=gsearch4.best_params_['random_state']),
                            param_grid = param_test2, scoring='roc_auc', cv=5)
    gsearch2.fit(x_train, y_train)
    print("best_params:", gsearch2.best_params_)
    print("best_score:", gsearch2.best_score_)
    # print("results:", gsearch2.cv_results_)

    print('--------------starting select min_samples_leaf------------------')
    param_test3 = {'min_samples_leaf':range(2,20,2)}
    gsearch3 = GridSearchCV(estimator = RandomForestClassifier(criterion="entropy", n_estimators= gsearch1.best_params_['n_estimators'], 
                                                               max_depth=gsearch2.best_params_['max_depth'],
                                                               min_samples_split = gsearch2.best_params_['min_samples_split'],
                                  max_features='sqrt', oob_score=True, random_state=gsearch4.best_params_['random_state']),
                            param_grid = param_test3, scoring='roc_auc', cv=5)
    gsearch3.fit(x_train, y_train)
    print("best_params:", gsearch3.best_params_)
    print("best_score:", gsearch3.best_score_)
    # print("results:", gsearch3.cv_results_)

    return gsearch1.best_params_['n_estimators'], gsearch2.best_params_['max_depth'], gsearch2.best_params_['min_samples_split'], gsearch4.best_params_['random_state'], gsearch3.best_params_['min_samples_leaf']


def prepare_datas(data_index, data_features, data_labels, module_types=["face_fix_par", "face_monologue", "blink", "glance", "voice_sus_vowel", "voice_alt_pro",
                                                                        "voice_fix_par", "voice_monologue", "gait",  "finger_tap", "toe_tap"]):
    features = []
    labels = []
    for idx_i, idx in enumerate(data_index):
        f_s = data_features[idx]
        fe_s = []
        for module_type in f_s.keys():
            if module_type in module_types:
                fe_s += f_s[module_type].values()
        features.append(fe_s)
        labels.append(data_labels[idx])
    features = np.array(features)
    labels = np.array(labels)
    return features, labels

    
if __name__ == "__main__":
    save_dir = "/data0/wxy/PD/PD_early/res_80-78"
    file_path = "/data0/wxy/PD/PD_early/data/PD_dataset/127PD-116HC数据分析"
    csv_path = "/data0/wxy/PD/PD_early/data/PD_dataset/subjects_80-78.csv"
    parameter_path = "/data0/wxy/PD/PD_early/data/PD_dataset/parameters_74.json"
    module_types = ["face_fix_par", "gait"]
    pd_dataset = PDDataset(file_path, csv_path, parameter_path)
    # module_types = ["face_fix_par", "face_monologue", "voice_sus_vowel", "gait", "other_mov"]

    train_index, train_features, train_labels = pd_dataset.get_all(is_train=True)
    train_feature_list, train_labels_list = prepare_datas(train_index, train_features, train_labels, module_types=module_types)
    print("feature_shape:", train_feature_list[0].shape)
    train_ablation(train_feature_list, train_labels_list, save_dir=save_dir)


