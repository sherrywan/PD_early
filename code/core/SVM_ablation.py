'''
Author: sherrywaan sherrywaan@outlook.com
Date: 2024-06-12 10:45:47
LastEditors: sherrywaan sherrywaan@outlook.com
LastEditTime: 2024-09-25 16:32:45
FilePath: /wxy/PD/PD_early/RF.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

from dataset.pd_dataset import PDDataset


def train_ablation(
    x_train_ori, y_train, save_dir,
    seed = 10,
):

    model = SVC(
            kernel = 'linear',
            C = 0.1,
            probability=True,
            gamma = 0.0001,
            random_state = 12
        )


    # normalize features
    norm_obj = StandardScaler().fit(x_train_ori)
    x_train = norm_obj.transform(x_train_ori)
    # store the normalization object for future use
    model.norm_obj_ = norm_obj

    # convert labels to integers
    y_train = y_train.astype(np.int32)

    # train model
    model.fit(x_train, y_train)
    y_predprob = model.predict_proba(x_train)[:, 1]
    print("AUC Score (Train): %f"%metrics.roc_auc_score(y_train, y_predprob))
    
    print('--------------starting select kernel------------------')
    param_test1 = {'kernel':["rbf", "linear", "sigmoid"]}
    gsearch1 = GridSearchCV(estimator = SVC(C = 0.1,
            probability=True,
            random_state = 12),
                            param_grid = param_test1, scoring='roc_auc', cv=5)
    gsearch1.fit(x_train, y_train)
    print("best_params:", gsearch1.best_params_)
    print("best_score:", gsearch1.best_score_)
    # print("results:", gsearch1.cv_results_)

    # print('--------------starting select degree------------------')
    # param_test3 = {'degree':range(1,5,1)}
    # gsearch3 = GridSearchCV(estimator = SVC(C = 0.1, kernel=gsearch1.best_params_['kernel'],
    #         probability=True,
    #         random_state = 0),
    #                         param_grid = param_test3, scoring='roc_auc', cv=5)
    # gsearch3.fit(x_train, y_train)
    # print("best_params:", gsearch3.best_params_)
    # print("best_score:", gsearch3.best_score_)
    # print("results:", gsearch3.cv_results_)

    print('--------------starting select c------------------')
    param_test2 = {'C':[0.2,0.4,0.6,0.8,1]}
    gsearch2 = GridSearchCV(estimator = SVC( kernel=gsearch1.best_params_['kernel'],
            probability=True,
            random_state = 0),
                            param_grid = param_test2, scoring='roc_auc', cv=5)
    gsearch2.fit(x_train, y_train)
    print("best_params:", gsearch2.best_params_)
    print("best_score:", gsearch2.best_score_)
    # print("results:", gsearch2.cv_results_)

    return gsearch1.best_params_['kernel'], gsearch2.best_params_['C']



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


