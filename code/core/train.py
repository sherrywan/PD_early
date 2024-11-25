'''
Author: sherrywaan sherrywaan@outlook.com
Date: 2024-06-12 10:45:47
LastEditors: sherrywaan sherrywaan@outlook.com
LastEditTime: 2024-09-24 17:30:42
FilePath: /wxy/PD/PD_early/RF.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from core.MLP_module import MLPClassifier as MLP
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import joblib

from dataset.pd_dataset import PDDataset


def train(
    cross_set, save_dir, config, model_type="RF", modality=0, cross_i=0, time_t=0
):
    '''
    once-training for one-model
    parameters:
        cross_set: (x_train,y_train,x_test,y_test)
        config: model hyper-parameters
        modality: modality included
        cross_i: the index of cross validation experiments, default: 0
        time_t: the index of experiments repeat, default: 0
    '''

    assert model_type in ['RF', 'LR', 'SVM', 'MLP'], "model type is not supported"
    print("-----------train {} model with cross val: {}th cross, {}th iter -----------".format(model_type, cross_i, time_t))
    
    if model_type == "RF":
        model = RandomForestClassifier(criterion="entropy", 
                                       n_estimators=config.RF.n_estimators if hasattr(config.RF, "n_estimators") else 500, 
                                       max_depth=config.RF.max_depth if hasattr(config.RF, "max_depth") else 11, 
                                       min_samples_split=config.RF.min_samples_split if hasattr(config.RF, "min_samples_split") else 2, 
                                       bootstrap=False, 
                                       random_state=config.RF.random_state if hasattr(config.RF, "random_state") else 10, 
                                       min_samples_leaf=config.RF.min_samples_leaf if hasattr(config.RF, "min_samples_leaf") else 4)
    elif model_type == "LR":
        model = LogisticRegression(
            C =config.LR.C if hasattr(config.LR, "C") else 0.4,
            penalty =config.LR.penalty if hasattr(config.LR, "penalty") else 'l2',
            solver =config.LR.solver if hasattr(config.LR, "solver") else 'sag',
            random_state = 0,
            max_iter=1000
        )
    elif model_type == "MLP":
        model = MLP(
            hidden_dims = config.MLP.hidden_dims if hasattr(config.MLP, "hidden_dims") else (32,),
            num_epochs = config.MLP.num_epoches if hasattr(config.MLP, "num_epoches") else 32,
            batch_size = 16,
            device = 'cpu',
            lambda_l1= config.MLP.lambda_l1 if hasattr(config.MLP, "lambda_l1") else 0.01,
            lambda_l2= config.MLP.lambda_l2 if hasattr(config.MLP, "lambda_l2") else 0.00001
        )
    elif model_type == 'SVM':
        model = SVC(
            kernel = config.SVM.kernel if hasattr(config.SVM, "kernel") else 'linear',
            C = config.SVM.C if hasattr(config.SVM, "C") else 0.2,
            probability=True,
            random_state = 0
        )

    (x_train,y_train,x_test,y_test) = cross_set
    train_main(model, x_train, y_train)
    # save model checkpoint
    checkpoint_path = '{}/checkpoints/train_{}_{}modal_{}iter_{}crossval.ckpt'.format(save_dir,model_type,modality,time_t,cross_i)
    joblib.dump(model, checkpoint_path)
    # test and save test results
    test_result_path = '{}/preds/train_{}_{}modal_{}iter_{}crossval.csv'.format(save_dir,model_type,modality,time_t,cross_i)
    test(x_test, y_test, checkpoint_path, test_result_path)


def train_main(model, x_train,y_train):
    # normalize features
    norm_obj = StandardScaler().fit(x_train)
    x_train = norm_obj.transform(x_train)
    # store the normalization object for future use
    model.norm_obj_ = norm_obj
    # convert labels to integers
    y_train = y_train.astype(np.int32)
    # train model
    model.fit(x_train, y_train)
    y_predprob = model.predict_proba(x_train)[:, 1]
    print("AUC Score (Train): %f"%metrics.roc_auc_score(y_train, y_predprob))


def test(
    x_test, y_test, model_path, test_result_path
):
    model = joblib.load(model_path)    
    # normalize features
    norm_obj =  model.norm_obj_
    x_test = norm_obj.transform(x_test)
    # convert labels to integers
    y_test = y_test.astype(np.int32)
    # test
    y_predprob = model.predict_proba(x_test)[:, 1]
    print("AUC Score (Test): %f"%metrics.roc_auc_score(y_test, y_predprob))
    y_pred = model.predict(x_test)
    # save
    save_results(y_test, y_pred, y_predprob, test_result_path)
    

def save_results(labels, predictions, scores, save_path):
    df_results = pd.DataFrame()
    df_results['Y Label'] = labels
    df_results['Y Predicted'] = predictions
    df_results['Predicted score'] = scores
    df_results.to_csv(save_path, index=False)     


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
    save_dir = "/data0/wxy/PD/PD_early/res_126-113"
    file_path = "/data0/wxy/PD/PD_early/data/PD_dataset/127PD-116HC数据_缺省已填充_mean"
    csv_path = "/data0/wxy/PD/PD_early/data/PD_dataset/subjects_126-113.xlsx"
    parameter_path = "/data0/wxy/PD/PD_early/data/PD_dataset/parameters_80.json"
    model_type = ['RF', 'LR', 'SVM', 'MLP']
    # module_types = ["face_monologue", "face_fix_par", "gait"] 
    module_types = ["face_fix_par", "face_monologue", "blink", "glance", "voice_sus_vowel", "voice_alt_pro",
                                                                        "voice_fix_par", "voice_monologue", "gait",  "finger_tap", "toe_tap"]
    cross_val_nums=10

    # # prepare dataset
    # pd_dataset = PDDataset(file_path, csv_path, parameter_path)

    # train_index, train_features, train_labels = pd_dataset.get_all(is_train=True)
    # train_feature_list, train_labels_list = prepare_datas(train_index, train_features, train_labels, module_types=module_types)

    # test_index, test_features, test_labels = pd_dataset.get_all(is_train=False)
    # test_feature_list, test_labels_list = prepare_datas(test_index, test_features, test_labels, module_types=module_types)
    # print("feature_shape:", train_feature_list[0].shape)

    # # train model
    # for model_t in model_type:
    #     print("-----------train {} model-----------".format(model_t))
    #     if model_t == "MLP":
    #         for time_t in range(10):
    #             model_path = train(train_feature_list, train_labels_list, save_dir=save_dir, model_type=model_t, cross_val=False, time_t=time_t, modality=module_types)
    #             test_result_path = '{}/preds/test_{}_{}modal_{}iter.csv'.format(save_dir,model_t, module_types, time_t)
    #             test(test_feature_list, test_labels_list, model_path, test_result_path)
    #     else:
    #         model_path = train(train_feature_list, train_labels_list, save_dir=save_dir, model_type=model_t, cross_val=False, cross_train=False, modality=module_types)
    #         test_result_path = '{}/preds/test_{}_{}modal_0iter.csv'.format(save_dir,model_t, module_types)
    #         test(test_feature_list, test_labels_list, model_path, test_result_path)

    # train model with cross_val (cross x-fold in whole set)
    cross_set = []
    for cross_idx in range(cross_val_nums):
        pd_dataset = PDDataset(file_path, csv_path, parameter_path, start_idx=cross_idx)
        train_index, train_features, train_labels = pd_dataset.get_all(is_train=True)
        train_feature_list, train_labels_list = prepare_datas(train_index, train_features, train_labels, module_types=module_types)
        test_index, test_features, test_labels = pd_dataset.get_all(is_train=False)
        test_feature_list, test_labels_list = prepare_datas(test_index, test_features, test_labels, module_types=module_types)
        datas = (train_feature_list, train_labels_list, test_feature_list, test_labels_list)
        cross_set.append(datas)
    
    for model_t in model_type:
        if model_t == "MLP":
            for time_t in range(10):
                model_path = train(train_feature_list, train_labels_list, save_dir=save_dir, model_type=model_t, cross_val=True, cross_set=cross_set, modality=module_types, time_t=time_t)
        else:
            model_path = train(train_feature_list, train_labels_list, save_dir=save_dir, model_type=model_t, cross_val=True, cross_set=cross_set, modality=module_types)
        