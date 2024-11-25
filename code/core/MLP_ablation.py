'''
Author: sherrywaan sherrywaan@outlook.com
Date: 2024-06-12 10:45:47
LastEditors: sherrywaan sherrywaan@outlook.com
LastEditTime: 2024-09-26 10:51:35
FilePath: /wxy/PD/PD_early/RF.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import joblib

from core.MLP_module import MLPClassifier as MLP


def train(
    x_train_ori, y_train_ori, save_dir, model_type="RF", cross_val=True, cross_nums = 10, modality=0, hidden_dim=64, n_epoch=64, l1_weight=0.001, l2_weight=0.001
):
    assert model_type in ['RF', 'LR', 'SVM', 'MLP'], "model type is not supported"

    if model_type == "RF":
        model = RandomForestClassifier(criterion="entropy", 
                                       n_estimators=120, 
                                       max_depth=7, 
                                       min_samples_split=8, 
                                       bootstrap=False, 
                                       random_state=12, 
                                       min_samples_leaf=2)
    elif model_type == "LR":
        model = LogisticRegression(
            C = 241,
            penalty = 'l1',
            solver = 'liblinear',
            random_state = 12
        )
    elif model_type == "MLP":
        model = MLP(
            hidden_dims = hidden_dim,
            num_epochs = n_epoch,
            batch_size = 16,
            device = 'cpu',
            lambda_l1=l1_weight,
            lambda_l2=l2_weight
        )
    elif model_type == 'SVM':
        model = SVC(
            kernel = 'linear',
            C = 0.1,
            probability=True,
            gamma = 0.0001,
            random_state = 12
        )
    
    # train
    if cross_val:
        # split dataset
        kf = KFold(n_splits=cross_nums, random_state=cross_nums, shuffle=True)
        for i, (train_index, test_index) in enumerate(kf.split(x_train_ori)):
            print("-----------train {} model with {}-cross val: {}th cross-----------".format("MLP", cross_nums, i))
            x_train = x_train_ori[train_index]
            y_train = y_train_ori[train_index]
            x_test = x_train_ori[test_index]
            y_test = y_train_ori[test_index]
            train_main(model, x_train, y_train)
            # save model checkpoint
            checkpoint_path = '{}/checkpoints/train_{}_crossval_{}.ckpt'.format(save_dir,model_type,i)
            joblib.dump(model, checkpoint_path)
            test_result_path = '{}/preds/train_{}_crossval_{}.csv'.format(save_dir,model_type,i)
            test(x_test, y_test, checkpoint_path, test_result_path)

    else:
        train_main(model, x_train_ori, y_train_ori)
        # save model checkpoint
        if modality==0:
            checkpoint_path = '{}/checkpoints/train_{}.ckpt'.format(save_dir,model_type)
        else:
            checkpoint_path = '{}/checkpoints/train_{}_{}modal.ckpt'.format(save_dir,model_type,modality)
        joblib.dump(model, checkpoint_path)

    return checkpoint_path


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
    # print("AUC Score (Train): %f"%metrics.roc_auc_score(y_train, y_predprob))


def test(
    x_test, y_test, model_path, test_result_path
):
    model = joblib.load(model_path)
    
    # normalize features
    norm_obj = StandardScaler().fit(x_test)
    x_test = norm_obj.transform(x_test)
    # store the normalization object for future use
    model.norm_obj_ = norm_obj

    # convert labels to integers
    y_test = y_test.astype(np.int32)

    # test
    y_predprob = model.predict_proba(x_test)[:, 1]
    pred_auc = metrics.roc_auc_score(y_test, y_predprob)
    # print("AUC Score (Test): %f"%pred_auc)
    y_pred = model.predict(x_test)

    save_results(y_test, y_pred, y_predprob, test_result_path)

    return pred_auc

def save_results(labels, predictions, scores, save_path):
    df_results = pd.DataFrame()
    df_results['Y Label'] = labels
    df_results['Y Predicted'] = predictions
    df_results['Predicted score'] = scores
    df_results.to_csv(save_path, index=False)     


def main(train_feature_list, train_labels_list, test_feature_list, test_labels_list, save_dir):
    # train MLP model
    print("start MLP ablation")
    hidden_dims = [32,64]
    epoches = [32,64]
    hidden_layers = [1,2]
    l1 = [0.01, 0.001]
    l2 = [0.001, 0.0001, 0.00001]
    max_auc = 0.1
    best_hidden_dim = None
    best_epoch = 32
    best_l1=0
    best_l2=0
    # ablation hidden_dim
    for _, h_dim in enumerate(hidden_dims):
        for _, hidden_layer in enumerate(hidden_layers):
            hidden_dim = []
            for i in range(hidden_layer):
                hidden_dim.append(h_dim)
            # ablation epoch
            for _, epoch in enumerate(epoches):
                print("-----------train MLP model with {} hidden_dim and {} epoch-----------".format(hidden_dim, epoch))
                for l1_weight in l1:
                    for l2_weight in l2:
                        print("{} l1_weight, {} l2_weight".format(l1_weight, l2_weight))
                        auc_res=0
                        for i in range(5):
                            model_path = train(train_feature_list, train_labels_list, save_dir=save_dir, model_type="MLP", cross_val=False, hidden_dim=hidden_dim, n_epoch=epoch, l1_weight=l1_weight, l2_weight=l2_weight)
                            test_result_path = '{}/preds/test_{}.csv'.format(save_dir, "MLP")
                            auc_res += test(test_feature_list, test_labels_list, model_path, test_result_path)
                        auc_res = auc_res / 5
                        print("MEAN AUC Score (Test): %f"%auc_res)
                        if auc_res > max_auc:
                            max_auc = auc_res
                            best_epoch = epoch
                            best_hidden_dim = hidden_dim   
                            best_l1=l1_weight
                            best_l2=l2_weight             
    
    print("best_hidden_dim:", best_hidden_dim)
    print("best_epoch:", best_epoch)
    print("l1_weight:", best_l1)
    print("l2_weight:", best_l2)

    return best_hidden_dim, best_epoch, best_l1, best_l2
