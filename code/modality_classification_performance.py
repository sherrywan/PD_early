'''
Author: sherrywaan sherrywaan@outlook.com
Date: 2024-06-12 10:45:47
LastEditors: sherrywaan sherrywaan@outlook.com
LastEditTime: 2024-11-13 15:10:03
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
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn import metrics
import joblib
import tqdm

import sys
sys.path.append("/data0/wxy/PD/PD_early")

from dataset.pd_dataset import PDDataset
from core.MLP_module import MLPClassifier as MLP
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from core.misc import get_metrics


def metric_res(file_path):
    y_true_all, y_pred_all, scores_all = [], [], []

    df = pd.read_csv(file_path)
    y_true_all.append(df['Y Label'].to_numpy())
    y_pred_all.append(df['Y Predicted'].to_numpy())
    scores_all.append(df['Predicted score'].to_numpy())

    met_all = get_metrics(y_true_all, y_pred_all, scores_all)

    return met_all


def metric_cross_res(file_paths):
    y_true_all, y_pred_all, scores_all = [], [], []
    
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        y_true_all.append(df['Y Label'].to_numpy())
        y_pred_all.append(df['Y Predicted'].to_numpy())
        scores_all.append(df['Predicted score'].to_numpy())

    met_all = get_metrics(y_true_all, y_pred_all, scores_all, std=True)

    return met_all

def train(
    x_train_ori, y_train_ori, save_dir, model_type="RF", cross_train=False, cross_val=False, cross_nums = 10, cross_set=None, modality=0, time_t=0
):
    assert model_type in ['RF', 'LR', 'SVM', 'MLP'], "model type is not supported"

    if model_type == "RF":
        model = RandomForestClassifier(criterion="entropy", 
                                       n_estimators=730, 
                                       max_depth=11, 
                                       min_samples_split=2, 
                                       bootstrap=False, 
                                       random_state=97, 
                                       min_samples_leaf=4)
    elif model_type == "LR":
        model = LogisticRegression(
            C = 0.4,
            penalty = 'l2',
            solver = 'sag',
            random_state = 0,
            max_iter=1000
        )
    elif model_type == "MLP":
        model = MLP(
            hidden_dims = (32,),
            num_epochs = 32,
            batch_size = 16,
            device = 'cpu',
            lambda_l1=0.01,
            lambda_l2=0.00001
        )
    elif model_type == 'SVM':
        model = SVC(
            kernel = 'linear',
            C = 0.2,
            probability=True,
            random_state = 0
        )
    
    # cross_nums-fold in training set
    if cross_train:
        # split dataset
        kf = KFold(n_splits=cross_nums, random_state=cross_nums, shuffle=True)
        for i, (train_index, test_index) in enumerate(kf.split(x_train_ori)):
            print("-----------train {} model {} iter with {}-cross train: {}th cross-----------".format(model_type, time_t, cross_nums, i))
            x_train = x_train_ori[train_index]
            y_train = y_train_ori[train_index]
            x_test = x_train_ori[test_index]
            y_test = y_train_ori[test_index]
            train_main(model, x_train, y_train)
            # save model checkpoint
            checkpoint_path = '{}/checkpoints/train_{}_{}modal_{}iter_{}crosstrain.ckpt'.format(save_dir,model_type,modality,time_t,i)
            joblib.dump(model, checkpoint_path)
            test_result_path = '{}/preds/train_{}_{}modal_{}iter_{}crosstrain.csv'.format(save_dir,model_type,modality,time_t,i)
            test(x_test, y_test, checkpoint_path, test_result_path)
    
    # cross x-fold in whole set
    elif cross_val:
        assert cross_set is not None, "check if there is cross_set input"

        for i, (x_train,y_train,x_test,y_test) in enumerate(cross_set):
            print("-----------train {} model {} iter with cross val: {}th cross-----------".format(model_type, time_t, i))
            train_main(model, x_train, y_train)
            # save model checkpoint
            checkpoint_path = '{}/checkpoints/train_{}_{}modal_{}iter_{}crossval.ckpt'.format(save_dir,model_type,modality,time_t,i)
            joblib.dump(model, checkpoint_path)
            test_result_path = '{}/preds/train_{}_{}modal_{}iter_{}crossval.csv'.format(save_dir,model_type,modality,time_t,i)
            test(x_test, y_test, checkpoint_path, test_result_path)


    else:
        train_main(model, x_train_ori, y_train_ori)
        # save model checkpoint       
        checkpoint_path = '{}/checkpoints/train_{}_{}modal_{}iter.ckpt'.format(save_dir,model_type,modality, time_t)
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

    save_results(y_test, y_pred, y_predprob, test_result_path)
    

def save_results(labels, predictions, scores, save_path):
    df_results = pd.DataFrame()
    df_results['Y Label'] = labels
    df_results['Y Predicted'] = predictions
    df_results['Predicted score'] = scores
    df_results.to_csv(save_path, index=False)     


def prepare_datas(data_index, data_features, data_labels, modality_types=["face_fix_par", "face_monologue", "blink", "glance", "voice_sus_vowel", "voice_alt_pro",
                                                                        "voice_fix_par", "voice_monologue", "gait",  "finger_tap", "toe_tap"]):
    features = []
    labels = []
    for idx_i, idx in enumerate(data_index):
        f_s = data_features[idx]
        fe_s = []
        for modality_type in f_s.keys():
            if modality_type in modality_types:
                fe_s += f_s[modality_type].values()
        features.append(fe_s)
        labels.append(data_labels[idx])
    features = np.array(features)
    labels = np.array(labels)
    return features, labels


if __name__ == "__main__":
    save_dir = "/data0/wxy/PD/PD_early/res_126-113-78/sole_modality"
    file_path = "data/PD_dataset/126PD-113HC数据_缺省已填充"
    csv_path = "data/PD_dataset/subjects126-113.xlsx"
    parameter_path = "data/PD_dataset/parameters_78.json"
    model_types = ['LR', 'SVM', 'RF', 'MLP']
    modality_types = ["face_fix_par", "face_monologue", "blink", 
                    "glance", "voice_sus_vowel", "voice_alt_pro",
                    "voice_fix_par", "voice_monologue", "gait",  
                    "finger_tap", "toe_tap"]
    modality_types_vis = ["gait", "finger_tap", "toe_tap",
                          "blink","face_fix_par", "face_monologue",  
                    "glance", "voice_sus_vowel", "voice_alt_pro",
                    "voice_fix_par", "voice_monologue"]
    modality_shows = ["Limb_Walking", "Limb_Finger_Tapping", "Limb_Toe_Tapping",
                      "Facial_Blinking", "Facial_Reading", "Facial_Monologue",  "Eyeball_Saccade", 
                      "Laryngeal_Sustained_Vowel", "Laryngeal_Sequential", "Laryngeal_Reading", "Laryngeal_Monologue"]
    cross_val_nums=10

    palette = {  
    'LR': (249/255, 208/255, 227/255),  # RGB(249,208,227)  
    'SVM': (182/255, 215/255, 231/255),  # RGB(182,215,231)  
    'RF': (176/255, 176/255, 218/255),  # RGB(176,176,218)  
    'MLP': (253/255, 229/255, 176/255)   # RGB(253,229,176)  
    }  
    fig, axs = plt.subplots(4,1, figsize=(12,20))
    plt.style.use('default')
    sns.set_theme(style="whitegrid")
    for model_idx, model_t in enumerate(model_types):
        # build metric results dataframe
        met_res_list = []
        for modal_idx, modality in enumerate(modality_types):
            modality = [modality]
            best_crossval_time_t=np.load(os.path.join(save_dir, "MLP_{}modal_best_idx.npy".format(modality)))
            file_paths = []
            for cross_i in range(cross_val_nums):
                if model_t == "MLP":
                    file_name = "train_{}_{}modal_{}iter_{}crossval.csv".format(model_t,modality,int(best_crossval_time_t[cross_i]),cross_i)
                else:
                    file_name = "train_{}_{}modal_0iter_{}crossval.csv".format(model_t,modality,cross_i)
                file_path = os.path.join(save_dir, "preds", file_name)
                file_paths.append(file_path)
                met_res_dict = metric_res(file_path)
                met_res = {}
                met_res['modality'] = modality[0]
                met_res['cross_i'] = cross_i
                met_res['AUC (ROC)'] = met_res_dict['AUC (ROC)'][0]
                met_res_list.append(met_res)
            met_cross_res = metric_cross_res(file_paths)
            print("------------{} {} {}-cross val results in training dataset------------".format("SVM", modality, cross_val_nums))
            print(met_cross_res)        
        test_datas = pd.DataFrame(met_res_list)

        # calculate CV of each modality
        cv = test_datas.groupby('modality')['AUC (ROC)'].apply(lambda x: np.std(x) / np.mean(x))
        cv.to_csv(os.path.join(save_dir,"{}_modality_cv.csv".format(model_t)))

        # vis test results comparison between different models
        axs[model_idx].set_title("{}".format(model_t), fontsize=28)    
        for i, moda in enumerate(modality_types_vis):  
            if cv[moda] < 0.15:
                sns.violinplot(data=test_datas[test_datas['modality'] == moda], x="modality", y="AUC (ROC)", color=palette[model_t], ax=axs[model_idx])
            else:
                sns.violinplot(data=test_datas[test_datas['modality'] == moda], x="modality", y="AUC (ROC)", color=palette[model_t], alpha=0.5, ax=axs[model_idx])
        if model_idx<3:
            axs[model_idx].set_xticks([])
        else:
            axs[model_idx].set_xticks(range(11), modality_shows, ha='right', rotation=60, fontsize=28)
        axs[model_idx].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8], [0.0, 0.2, 0.4, 0.6, 0.8], fontsize=20)
        axs[model_idx].axhline(0.6, linestyle="--", color="black", alpha=0.8)
        axs[model_idx].set_ylabel("AUC", fontsize=24)
        axs[model_idx].set_xlabel(None)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,"modality_stability_results.svg"),dpi=300,format="svg")