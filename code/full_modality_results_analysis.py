'''
Author: sherrywaan sherrywaan@outlook.com
Date: 2024-06-12 21:49:28
LastEditors: sherrywaan sherrywaan@outlook.com
LastEditTime: 2024-11-13 15:30:04
FilePath: /wxy/PD/PD_early/code/results_analysis.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve
import logging

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


def roc_vis(ax, file_path , cross_i, color, auc):
    df = pd.read_csv(file_path)
    y_true_all = df['Y Label'].to_numpy()
    y_pred_all = df['Y Predicted'].to_numpy()
    scores_all = df['Predicted score'].to_numpy()

    fpr = dict()
    tpr = dict()
    fpr[0], tpr[0], _ = roc_curve(y_true_all, scores_all, drop_intermediate=False)
    linewidth=2
    ax.plot(fpr[0], tpr[0], lw=linewidth, label="fold-{} ({:.02f})".format(cross_i,auc), alpha=.8)
    ax.plot([0, 1], [0, 1], color=color, lw=linewidth, linestyle='--')


if __name__=="__main__":
    save_dir = "/data0/wxy/PD/PD_early/res_126-113-78/full_modality"
    file_path = "data/PD_dataset/126PD-113HC数据_缺省已填充"
    csv_path = "data/PD_dataset/subjects126-113.xlsx"
    parameter_path = "data/PD_dataset/parameters_78.json"
    model_types = ['LR', 'SVM', 'RF', 'MLP']
    modality_types = ["face_fix_par", "face_monologue", "blink", 
                    "glance", "voice_sus_vowel", "voice_alt_pro",
                    "voice_fix_par", "voice_monologue", "gait",  
                    "finger_tap", "toe_tap"]
    metrics = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'AUC (ROC)']
    vis_index = [4, 5, 3, 6, 7, 8, 9, 10, 0, 1, 2]
    cross_val_nums=10
    best_crossval_time_t=np.zeros((cross_val_nums))

    logging.basicConfig(  
        filename=os.path.join(save_dir,'data.log'),  
        level=logging.INFO,  
        format='%(asctime)s - %(levelname)s - %(message)s' 
    )  

    # analysis test results
    # find best MLP    
    for cross_i in range(cross_val_nums):
        max_auc = 0.5
        for time_t in range(10):
            file_name = "train_{}_{}modal_{}iter_{}crossval.csv".format("MLP",modality_types,time_t,cross_i)
            file_path = os.path.join(save_dir, "preds", file_name)
            if not os.path.exists(file_path):
                continue
            met_res_dict = metric_res(file_path)
            print(met_res_dict)
            if met_res_dict['AUC (ROC)'][0] > max_auc:
                max_auc = met_res_dict['AUC (ROC)'][0]
                best_crossval_time_t[cross_i] = time_t
    print("best_crossval_time_t:", best_crossval_time_t)
    np.save(os.path.join(save_dir, "MLP_best_idx.npy"), best_crossval_time_t)

    # analysis cross validation results
    # build metric results dataframe
    met_cross_res = {}
    met_res_list = []
    auc_res_dict={}
    for model_t in model_types:
        file_paths = []
        auc_res_dict[model_t]={}
        for cross_i in range(cross_val_nums):
            if model_t == "MLP":
                file_name = "train_{}_{}modal_{}iter_{}crossval.csv".format(model_t, modality_types, int(best_crossval_time_t[cross_i]), cross_i)
            else:
                file_name = "train_{}_{}modal_0iter_{}crossval.csv".format(model_t, modality_types, cross_i)
            file_path = os.path.join(save_dir, "preds", file_name)
            file_paths.append(file_path)
            met_res_dict = metric_res(file_path)
            for met_name in met_res_dict.keys():
                met_name_key= met_name
                if met_name == "AUC (ROC)":
                    met_name_key = "AUC"
                if met_name not in metrics:
                    continue
                met_res = {}
                met_res['model_type'] = model_t
                met_res['cross_i'] = cross_i
                met_res['metric'] = met_name_key
                met_res['value'] = met_res_dict[met_name][0]
                met_res_list.append(met_res)
            auc_res_dict[model_t][cross_i] = met_res_dict['AUC (ROC)'][0]
        met_cross_res[model_t] = metric_cross_res(file_paths)
        print("------------{} {}-cross val results in training dataset------------".format(model_t, cross_val_nums))
        print(met_cross_res) 
    for key, value in met_cross_res.items():  
        logging.info(f'{key}: {value}')  
    test_datas = pd.DataFrame(met_res_list)

    # vis test results comparison between different models
    palette = {  
        'LR': (249/255, 208/255, 227/255),  # RGB(249,208,227)  
        'SVM': (182/255, 215/255, 231/255),  # RGB(182,215,231)  
        'RF': (176/255, 176/255, 218/255),  # RGB(176,176,218)  
        'MLP': (253/255, 229/255, 176/255)   # RGB(253,229,176)  
    }  
    plt.figure(figsize=(24, 6))
    sns.set_style('whitegrid')
    g = sns.barplot(data=test_datas, x="metric", y="value", hue="model_type", errorbar="sd", gap=0.1, palette=palette, edgecolor='black')
    for p in g.patches:  
        height = p.get_height()
        if height <= 0:
            continue
        g.annotate(format(p.get_height(), '.2f'),   
                      (p.get_x() + p.get_width() / 2., 0),   
                      ha='center', va='baseline',   
                      fontsize=16, color='black',
                      xytext=(0, 150),  # 位置偏移  
                      textcoords='offset points')  
    plt.legend(title=None, loc='lower right', fontsize=20)  
    plt.xlabel(None)
    plt.ylabel(None)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,"crossval_results.svg"), dpi=300, format="svg")
    plt.cla()
    plt.close()

    # vis auc(roc)
    fig, axs = plt.subplots(1, 4, figsize=(24, 6))
    for idx, model_t in enumerate(model_types):
        color = palette[model_t]
        axs[idx].set_title("AUC of {}".format(model_t), fontsize=24)
        for cross_i in range(cross_val_nums):
            if model_t == "MLP":
                file_name = "train_{}_{}modal_{}iter_{}crossval.csv".format(model_t, modality_types, int(best_crossval_time_t[cross_i]), cross_i)
            else:
                file_name = "train_{}_{}modal_0iter_{}crossval.csv".format(model_t, modality_types, cross_i)
            file_path = os.path.join(save_dir, "preds", file_name)
            roc_vis(axs[idx], file_path, cross_i, color, auc_res_dict[model_t][cross_i])
        fontsize = 20
        axs[idx].set_xlabel('1 - Specificity', fontsize = fontsize)
        axs[idx].set_ylabel('Recall', fontsize = fontsize)
        axs[idx].legend(loc="lower right", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,"crossval_auc.svg"), dpi=300, format="svg")
