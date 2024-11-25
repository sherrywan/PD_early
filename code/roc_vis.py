'''
Author: sherrywaan sherrywaan@outlook.com
Date: 2024-06-12 21:49:28
LastEditors: sherrywaan sherrywaan@outlook.com
LastEditTime: 2024-09-11 09:49:02
FilePath: /wxy/PD/PD_early/code/results_analysis.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


def roc_vis(file_path , corss_i):
    df = pd.read_csv(file_path)
    y_true_all = df['Y Label'].to_numpy()
    y_pred_all = df['Y Predicted'].to_numpy()
    scores_all = df['Predicted score'].to_numpy()

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr[0], tpr[0], _ = roc_curve(y_true_all, scores_all, drop_intermediate=False)
    linewidth=2
    plt.plot(fpr[0], tpr[0], lw=linewidth, label="fold-{}".format(cross_i))
    plt.plot([0, 1], [0, 1], color='navy', lw=linewidth, linestyle='--')
    
    



if __name__=="__main__":
    save_dir = "/data0/wxy/PD/PD_early/res_127-116/preds"
    # model_type = ['RF', 'LR', 'SVM', 'MLP']
    model_type = ['RF', 'LR', 'SVM', 'MLP']
    metrics = ['Accuracy', 'Precision', 'Recall', 'AUC (ROC)']
    cross_nums=4
    modality=11

    
    # analysis cross val results
    # build metric results dataframe
    met_res_list = []
    for model_t in model_type:
        plt.figure(figsize=(6, 6))
        plt.title("ROC(AUC) of {}".format(model_t), fontsize=18)
        for cross_i in range(cross_nums):
            file_name = "train_{}_{}modal_0iter_{}crossval.csv".format(model_t,modality,cross_i)
            file_path = os.path.join(save_dir, file_name)
            roc_vis(file_path, cross_i)
        fontsize = 14
        plt.xlabel('False Positive Rate', fontsize = fontsize)
        plt.ylabel('True Positive Rate', fontsize = fontsize)
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(save_dir,"crossval_auc_{}.png".format(model_t)))
        plt.cla()
        plt.close()
