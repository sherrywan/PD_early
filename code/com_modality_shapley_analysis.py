'''
Author: sherrywaan sherrywaan@outlook.com
Date: 2024-08-23 15:52:04
LastEditors: sherrywaan sherrywaan@outlook.com
LastEditTime: 2024-11-12 16:35:45
FilePath: /wxy/PD/PD_early/code/shapley_analysis.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import joblib
import json
import shap
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
import pandas as pd
import seaborn as sns

from dataset.pd_dataset import PDDataset

def prepare_datas(data_index, data_features, data_labels, modality_types=["face_fix_par", "face_monologue", "blink", 
                                                                        "glance", "voice_sus_vowel", "voice_alt_pro",
                                                                        "voice_fix_par", "voice_monologue", "gait",  
                                                                        "finger_tap", "toe_tap"]):
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
    save_dir = "/data0/wxy/PD/PD_early/res_126-113-78/com_modality"
    file_path = "data/PD_dataset/126PD-113HC数据_缺省已填充"
    csv_path = "data/PD_dataset/subjects126-113.xlsx"
    parameter_path = "data/PD_dataset/parameters_78.json"
    model_types = ['LR', 'SVM', 'RF', 'MLP']
    modality_types = ["face_fix_par", "glance", "gait"]
    modality_shows = ["Facial_Reading", "Eyeball_Saccade", "Limb_Walking"]
    vis_index=[1,2,0]
    cross_val_nums=10
    best_crossval_time_t=np.load(os.path.join(save_dir, "MLP_['face_fix_par', 'glance', 'gait']modal_best_idx.npy"))

    # load parameters name
    with open(parameter_path, 'r') as json_file:
        loaded_parameters = json.load(json_file)
    feat_labels = []
    moda_type = []
    for key in list(loaded_parameters.keys()):
        if key not in modality_types:
            loaded_parameters.pop(key)
    for key in loaded_parameters:
        moda_type.append(key)
        for sub_key in loaded_parameters[key]:
            feat_labels.append(key+'_'+sub_key)
    feat_labels = np.array(feat_labels)
    print("parameters:", loaded_parameters)

    if not os.path.exists(os.path.join(save_dir, "modalily_shapley_ratio.yaml")):
        modality_shap_ratio = {}
        for model_t in model_types:
            modality_shap_ratio[model_t] = {}
        for cross_idx in range(cross_val_nums):
            # prepare dataset
            pd_dataset = PDDataset(file_path, csv_path, parameter_path, split_step=cross_val_nums, start_idx=cross_idx)
            train_index, train_features, train_labels = pd_dataset.get_all(is_train=True)
            train_feature_list, train_labels_list = prepare_datas(train_index, train_features, train_labels, modality_types=modality_types)
            test_index, test_features, test_labels = pd_dataset.get_all(is_train=False)
            test_feature_list, test_labels_list = prepare_datas(test_index, test_features, test_labels, modality_types=modality_types)
            cross_set = (train_feature_list, train_labels_list, test_feature_list, test_labels_list)
            
            for model_t in model_types:
                # load model
                if model_t == "MLP":
                    model_path = os.path.join(save_dir,"checkpoints","train_{}_{}modal_{}iter_{}crossval.ckpt".format(model_t, modality_types, int(best_crossval_time_t[cross_idx]), cross_idx))
                else:
                    model_path = os.path.join(save_dir,"checkpoints","train_{}_{}modal_0iter_{}crossval.ckpt".format(model_t, modality_types, cross_idx))
                model = joblib.load(model_path)
                # normalize features
                norm_obj =  model.norm_obj_
                x_test = norm_obj.transform(test_feature_list)
                x_train = norm_obj.transform(train_feature_list)
                # convert labels to integers
                y_test = test_labels_list.astype(np.int32)
                y_train = train_labels_list.astype(np.int32)
                # shap explainer
                # feature analysis
                explainer = shap.Explainer(model.predict, x_train, feature_names=feat_labels)
                shap_values = explainer(x_test)
                # modality ratio
                mean_abs_shap = np.mean(abs(shap_values.values), axis=0)
                sum_shap = np.sum(mean_abs_shap)
                mean_abs_shap = mean_abs_shap/sum_shap
                # build dataframe 
                idx_i = 0
                modality_shap_ratio[model_t]['modality'] = []
                modality_shap_ratio[model_t][f'fold-{cross_idx}'] = []
                for key in loaded_parameters:
                    modality_shap_ratio[model_t]['modality'].append(key)
                    shap_r = 0
                    for sub_key in loaded_parameters[key]:
                        shap_r += mean_abs_shap[idx_i]
                        idx_i += 1
                    modality_shap_ratio[model_t][f'fold-{cross_idx}'].append(shap_r)
        with open(os.path.join(save_dir, "modalily_shapley_ratio.yaml"), 'w', encoding='utf-8') as f:
            yaml.dump(data=json.loads(json.dumps(modality_shap_ratio)), stream=f, allow_unicode=True)
    
    else:
        with open(os.path.join(save_dir, "modalily_shapley_ratio.yaml"), 'r', encoding='utf-8') as f:
            modality_shap_ratio = yaml.safe_load(f)
        # vis modality shap for each model
        palette = {  
            'LR': (249/255, 208/255, 227/255),  # RGB(249,208,227)  
            'SVM': (182/255, 215/255, 231/255),  # RGB(182,215,231)  
            'RF': (176/255, 176/255, 218/255),  # RGB(176,176,218)  
            'MLP': (253/255, 229/255, 176/255)   # RGB(253,229,176)  
        }  
        fig, axs = plt.subplots(1,4, figsize=(24, 18))
        bar_width = 0.5
        y_pos = range(len(modality_shows))
        for idx, model_t in enumerate(model_types):
            test_datas = pd.DataFrame(modality_shap_ratio[model_t])
            test_datas.index = vis_index
            test_datas.to_csv(os.path.join(save_dir, f"{model_t}_modality_shap.csv"), index=False)
            cross_colors = [(*palette[model_t],0.1+cross_i*0.1) for cross_i in range(cross_val_nums)] 
            for cross_idx in range(cross_val_nums):
                if cross_idx > 0:
                    left_loc = 0
                    for i in range(cross_idx):
                        left_loc += test_datas[f'fold-{i}']
                    axs[idx].barh(y_pos, test_datas[f'fold-{cross_idx}'], height=bar_width, left=left_loc, label=f'fold-{cross_idx}', color=cross_colors[cross_idx])
                else:
                    axs[idx].barh(y_pos, test_datas[f'fold-{cross_idx}'], height=bar_width, label=f'fold-{cross_idx}',color=cross_colors[cross_idx])
            if idx == 0:
                axs[idx].set_yticks(y_pos, np.array(modality_shows))
            else:
                axs[idx].set_yticks([])
            axs[idx].set_title(model_t)
            axs[idx].legend()
        plt.savefig(os.path.join(save_dir, "modalily_shapley_each.svg"), dpi=300, format="svg")

        # vis modality shap for all model (sum of the mean shap in each model)
        modality_shap_ratio_all = {}
        for idx, model_t in enumerate(model_types):   
            test_datas = pd.DataFrame(modality_shap_ratio[model_t])
            modality_shap_ratio_all['modality'] = test_datas['modality']
            for cross_idx in range(cross_val_nums):
                if cross_idx == 0:
                    modality_shap_ratio_all[model_t] = test_datas[f'fold-{cross_idx}']
                else:
                    modality_shap_ratio_all[model_t] += test_datas[f'fold-{cross_idx}']
            modality_shap_ratio_all[model_t] = modality_shap_ratio_all[model_t]/cross_val_nums
        test_datas = pd.DataFrame(modality_shap_ratio_all)
        test_datas.index = vis_index
        fig= plt.subplots(figsize=(8,8))
        bar_width = 0.5
        y_pos = range(len(modality_shows)) 
        left_loc = 0    
        for idx, model_t in enumerate(model_types):   
            plt.barh(vis_index, test_datas[model_t], height=bar_width, left=left_loc, label=model_t, color=palette[model_t])
            left_loc += test_datas[model_types[idx]]
            if idx == 3:
                for a,b in zip(vis_index, left_loc):
                    plt.text(b,a,"{:.2f}".format(b),fontsize=20)
        # plt.axvline(0.36,linestyle="--", color="gray", alpha=0.8)
        x_ticks = plt.xticks()[0]
        # new_x_ticks = np.append(x_ticks, 0.36)
        plt.xticks(np.sort(x_ticks), fontsize=16)
        plt.yticks(vis_index, np.array(modality_shows), fontsize=24)
        # plt.title("SUM of |SHAP|-ratio", fontsize=20)
        plt.legend(fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "modalily_shapley_all.svg"), dpi=300, format="svg")
            