'''
Author: sherrywaan sherrywaan@outlook.com
Date: 2024-09-24 12:12:57
LastEditors: sherrywaan sherrywaan@outlook.com
LastEditTime: 2024-11-25 14:44:15
FilePath: /wxy/PD/PD_early/code/full_modality_main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
import os
from easydict import EasyDict as edict
import yaml
import json

from core.train import train
from dataset.pd_dataset import PDDataset
from core.MLP_ablation import main as MLP_ablation
from core.RF_ablation import train_ablation as RF_ablation
from core.SVM_ablation import train_ablation as SVM_ablation
from core.LR_ablation import train_ablation as LR_ablation


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
    save_dir = "/data0/wxy/PD/PD_early/res_126-113-78/full_modality"
    file_path = "data/PD_dataset/126PD-113HC数据_缺省已填充"
    csv_path = "data/PD_dataset/subjects126-113.xlsx"
    parameter_path = "data/PD_dataset/parameters_78.json"
    model_types = ['RF', 'LR', 'SVM', 'MLP']
    modality_types = ["face_fix_par", "face_monologue", "blink", 
                    "glance", "voice_sus_vowel", "voice_alt_pro",
                    "voice_fix_par", "voice_monologue", "gait",  
                    "finger_tap", "toe_tap"]
    cross_val_nums=10

    os.makedirs("/data0/wxy/PD/PD_early/res_126-113-78", exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir,"preds"), exist_ok=True)
    os.makedirs(os.path.join(save_dir,"checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(save_dir,"configs"), exist_ok=True)
    
    # 10-fold cross validation
    for cross_idx in range(cross_val_nums):
        # prepare dataset
        pd_dataset = PDDataset(file_path, csv_path, parameter_path, split_step=cross_val_nums, start_idx=cross_idx)
        train_index, train_features, train_labels = pd_dataset.get_all(is_train=True)
        train_feature_list, train_labels_list = prepare_datas(train_index, train_features, train_labels, modality_types=modality_types)
        test_index, test_features, test_labels = pd_dataset.get_all(is_train=False)
        test_feature_list, test_labels_list = prepare_datas(test_index, test_features, test_labels, modality_types=modality_types)
        print("feature shape:", train_feature_list[0].shape)
        cross_set = (train_feature_list, train_labels_list, test_feature_list, test_labels_list)
        
        # hyper-parameter ablation
        param_config = edict()
        param_config.MLP,  param_config.RF,  param_config.SVM,  param_config.LR = {}, {}, {}, {}
        param_config.MLP.hidden_dims, param_config.MLP.num_epoches, param_config.MLP.lambda_l1, param_config.MLP.lambda_l2 = MLP_ablation(train_feature_list, train_labels_list, test_feature_list, test_labels_list, save_dir)
        param_config.RF.n_estimators, param_config.RF.max_depth, param_config.RF.min_samples_split, param_config.RF.random_state, param_config.RF.min_samples_leaf = RF_ablation(train_feature_list, train_labels_list, save_dir=save_dir)
        param_config.SVM.kernel, param_config.SVM.C = SVM_ablation(train_feature_list, train_labels_list, save_dir=save_dir)
        param_config.LR.penalty, param_config.LR.solver, param_config.LR.C = LR_ablation(train_feature_list, train_labels_list, save_dir=save_dir)
        with open(os.path.join(save_dir,"configs", "config_{}modal_{}cross.yml".format(modality_types, cross_idx)), 'w', encoding='utf-8') as f:
            yaml.dump(data=json.loads(json.dumps(param_config)), stream=f, allow_unicode=True)
        
        for model_t in model_types:
            # train and test
            if model_t == "MLP":
                for time_t in range(10):
                    model_path = train(cross_set, save_dir, param_config, model_type=model_t, modality=modality_types, cross_i=cross_idx, time_t=time_t)
            else:
                model_path = train(cross_set, save_dir, param_config, model_type=model_t, modality=modality_types, cross_i=cross_idx)
            