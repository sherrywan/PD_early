'''
Author: sherrywaan sherrywaan@outlook.com
Date: 2024-06-12 21:31:35
LastEditors: sherrywaan sherrywaan@outlook.com
LastEditTime: 2024-08-11 15:45:00
FilePath: /wxy/PD/PD_early/code/RF_feature_vis.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.special import softmax
import json
import joblib

model_path = "/data0/wxy/PD/PD_early/res/checkpoints/train_MLP_8.ckpt"
paras_path = "/data0/wxy/PD/PD_early/data/PD_dataset/parameters.json"
# 加载模型
mlp_state = joblib.load(model_path)

# 获取特征重要性
mlp_linear1 = mlp_state.net_.module[1].weight
mlp_linear1 = abs(mlp_linear1).detach().numpy()
mlp_linear_paras = np.sum(mlp_linear1, axis=0)
importances = softmax(mlp_linear_paras)
indices = np.argsort(importances)[::-1]

# 获取特征名称 and 不同模态的权重之和
with open(paras_path, 'r') as json_file:
    loaded_parameters = json.load(json_file)
feat_labels = []
module_importances = []
module_type = []
imp_idx = 0
for key in loaded_parameters:
    module_type.append(key)
    module_imp = 0
    for sub_key in loaded_parameters[key]:
        module_imp += importances[imp_idx]
        imp_idx += 1
        feat_labels.append(key+sub_key)
    module_importances.append(module_imp)

# 打印每个特征重要性
for f in range(len(importances)):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))


# 可视化特征重要性
plt.figure(figsize=(30, 7))
plt.title("各个特征的重要程度", fontsize=18)
plt.ylabel("Importance level", fontsize=15, rotation=90)
plt.rcParams['font.sans-serif'] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False

for i in range(len(importances)):
    plt.bar(i, importances[indices[i]], color='orange', align='center')

plt.xticks(np.arange(len(importances)), np.array(feat_labels)[indices], rotation=45, fontsize=15)
plt.show()
plt.savefig("/data0/wxy/PD/PD_early/res/checkpoints/train_MLP_feature_importances.png")
plt.cla()
plt.close()


# 打印每个模态特征重要性
for f in range(len(module_importances)):
    print("%2d) %-*s %f" % (f + 1, 30, module_type[f], module_importances[f]))

# 可视化模态特征重要性之和
plt.figure(figsize=(15, 7))
plt.title("各个模态特征的重要程度", fontsize=18)
plt.ylabel("Importance level", fontsize=15, rotation=90)
plt.rcParams['font.sans-serif'] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False

for i, m in enumerate(module_type):
    plt.bar(m, module_importances[i], color='orange', align='center')

plt.xticks(np.arange(len(module_importances)), np.array(module_type), rotation=10, fontsize=15)
plt.show()
plt.savefig("/data0/wxy/PD/PD_early/res/checkpoints/train_MLP_module_feature_importances.png")