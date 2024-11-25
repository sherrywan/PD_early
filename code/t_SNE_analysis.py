'''
Author: sherrywaan sherrywaan@outlook.com
Date: 2024-09-12 15:16:14
LastEditors: sherrywaan sherrywaan@outlook.com
LastEditTime: 2024-09-24 11:59:30
FilePath: /wxy/PD/PD_early/code/t_SNE_analysis.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
import sys
sys.path.append("/data0/wxy/PD/PD_early")
from PD.PD_early.code.dataset.pd_dataset import PDDataset



# 对样本进行预处理并画图
def plot_embedding(data, label, title):
    """
    :param data:数据集
    :param label:样本标签
    :param title:图像标题
    :return:图像
    """
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)     # 对数据进行归一化处理
    fig = plt.figure()      # 创建图形实例
    ax = plt.subplot(111)       # 创建子图
    # 遍历所有样本
    for i in range(data.shape[0]):
        # 在图中为每个数据点画出标签
        plt.text(data[i, 0], data[i, 1], str(int(label[i])), color=plt.cm.Set1(label[i] / 1),
                 fontdict={ "weight" :  "bold" ,  "size" : 10})
    plt.xticks()        # 指定坐标的刻度
    plt.yticks()
    plt.title(title, fontsize=14)
    
    # 返回值
    return fig


# 主函数，执行t-SNE降维
def main(data, label, n_samples, n_features, title):
    print("Starting compute t-SNE Embedding...")
    ts = TSNE(n_components=2, init="pca", random_state=0, method="exact")
    # t-SNE降维
    result = ts.fit_transform(data)
    # 调用函数，绘制图像
    fig = plot_embedding(result, label, title)
    # 显示图像
    plt.show()
    plt.savefig("/data0/wxy/PD/PD_early/tsne_res/{}.png".format(title))


def prepare_datas(data_index, data_features, data_labels, module_types=["face_fix_par", "face_monologue", "blink", "glance", "voice_sus_vowel", "voice_alt_pro",
                                                                        "voice_fix_par", "voice_monologue", "gait", "finger_tap", "toe_tap"]):
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


# 主函数
if __name__ == "__main__":
    # load data
    file_path = "/data0/wxy/PD/PD_early/data/PD_dataset/127PD-116HC数据分析"
    csv_path_1 = "/data0/wxy/PD/PD_early/data/PD_dataset/subjects_47-38.xlsx"
    csv_path_2 = "/data0/wxy/PD/PD_early/data/PD_dataset/subjects_80-78.csv"
    csv_path_3 = "/data0/wxy/PD/PD_early/data/PD_dataset/127PD-116HC数据分析/早筛前六期127PD-116HC-训练和测试划分--最终版.xlsx"
    paras_path = "/data0/wxy/PD/PD_early/data/PD_dataset/parameters_94.json"
    paras_path_1 = "/data0/wxy/PD/PD_early/data/PD_dataset/parameters_74.json"
    module_types = ['voice_sus_vowel']

    # different datas
    # prepare dataset
    pd_dataset_1 = PDDataset(file_path, csv_path_1, paras_path)
    train_index, train_features, train_labels = pd_dataset_1.get_all(is_train=True)
    train_feature_list, train_labels_list = prepare_datas(train_index, train_features, train_labels, module_types=module_types)
    test_index, test_features, test_labels = pd_dataset_1.get_all(is_train=False)
    test_feature_list, test_labels_list = prepare_datas(test_index, test_features, test_labels, module_types=module_types)
    feature_list_1 = np.concatenate((train_feature_list, test_feature_list), axis=0)
    label_list_1 = np.zeros((feature_list_1.shape[0]))

    pd_dataset_2 = PDDataset(file_path, csv_path_2, paras_path)
    train_index, train_features, train_labels = pd_dataset_2.get_all(is_train=True)
    train_feature_list, train_labels_list = prepare_datas(train_index, train_features, train_labels, module_types=module_types)
    test_index, test_features, test_labels = pd_dataset_2.get_all(is_train=False)
    test_feature_list, test_labels_list = prepare_datas(test_index, test_features, test_labels, module_types=module_types)
    feature_list_2 = np.concatenate((train_feature_list, test_feature_list), axis=0)
    label_list_2 = np.ones((feature_list_2.shape[0]))

    features = np.concatenate((feature_list_1,feature_list_2), axis=0)
    labels = np.concatenate((label_list_1, label_list_2), axis=0)

    # tsne
    main(features, labels, features.shape[0], features.shape[1], "t-SNE analysis in voice of different datas (0:47-38, 1:80-78)")

    # different features
    # prepare dataset
    pd_dataset_1 = PDDataset(file_path, csv_path_3, paras_path)
    train_index, train_features, train_labels = pd_dataset_1.get_all(is_train=True)
    train_feature_list, train_labels_list = prepare_datas(train_index, train_features, train_labels)
    test_index, test_features, test_labels = pd_dataset_1.get_all(is_train=False)
    test_feature_list, test_labels_list = prepare_datas(test_index, test_features, test_labels)
    feature_list_1 = np.concatenate((train_feature_list, test_feature_list), axis=0)
    label_list_1 = np.concatenate((train_labels_list, test_labels_list), axis=0)
    # tsne
    main(feature_list_1, label_list_1, feature_list_1.shape[0], feature_list_1.shape[1], "t-SNE analysis of 94 features")

    # prepare dataset
    pd_dataset_1 = PDDataset(file_path, csv_path_3, paras_path_1)
    train_index, train_features, train_labels = pd_dataset_1.get_all(is_train=True)
    train_feature_list, train_labels_list = prepare_datas(train_index, train_features, train_labels)
    test_index, test_features, test_labels = pd_dataset_1.get_all(is_train=False)
    test_feature_list, test_labels_list = prepare_datas(test_index, test_features, test_labels)
    feature_list_1 = np.concatenate((train_feature_list, test_feature_list), axis=0)
    label_list_1 = np.concatenate((train_labels_list, test_labels_list), axis=0)
    # tsne
    main(feature_list_1, label_list_1, feature_list_1.shape[0], feature_list_1.shape[1], "t-SNE analysis of 74 features")
