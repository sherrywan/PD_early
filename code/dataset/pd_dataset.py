import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import json
import os
from sklearn.impute import KNNImputer


class PDDataset(Dataset):

    def __init__(self, root_folder, subject_csv_path,parameter_path,start_idx=0,split_step=5,save_folder=None, read_whole_feature=False, fill_whole_feature=False):
        self.root_folder = root_folder
        self.save_folder = save_folder
        self.subject_csv_path = subject_csv_path
        self.parameter_path = parameter_path
        self.train_index,self.train_label,self.test_index,self.test_label,self.all_index,self.all_label = self.split_train_test(split_step, start_idx=start_idx)
        if read_whole_feature:
            self.check_whole_features(self.train_index)
            self.check_whole_features(self.test_index)
        elif fill_whole_feature:
            self.check_whole_features(self.all_index, self.all_label, fill_whole_feature=True)
        else:
            self.train_features = self.read_features(self.train_index)
            self.test_features = self.read_features(self.test_index)
    
    def mean_supp(self,df):
        for col in range(3,df.shape[1]):
            reserve_0 = []
            reserve_1 = []
            for row in range(df.shape[0]):
                if not pd.isna(df.iloc[row,col]):
                    try:
                        name_index = self.all_index.index(df.iloc[row,0])
                        name_label = self.all_label[name_index]
                    except:
                        continue
                    if name_label == 0:
                        reserve_0.append(float(df.iloc[row,col]))
                    else:
                        reserve_1.append(float(df.iloc[row,col]))
            #print(len(reserve_0),len(reserve_1),"############")
            reserve_0 = reserve_0[0:min(len(reserve_0),len(reserve_1))]
            reserve_1 = reserve_1[0:min(len(reserve_0),len(reserve_1))]
            #print(reserve_0)
            #print(reserve_1)
            if len(reserve_0)+len(reserve_1) <1:
                continue
            mean_value = (np.mean(reserve_0)+np.mean(reserve_1))/2
            for row in range(df.shape[0]):
                if pd.isna(df.iloc[row,col]):
                    df.iloc[row,col] = mean_value
        return df
    
    def split_train_test(self, split_step, start_idx=0):
        if ".xlsx" in self.subject_csv_path:
            df = pd.read_excel(self.subject_csv_path)
        else:
            df = pd.read_csv(self.subject_csv_path)
        # 取出第一列和第三列
        col1 = df.iloc[:, 0]
        col3 = df.iloc[:, 2]

        # 创建测试集和训练集
        test_indices = list(range(start_idx, len(col1), split_step))
        train_indices = [i for i in range(len(col1)) if i not in test_indices]

        # 划分测试集和训练集
        test_index = col1.iloc[test_indices].tolist()
        test_label = col3.iloc[test_indices].tolist()

        train_index = col1.iloc[train_indices].tolist()
        train_label = col3.iloc[train_indices].tolist()

        all_index = col1.tolist()
        all_label  = col3.tolist()

        train_label_dic = {}
        for i in range(len(train_index)):
            train_label_dic[train_index[i]] = train_label[i]

        test_label_dic = {}
        for i in range(len(test_index)):
            test_label_dic[test_index[i]] = test_label[i]

        return train_index,train_label_dic,test_index,test_label_dic,all_index,all_label
    
    def read_features(self,index):
        features = {}

        with open(self.parameter_path, 'r') as json_file:

            loaded_parameters = json.load(json_file)
            for subject in index:
                #print(subject)
                catogory_dic = {}
                for key in loaded_parameters:
                    file_key = key
                    if key == "finger_tap" or key == "toe_tap":
                        file_key = "other_mov"
                    # if key == "glance" and ("74" in self.parameter_path):
                    #     file_key = "glance_80-78order"
                    dic_create = {}
                    df = pd.read_excel(os.path.join(self.root_folder,file_key+".xlsx"))
                    #print(os.path.join(self.root_folder,key+".xlsx"))
                    #print(df.iloc[33:38,:])
                    #------用这个mean_supp来填充空的表格，新的表格会保存在save文件夹下------
                    #df = self.mean_supp(df)
                    #------这里可以保存填充后的表格------
                    #df.to_excel(os.path.join(self.save_folder,key+".xlsx"), index=False)
                    #print(df.iloc[33:38,:])
                    #return
                    #print(df.iloc[:,0])
                    #最下层的字典
                    for sub_key in loaded_parameters[key]:
                        matching_row = df[df.iloc[:, 0] == subject]
                        try:
                            #print(loaded_parameters[key][sub_key])
                            #print(sub_key)
                            dic_create[sub_key] = matching_row.iloc[0,loaded_parameters[key][sub_key]]
                        except IndexError:
                            print("Value not found or no matching row.",subject," ",key," ",sub_key)
                        
                    catogory_dic[key] = dic_create
                features[subject] = catogory_dic

        # print(features)
        return features
    
    def check_whole_features(self, index, label, fill_whole_feature=False):
        features = []
        subjects = []
        features_nan = {}
        subject_num = 0
        with open(self.parameter_path, 'r') as json_file:
            loaded_parameters = json.load(json_file)
            for idx, subject in enumerate(index):
                dic_create = [label[idx]]
                for key in loaded_parameters:
                    df = pd.read_excel(os.path.join(self.root_folder,key+".xlsx"))
                    start, end = loaded_parameters[key]['start'], loaded_parameters[key]['end']
                    matching_row = df[df.iloc[:, 0] == subject]
                    matching_row_np = np.array(matching_row)
                    try:
                        dic_create += matching_row_np[0, start:end].tolist()
                        if np.isnan(matching_row_np[0, start:end].astype('float')).any():
                            print(key)
                            if key not in features_nan.keys():
                                features_nan[key] = [subject]
                            else:
                                features_nan[key].append(subject)
                    except IndexError:
                        print("Value not found or no matching row.",subject," ",key)
                dic_create = np.array(dic_create)  
                features.append(dic_create)
                subjects.append(subject)
                if np.isnan(dic_create).all():
                    print("features of {} is nan.".format(subject))
                if np.isnan(dic_create).any():
                    subject_num += 1
                    # print("features of {} have nan.".format(subject))
            print("{} subjects have nan features.".format(subject_num))
            for mo in features_nan.keys():
                print("{} subjects have nan {}.".format(len(features_nan[mo]), mo))

            if fill_whole_feature:
                # KNN
                features = np.array(features)
                imputer = KNNImputer(n_neighbors=5)
                fill_features = imputer.fit_transform(features)

                feature_start = 0
                feature_end = 0
                for key in loaded_parameters:
                    subject_nan = features_nan[key]
                    df = pd.read_excel(os.path.join(self.root_folder,key+".xlsx"))
                    start, end = loaded_parameters[key]['start'], loaded_parameters[key]['end']
                    feature_end = feature_start + end - start
                    for idx, subject in enumerate(subjects):
                        if subject not in subject_nan:
                            continue
                        fill_fe = fill_features[idx][1:]
                        matching_row = df[df.iloc[:, 0] == subject]
                        matching_row.iloc[0, start:end] = fill_fe[feature_start:feature_end]
                        df[df.iloc[:, 0] == subject] = matching_row
                    feature_start = feature_end
                    df.to_excel(os.path.join(self.save_folder+"_KNN", key+".xlsx"), index=False)
                
                # add new indicator for nan feature
                for key in loaded_parameters:
                    subject_nan = features_nan[key]
                    df = pd.read_excel(os.path.join(self.root_folder,key+".xlsx"))
                    df['indicator'] = 1
                    start, end = loaded_parameters[key]['start'], loaded_parameters[key]['end']
                    for idx, subject in enumerate(subjects):
                        if subject not in subject_nan:
                           continue
                        else:
                            matching_row = df[df.iloc[:, 0] == subject]
                            matching_row.iloc[0, start:end] = 0
                            matching_row.iloc[0, -1] = 0
                            df[df.iloc[:, 0] == subject] = matching_row
                    df.to_excel(os.path.join(self.save_folder+"_indicator", key+".xlsx"), index=False)
        return

    def get_all(self, is_train=True):
        if is_train:
            return self.train_index, self.train_features, self.train_label
        else:
            return self.test_index, self.test_features, self.test_label
    
    def __len__(self):
        return len(self.train_index)+len(self.test_index)

    def __getitem__(self, key):
        if key in self.train_index:
            data = self.train_features[key]
            label = self.train_label[key]
        elif key in self.test_index:
            data = self.test_features[key]
            label = self.test_label[key]
        else:
            print("This subject don't exist!")
        return data,label
    
if __name__ == "__main__":
    save_dir = "/data0/wxy/PD/PD_early/data/PD_dataset/127PD-116HC数据_缺省已填充"
    file_path = "/data0/wxy/PD/PD_early/data/PD_dataset/127PD-116HC数据_缺省未填充"
    csv_path = "/data0/wxy/PD/PD_early/data/PD_dataset/subjects_126-113.xlsx"
    parameter_path = "/data0/wxy/PD/PD_early/data/PD_dataset/parameters_whole.json"
    pd_dataset = PDDataset(file_path,csv_path,parameter_path,save_folder=save_dir, fill_whole_feature=True)