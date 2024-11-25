import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import os
from data.block import AbnormalGaitDataset
import torch.distributed as dist
import numpy as np
from torch.utils.data.distributed import DistributedSampler

def data_loaders(dataset,batch_size):
    if dataset == "human36":
        root_folder = "/data1/bingzi/MyGait/data/human36"
    elif dataset == "Gait":
        root_folder = "/data1/bingzi/MyGait/data/Pathological_Gaits_Processed"
        #root_folder = '/data1/bingzi/MyGait/data/test_part'
        #root_folder = "/data1/bingzi/MyGait/data/selected_data"
    train_dataset = AbnormalGaitDataset(root_folder, train=True)
    val_dataset = AbnormalGaitDataset(root_folder, train=False)
    
    train = np.array(train_dataset.data)
    train = np.nan_to_num(train)
    val = np.array(val_dataset.data)
    val = np.nan_to_num(val)

    """ train = train_dataset.data
    val = val_dataset.data """


    print("Train dataset size:", len(train))
    print("Val dataset size:", len(val))

    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              pin_memory=False,
                              num_workers = 8,
                              drop_last = True,
                              sampler=train_sampler
                              )
    val_sampler = DistributedSampler(val_dataset)                        
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            pin_memory=False,
                            num_workers = 8,
                            drop_last = True,
                            sampler = val_sampler 
                            )
    return train_loader, val_loader,train,val,len(train),len(val)


def load_data_and_data_loaders(dataset, batch_size):
    #training_data, validation_data,train_data_size,val_data_size = load_gait_block(dataset)
    training_loader, validation_loader,training_data,validation_data,train_data_size,val_data_size = data_loaders(dataset,batch_size) 
    #x_train_var = np.var(training_data)
    #y_train_var = np.var(validation_data)

    #return training_data, validation_data, training_loader, validation_loader, x_train_var,y_train_var,train_data_size,val_data_size
    return training_data, validation_data, training_loader, validation_loader,train_data_size,val_data_size


def readable_timestamp():
    return time.ctime().replace('  ', ' ').replace(
        ' ', '_').replace(':', '_').lower()


def save_model_and_results(model, results, hyperparameters, timestamp,epoch):
    SAVE_MODEL_PATH = os.getcwd() + '/checkpoint_training'
    if not os.path.exists(SAVE_MODEL_PATH):
        os.makedirs(SAVE_MODEL_PATH)

    print('Results will be saved in ./checkpoint_training')
    results_to_save = {
        'model': model.module.state_dict(),
        'results': results,
        'hyperparameters': hyperparameters
    }
    print("storage begin!!!!!!!!!")
    torch.save(results_to_save,
            SAVE_MODEL_PATH + '/transformer_prediction_' + timestamp +'.pth')
            #SAVE_MODEL_PATH + '/transformer_prediction_' + timestamp +"_epoch_" +str(epoch+1)+'.pth')
    print("storage finished!!!!!!!!!")