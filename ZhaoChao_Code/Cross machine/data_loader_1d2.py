
from torchvision import datasets, transforms
import torch
import numpy as np
from scipy.fftpack import fft
import scipy.io as scio
import math
import os

def wgn(x, snr):
    Ps = np.sum(abs(x)**2,axis=1)/len(x)
    Pn = Ps/(10**((snr/10)))
    row,columns=x.shape
    Pn = np.repeat(Pn.reshape(-1,1),columns, axis=1)

    noise = np.random.randn(row,columns) * np.sqrt(Pn)
    signal_add_noise = x + noise
    return signal_add_noise


def zscore(Z):
    Zmax, Zmin = Z.max(axis=1), Z.min(axis=1)
    Z = (Z - Zmin.reshape(-1,1)) / (Zmax.reshape(-1,1) - Zmin.reshape(-1,1))
    return Z


def redu_sample_np(data,class_num):
    row,col=data.shape#7200,512
    sca=800

    A = np.zeros((sca*class_num, col))  # (1800,512)
    for i in range(0, class_num):
        A[sca * i:sca * i + sca] = data[800 * i:800 * i + sca]

    return A

def redu_sample_to(data,class_num):
    row,col=data.shape#7200,512
    sca=800

    A = torch.zeros((sca*class_num, col))  # (1800,512)
    for i in range(0, class_num):
        A[sca * i:sca * i + sca] = data[800 * i:800 * i + sca]

    return A

def min_max(Z):
    Zmin = Z.min(axis=1)

    Z = np.log(Z - Zmin.reshape(-1, 1) + 1)
    return Z


def data_load(root_path, datasetname, domain_label):
    fft1 = False
    class_num = 3

    data = scio.loadmat(root_path)

    # Load the three variables from the .mat file
    train_fea_Normal = data['data_Normal']
    train_fea_IR = data['data_IR']
    train_fea_OR = data['data_OR']

    # Apply preprocessing (e.g., normalization)
    if fft1:
        train_fea_IR = zscore(min_max(abs(fft(train_fea_IR))[:, :1024]))
        train_fea_Normal = zscore(min_max(abs(fft(train_fea_Normal))[:, :1024]))
        train_fea_OR = zscore(min_max(abs(fft(train_fea_OR))[:, :1024]))
    else:
        train_fea_IR = zscore(train_fea_IR)
        train_fea_Normal = zscore(train_fea_Normal)
        train_fea_OR = zscore(train_fea_OR)

    # Generate labels for each category
    train_label_Normal = torch.zeros((len(train_fea_Normal), 2))
    for i in range(len(train_fea_Normal)):
        train_label_Normal[i][0] = 0  # Class 0: Normal
        train_label_Normal[i][1] = domain_label


    train_label_IR = torch.zeros((len(train_fea_IR), 2))
    for i in range(len(train_fea_IR)):
        train_label_IR[i][0] = 1  # Class 1: Inner Race Fault
        train_label_IR[i][1] = domain_label


    train_label_OR = torch.zeros((len(train_fea_OR), 2))
    for i in range(len(train_fea_OR)):
        train_label_OR[i][0] = 2  # Class 2: Outer Race Fault
        train_label_OR[i][1] = domain_label

    # Combine features and labels
    train_fea = np.vstack((train_fea_IR,
                           train_fea_Normal,
                           train_fea_OR))
    train_label = torch.cat([train_label_IR, train_label_Normal, train_label_OR], dim=0)

    return train_fea, train_label


def load_training(dataset1,dataset2,dataset3,batch_size, kwargs):



    class_num=3

    datasetPath = 'F:/PhdDoc/06Code/DT/Dataset/'
    root_path1 = datasetPath + dataset1 + '/PerK1Revolution.mat'
    root_path2 = datasetPath + dataset2 + '/PerK1Revolution.mat'
    root_path3 = datasetPath + dataset3 + '/PerK1Revolution.mat'


    train_fea_1, train_label_1 = data_load(root_path1, dataset1, domain_label=0)
    train_fea_2, train_label_2 = data_load(root_path2, dataset2, domain_label=1)
    train_fea_3, train_label_3 = data_load(root_path3, dataset3, domain_label=2)


    train_fea = np.vstack((train_fea_1, train_fea_2, train_fea_3))
    train_label = torch.cat([train_label_1, train_label_2, train_label_3], dim=0)





    train_label = train_label.long()
    train_fea=torch.from_numpy(train_fea)
    train_fea= torch.tensor(train_fea, dtype=torch.float32)
    data = torch.utils.data.TensorDataset(train_fea, train_label)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader




def load_testing( dataset,batch_size, kwargs):


    class_num = 3

    datasetPath = 'F:/PhdDoc/06Code/DT/Dataset/'
    root_path1 = datasetPath + dataset + '/PerK1Revolution.mat'

    train_fea, train_label = data_load(root_path1, dataset, domain_label=3)

    train_label = train_label.long()
    train_fea = torch.from_numpy(train_fea)
    train_fea = torch.tensor(train_fea, dtype=torch.float32)
    data = torch.utils.data.TensorDataset(train_fea, train_label)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=False, **kwargs)

    return train_loader




def load_source_training(dataset1,dataset2,dataset3,batch_size, kwargs):



    class_num=3



    root_path1 = '/home/zhaochao/research/DTL/data/' + dataset1 + 'data' + str(class_num) + '.mat'
    root_path2 = '/home/zhaochao/research/DTL/data/' + dataset2 + 'data' + str(class_num) + '.mat'
    root_path3 = '/home/zhaochao/research/DTL/data/' + dataset3 + 'data' + str(class_num) + '.mat'


    train_fea_1, train_label_1 = data_load(root_path1, dataset1, domain_label=0)
    train_fea_2, train_label_2 = data_load(root_path2, dataset2, domain_label=0)
    train_fea_3, train_label_3 = data_load(root_path3, dataset3, domain_label=0)


    train_fea = np.vstack((train_fea_1, train_fea_2, train_fea_3))
    train_label = torch.cat([train_label_1, train_label_2, train_label_3], dim=0)



    train_label = train_label.long()
    train_fea=torch.from_numpy(train_fea)
    train_fea= torch.tensor(train_fea, dtype=torch.float32)
    data = torch.utils.data.TensorDataset(train_fea, train_label)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader




def load_target_training(AP,SNR,dataset1,dataset2,dataset3,batch_size, kwargs):



    class_num=3



    root_path1 = '/home/zhaochao/research/DTL/data/' + dataset1 + 'data' + str(class_num) + '.mat'
    root_path2 = '/home/zhaochao/research/DTL/data/' + dataset2 + 'data' + str(class_num) + '.mat'
    root_path3 = '/home/zhaochao/research/DTL/data/' + dataset3 + 'data' + str(class_num) + '.mat'


    train_fea_1, train_label_1 = data_load_AP(AP,SNR,root_path1, dataset1, domain_label=1)
    train_fea_2, train_label_2 = data_load_AP(AP,SNR,root_path2, dataset2, domain_label=1)
    train_fea_3, train_label_3 = data_load_AP(AP,SNR,root_path3, dataset3, domain_label=1)


    train_fea = np.vstack((train_fea_1, train_fea_2, train_fea_3))
    train_label = torch.cat([train_label_1, train_label_2, train_label_3], dim=0)



    train_label = train_label.long()
    train_fea=torch.from_numpy(train_fea)
    train_fea= torch.tensor(train_fea, dtype=torch.float32)
    data = torch.utils.data.TensorDataset(train_fea, train_label)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader



def data_load_AP(AP,SNR,root_path,datasetname,domain_label):


    fft1=False
    class_num=3


    data = scio.loadmat(root_path)

    if (datasetname=='M_CWRU'):

        dir1 = 'load0_train'
        dir2 = 'load1_train'
        dir3 = 'load2_train'
        dir4 = 'load3_train'

        if fft1 == True:
            train_fea_1 = zscore((min_max(abs(fft(wgn(AP * data[dir1], SNR)))[:, 0:1600])))
        if fft1 == False:
            train_fea_1 = zscore(wgn(AP * data[dir1], SNR))


        train_label_1 = torch.zeros((800 * class_num, 2))
        for i in range(800 * class_num):
            train_label_1[i][0] = i // 800

            train_label_1[i][1] = domain_label

        if fft1 == True:
            train_fea_2 = zscore((min_max(abs(fft(wgn(AP * data[dir2], SNR)))[:, 0:1600])))

        if fft1 == False:
            train_fea_2 = zscore(wgn(AP * data[dir2], SNR))

        train_label_2 = torch.zeros((800 * class_num, 2))
        for i in range(800 * class_num):
            train_label_2[i][0] = i // 800

            train_label_2[i][1] = domain_label


        if fft1 == True:
            train_fea_3 = zscore((min_max(abs(fft(wgn(AP * data[dir3], SNR)))[:, 0:1600])))
        if fft1 == False:
            train_fea_3 = zscore(wgn(AP * data[dir3], SNR))


        train_label_3 = torch.zeros((800 * class_num, 2))
        for i in range(800 * class_num):
            train_label_3[i][0] = i // 800

            train_label_3[i][1] = domain_label


        if fft1 == True:
            train_fea_4 = zscore((min_max(abs(fft(wgn(AP * data[dir4], SNR)))[:, 0:1600])))

        if fft1 == False:
            train_fea_4 = zscore(wgn(AP * data[dir4], SNR))


        train_label_4 = torch.zeros((800 * class_num, 2))
        for i in range(800 * class_num):
            train_label_4[i][0] = i // 800

            train_label_4[i][1] = domain_label


        train_fea = np.vstack((train_fea_1, train_fea_2, train_fea_3, train_fea_4))
        train_label = torch.cat([train_label_1, train_label_2, train_label_3, train_label_4], dim=0)


    if (datasetname=='M_HUST'):

        dir1 = 'load0_train'
        dir2 = 'load1_train'
        dir3 = 'load2_train'
        dir4 = 'load3_train'
        dir5 = 'load4_train'

        if fft1 == True:
            train_fea_1 = zscore((min_max(abs(fft(wgn(AP * data[dir1], SNR)))[:, 0:1600])))
        if fft1 == False:
            train_fea_1 = zscore(wgn(AP * data[dir1], SNR))

        train_label_1 = torch.zeros((800 * class_num, 2))
        for i in range(800 * class_num):
            train_label_1[i][0] = i // 800

            train_label_1[i][1] = domain_label

        if fft1 == True:
            train_fea_2 = zscore((min_max(abs(fft(wgn(AP * data[dir2], SNR)))[:, 0:1600])))

        if fft1 == False:
            train_fea_2 = zscore(wgn(AP * data[dir2], SNR))

        train_label_2 = torch.zeros((800 * class_num, 2))
        for i in range(800 * class_num):
            train_label_2[i][0] = i // 800

            train_label_2[i][1] = domain_label

        if fft1 == True:
            train_fea_3 = zscore((min_max(abs(fft(wgn(AP * data[dir3], SNR)))[:, 0:1600])))
        if fft1 == False:
            train_fea_3 = zscore(wgn(AP * data[dir3], SNR))

        train_label_3 = torch.zeros((800 * class_num, 2))
        for i in range(800 * class_num):
            train_label_3[i][0] = i // 800

            train_label_3[i][1] = domain_label

        if fft1 == True:
            train_fea_4 = zscore((min_max(abs(fft(wgn(AP * data[dir4], SNR)))[:, 0:1600])))

        if fft1 == False:
            train_fea_4 = zscore(wgn(AP * data[dir4], SNR))

        train_label_4 = torch.zeros((800 * class_num, 2))
        for i in range(800 * class_num):
            train_label_4[i][0] = i // 800

            train_label_4[i][1] = domain_label




        if fft1 == True:
            train_fea_5 = zscore((min_max(abs(fft(wgn(AP * data[dir5], SNR)))[:, 0:1600])))


        if fft1 == False:
            train_fea_5 = zscore(wgn(AP * data[dir5], SNR))


        train_label_5 = torch.zeros((800 * class_num, 2))
        for i in range(800 * class_num):
            train_label_5[i][0] = i // 800

            train_label_5[i][1] = domain_label




        train_fea = np.vstack((train_fea_1, train_fea_2, train_fea_3, train_fea_4, train_fea_5))
        train_label = torch.cat([train_label_1, train_label_2, train_label_3, train_label_4, train_label_5], dim=0)


    if datasetname=='M_JNU':

        dir1 = 'load0_train'
        dir2 = 'load1_train'
        dir3 = 'load2_train'


        if fft1 == True:
            train_fea_1 = zscore((min_max(abs(fft(wgn(AP * data[dir1], SNR)))[:, 0:1600])))
        if fft1 == False:
            train_fea_1 = zscore(wgn(AP * data[dir1], SNR))

        train_label_1 = torch.zeros((800 * class_num, 2))
        for i in range(800 * class_num):
            train_label_1[i][0] = i // 800

            train_label_1[i][1] = domain_label

        if fft1 == True:
            train_fea_2 = zscore((min_max(abs(fft(wgn(AP * data[dir2], SNR)))[:, 0:1600])))

        if fft1 == False:
            train_fea_2 = zscore(wgn(AP * data[dir2], SNR))

        train_label_2 = torch.zeros((800 * class_num, 2))
        for i in range(800 * class_num):
            train_label_2[i][0] = i // 800

            train_label_2[i][1] = domain_label

        if fft1 == True:
            train_fea_3 = zscore((min_max(abs(fft(wgn(AP * data[dir3], SNR)))[:, 0:1600])))
        if fft1 == False:
            train_fea_3 = zscore(wgn(AP * data[dir3], SNR))

        train_label_3 = torch.zeros((800 * class_num, 2))
        for i in range(800 * class_num):
            train_label_3[i][0] = i // 800

            train_label_3[i][1] = domain_label



        train_fea = np.vstack((train_fea_1, train_fea_2, train_fea_3))
        train_label = torch.cat([train_label_1, train_label_2, train_label_3], dim=0)


    if (datasetname == 'M_SCP')or (datasetname == 'M_IMS')or (datasetname == 'M_XJTU')  :

        dir1 = 'load0_train'


        if fft1 == True:
            train_fea_1 = zscore((min_max(abs(fft(wgn(AP * data[dir1], SNR)))[:, 0:1600])))

        if fft1 == False:
            train_fea_1 = zscore(wgn(AP * data[dir1], SNR))


        train_label_1 = torch.zeros((800 * class_num, 2))
        for i in range(800 * class_num):
            train_label_1[i][0] = i // 800

            train_label_1[i][1] = domain_label


        train_fea = train_fea_1
        train_label =train_label_1




    return train_fea,train_label