import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch


class DatasetLoader():
    def __init__(self, data_path, window_size, ts_num,dataset):

        self.data_path = data_path
        self.dataset_name = dataset
        self.window_size = window_size
        self.ts_num = ts_num
        if dataset == 'ecg':


            self.ecg_dataset_name = ['chfdb_chf01_275.pkl',
                                    'chfdb_chf13_45590.pkl',
                                    'chfdbchf15.pkl',
                                    'ltstdb_20221_43.pkl',
                                    'ltstdb_20321_240.pkl',
                                    'mitdb__100_180.pkl',
                                    'qtdbsel102.pkl',
                                    'stdb_308_0.pkl',
                                    'xmitdb_x108_0.pkl']


            self.dataset = get_ECG_dataset(data_path=self.data_path,
                                      dataset_name=self.ecg_dataset_name,
                                      ts_num=self.ts_num, normalized=True)

        elif dataset == 'gesture':
            self.dataset = get_Gesture_dataset(
                data_path=self.data_path,
                normalized=True)

        elif dataset == 'pd':
            self.dataset = get_Power_Demand_dataset(
                data_path=self.data_path,
                normalized=True)

        else:
            raise ValueError("dataset must be 'ecg', 'gesture' or 'pd'")


        self.__sliding_window_generation(window_size=window_size)

    def __sliding_window_generation(self,window_size):
            # self.train_set, self.test_set, self.validation_set = sliding_window_generation(dataset=self.dataset, window_size=window_size)

        self.train_set, self.test_set = sliding_window_generation(dataset=self.dataset, window_size=window_size)


    def train_loader_generation(self, batch_size, shuffle=True):

        batch_train_data = torch.utils.data.TensorDataset(torch.tensor(self.train_set['samples'].astype(np.float32)),
                                                          torch.tensor(self.train_set['labels'].astype(np.float32)))
        train_loader = torch.utils.data.DataLoader(dataset=batch_train_data, batch_size=batch_size,
                                                         shuffle=shuffle)
        return train_loader

    def val_test_loader_generation(self,batch_size, shuffle=False):

        batch_test_data = torch.utils.data.TensorDataset(torch.tensor(self.test_set['samples'].astype(np.float32)),
                                                          torch.tensor(self.test_set['labels'].astype(np.float32)))
        test_loader = torch.utils.data.DataLoader(dataset=batch_test_data, batch_size=batch_size,
                                                   shuffle=shuffle, drop_last = False)

        # batch_val_data = torch.utils.data.TensorDataset(torch.tensor(self.validation_set['samples'].astype(np.float32)),
        #                                                 torch.tensor(self.validation_set['labels'].astype(np.float32)))
        # val_loader = torch.utils.data.DataLoader(dataset=batch_val_data, batch_size=batch_size,
        #                                          shuffle=shuffle)

        # return val_loader, test_loader

        return test_loader

    def plot(self, data_type='train', dataset_name=None):
        data =None
        labels=None
        if data_type=='train':
            data = self.dataset['train_data']
        elif data_type=='val':
            data = self.dataset['validate_data']
            labels = self.dataset['validate_label']
        elif data_type=='test':
            data = self.dataset['test_data']
            labels= self.dataset['test_label']

        # cl=['gray', 'black']

        for i in range (data.shape[0]):
            # plt.plot(data[i], color=cl[i])
            plt.plot(data[i])
            print(len(data[i]))
        if data_type=='test':
            start = 0
            end=0
            for k in range (len(labels)-1):
                if labels[k] ==0. and labels[k+1]==1:
                    start = k+1
                if labels[k] ==1. and labels[k+1]==0:
                    end = k
            plt.axvspan(start, end,0.,1., alpha=0.5, color='lightgreen')

        plt.title(str(data_type)+ ' data of dataset ' +str(dataset_name))
        plt.xlabel('time point')
        plt.ylabel('values')
        plt.show()

        if data_type in ['val', 'test']:
            plt.plot(labels)
            plt.xlabel('time point')
            plt.ylabel('anomaly labels')
            plt.title(str(data_type)+ ' label')
            plt.show()

    def statistics(self):
        if isinstance (self.dataset['test_label'], np.ndarray):
            label = list(self.dataset['test_label'])
            n_anomalies = label.count(1)
        else: n_anomalies = self.dataset['test_label'].count(1)
        perent_anomalies = n_anomalies/len(self.dataset['test_label'])
        return self.dataset['train_data'].shape, self.dataset['test_data'].shape, perent_anomalies
        pass


        #plot val data


def get_ECG_dataset(data_path,dataset_name,ts_num, normalized = True, validation_ratio=0.2):
    print(dataset_name[ts_num])
    trainfile = open(data_path + 'labeled/train/' + dataset_name[ts_num], 'rb')
    testfile = open(data_path + 'labeled/test/' + dataset_name[ts_num], 'rb')

    tr_data = pd.DataFrame(pd.read_pickle(trainfile))

    train_data = tr_data[[0, 1]].to_numpy()
    train_data = train_data.T

    te_data = pd.DataFrame(pd.read_pickle(testfile))
    test_data = te_data[[0, 1]].to_numpy()
    test_data = test_data.T
    test_label = te_data[[2]].to_numpy()
    test_label = np.reshape(test_label, newshape=(test_label.shape[0],))

    if normalized:
        max_x_train = np.max(train_data[0, :])
        min_x_train = np.min(train_data[0, :])
        max_y_train = np.max(train_data[1, :])
        min_y_train = np.min(train_data[1, :])
        f_max = max(max_x_train, max_y_train)
        f_min = min(min_x_train, min_y_train)
        for i in range(train_data.shape[1]):
            train_data[0][i] = ((train_data[0][i] - f_min) / (f_max - f_min))
            train_data[1][i] = ((train_data[1][i] - f_min) / (f_max - f_min))

        for j in range(test_data.shape[1]):
            test_data[0][j] = ((test_data[0][j] - f_min) / (f_max - f_min))
            test_data[1][j] = ((test_data[1][j] - f_min) / (f_max - f_min))

    # split to validation and testset

    n_validate = int(test_data.shape[1] * validation_ratio)
    validate_data = test_data[:, 0: n_validate]
    # test_data = test_data[:, n_validate:]
    validate_label = test_label[0: n_validate]
    # test_label = test_label[n_validate:]

    dataset = {'train_data': train_data, 'test_data': test_data, 'test_label': test_label,
               'validate_data': validate_data, 'validate_label': validate_label}
    return dataset

def get_Gesture_dataset(
        data_path,
        normalized=True,
        validation_ratio=0.2
):
    train_path = os.path.join(
        data_path,
        'labeled',
        'train',
        'ann_gun_CentroidA.pkl'
    )

    test_path = os.path.join(
        data_path,
        'labeled',
        'test',
        'ann_gun_CentroidA.pkl'
    )

    print('Train path:', train_path)
    print('Test path:', test_path)

    tr_data = pd.DataFrame(
        pd.read_pickle(train_path)
    )

    te_data = pd.DataFrame(
        pd.read_pickle(test_path)
    )

    train_data = (
        tr_data[[0, 1]]
        .to_numpy(dtype=np.float32)
        .T
    )

    test_data = (
        te_data[[0, 1]]
        .to_numpy(dtype=np.float32)
        .T
    )

    test_label = (
        te_data[[2]]
        .to_numpy()
        .reshape(-1)
    )

    if normalized:
        f_max = np.max(train_data)
        f_min = np.min(train_data)

        if f_max != f_min:
            train_data = (
                train_data - f_min
            ) / (
                f_max - f_min
            )

            test_data = (
                test_data - f_min
            ) / (
                f_max - f_min
            )

    n_validate = int(
        test_data.shape[1] * validation_ratio
    )

    validate_data = test_data[:, :n_validate]
    validate_label = test_label[:n_validate]

    return {
        'train_data': train_data,
        'test_data': test_data,
        'test_label': test_label,
        'validate_data': validate_data,
        'validate_label': validate_label
    }

def get_Power_Demand_dataset(
        data_path,
        normalized=True,
        validation_ratio=0.2
):
    train_path = os.path.join(
        data_path,
        'labeled',
        'train',
        'power_data.pkl'
    )

    test_path = os.path.join(
        data_path,
        'labeled',
        'test',
        'power_data.pkl'
    )

    print('Train path:', train_path)
    print('Test path:', test_path)

    tr_data = pd.DataFrame(
        pd.read_pickle(train_path)
    )

    te_data = pd.DataFrame(
        pd.read_pickle(test_path)
    )

    train_data = (
        tr_data[[0]]
        .to_numpy(dtype=np.float32)
        .T
    )

    test_data = (
        te_data[[0]]
        .to_numpy(dtype=np.float32)
        .T
    )

    test_label = (
        te_data[[1]]
        .to_numpy()
        .reshape(-1)
    )

    if normalized:
        f_max = np.max(train_data)
        f_min = np.min(train_data)

        if f_max != f_min:
            train_data = (
                train_data - f_min
            ) / (
                f_max - f_min
            )

            test_data = (
                test_data - f_min
            ) / (
                f_max - f_min
            )

    n_validate = int(
        test_data.shape[1] * validation_ratio
    )

    validate_data = test_data[:, :n_validate]
    validate_label = test_label[:n_validate]

    return {
        'train_data': train_data,
        'test_data': test_data,
        'test_label': test_label,
        'validate_data': validate_data,
        'validate_label': validate_label
    }



def sliding_window_generation(dataset, window_size):
    train_samples, reconstruction_label = training_samples_generation(train_data=dataset['train_data'], window_size=window_size)
    trainset = {'samples': train_samples, 'labels':reconstruction_label}

    test_samples, anomaly_labels = testing_samples_generation(test_data=dataset['test_data'],test_label=dataset['test_label'],window_size=window_size)
    testset = {'samples':test_samples, 'labels': anomaly_labels}

    # validate_samples, val_anomaly_labels = testing_samples_generation(test_data=dataset['validate_data'], test_label=dataset['validate_label'], window_size=window_size)
    # validationset = {'samples': validate_samples, 'labels': val_anomaly_labels}

    # return trainset,testset,validationset
    return trainset, testset

def training_samples_generation(train_data, window_size):

    # generate training data
    dimension = train_data.shape[0]

    samples = np.zeros(shape=(train_data.shape[1] - window_size + 1, window_size, dimension))
    reconstruction_label = np.zeros(shape=(train_data.shape[1] - window_size + 1, window_size, dimension))

    for i in range(0, train_data.shape[-1] - window_size + 1):
        # generate data and reconstructed_label
        reconstruction_label[i] = np.copy(train_data[:, i:i + window_size].T)
        samples[i] = np.copy(train_data[:, i:i + window_size].T)

    return samples, reconstruction_label

def testing_samples_generation(test_data, test_label, window_size):

    dimension = test_data.shape[0]

    samples = np.zeros(shape=(test_data.shape[1] - window_size + 1, window_size, dimension))
    test_labels = np.zeros(shape=(test_data.shape[1] - window_size + 1, 1, window_size))

    for i in range(0, test_data.shape[-1] - window_size + 1):
        # generate data and anomaly labels
        test_labels[i] = np.copy(test_label[i:i + window_size])
        samples[i] = np.copy(test_data[:, i:i + window_size].T)

    return samples, test_labels

