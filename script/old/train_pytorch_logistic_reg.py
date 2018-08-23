import pandas as pd
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader



def get_data(data_dir):
    fs = [data_dir + f for f in os.listdir(data_dir) if ('signal' in f or 'WZ' in f) and f[0] != '.']
    df = pd.DataFrame()

    for f in fs:
        print f
        new_df = pd.read_csv(f)
        df = pd.concat([df, new_df], ignore_index = True)
        df.index = range(len(df))

    return df

def add_cl_ix(df):
    df['is_sig'] = [1 if 'signal' in val else 0 for val in df.cl.values]
    return df

class WWdataset(Dataset):

    def __init__(self, data_dir):
        self.data_dir = data_dir
        dataset = get_data(data_dir)
        self.dataset = add_cl_ix(dataset)

        self.input_vars = [col for col in self.dataset.columns if not col in ['runNumber', 'lbNumber', 'eventNumber', 'SFOS', 'is_sig', 'weight', 'cl', 'preds']]

        self.target_var = ['is_sig']
        self.weight_var = ['weight']

        self.input_np = self.dataset[self.input_vars].as_matrix().astype(dtype=np.float32)
        self.target_np = self.dataset[self.target_var].as_matrix().astype(dtype=int)
        self.weight_np =self.dataset[self.weight_var].as_matrix().astype(dtype=np.float32)

        self.inputs = torch.from_numpy(self.input_np)
        self.target = torch.from_numpy(self.target_np)
        self.weight = torch.from_numpy(self.weight_np)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        inputs = self.inputs[idx]
        target = self.target[idx]
        weight = self.weight[idx]
        return inputs, target, weight

    def n_input(self):
        return len(self.input_vars)

def net_logistic_regression(n_input):
    model = torch.nn.Sequential(
        torch.nn.Linear(n_input, 1),
        # torch.nn.Sigmoid()
    )
    return model

def net_deep_logistic_regression(n_input):
    model = torch.nn.Sequential(
        torch.nn.Linear(n_input, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1),
    )
    return model

# def plot():
#     pass
    # x_bins = np.linspace(0, max(df.preds), 30)
    # fig = plt.figure(figsize=(10,6))
    # gs = gridspec.GridSpec(3, 1)
    # ax = plt.subplot(gs[0,:])
    # plt.title('XGBClassifier Separability')
    # n_bkg,bins,paint = plt.hist(df[df.is_sig == 0].preds, bins=x_bins, weights=df[df.is_sig == 0].weight, color='r')
    # plt.yscale('log')
    # plt.ylabel(r'Weighted Background Counts', size=9)
    # plt.legend(handles=[mpatches.Patch(color='red', label='Background')])
    # ax1 = plt.subplot(gs[1,:])
    # n_sig,bins,paint = plt.hist(df[df.is_sig == 1].preds, bins=x_bins, weights=df[df.is_sig == 1].weight, color='g')
    # plt.yscale('log')
    # plt.ylabel(r'Weighted Signal Counts', size=9)
    # plt.legend(handles=[mpatches.Patch(color='green', label='Signal')])
    # ax2 = plt.subplot(gs[2,:])
    # plt.bar((x_bins[:-1] + x_bins[1:]) / 2., n_sig / np.sqrt(n_bkg), width=x_bins[1] - x_bins[0], color='k')
    # plt.ylabel(r'Significance ($S/\sqrt{B})$', size=9)
    # plt.xlabel('Probability Event is a Signal')
    #
    # plt.tight_layout()
    # plt.savefig('plots/preds_train.pdf')
    # plt.close(fig)
    #
    #
    # fpr, tpr, thresholds = metrics.roc_curve(df.is_sig.values, df.preds.values, pos_label=1)
    # fig = plt.figure(figsize=(6,6))
    # plt.plot(fpr, tpr)
    # plt.title('XGBClassifier ROC')
    # plt.annotate('Area: ' + str(round(metrics.auc(fpr, tpr), 2)), xy=(.8,.2), xycoords='axes fraction')
    # plt.xlim((0,1))
    # plt.ylim((0,1))
    # plt.plot([0,1], [0,1], linestyle='--')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    #
    # plt.savefig('plots/roc_curve.pdf')
    # plt.close(fig)

if __name__ == '__main__':
    data_dir = argv[1]

    print "Start Loading"

    in_dataset = WWdataset(data_dir)
    trainloader = torch.utils.data.DataLoader(in_dataset, batch_size=200, shuffle=True, num_workers=2)

    print "Finishing loading"
    print
    print "Declaring variables net"
    net = net_deep_logistic_regression(in_dataset.n_input())
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss()
    criterion = nn.NLLLoss(reduce=False)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    print "Start Training"

    for epoch in range(3):
        print
        print "epoch: ", epoch
        running_loss = 0.
        for i, data in enumerate(trainloader):
            inputs, label, weight = data

            # wrap them in Variable
            inputs, label, weight = Variable(inputs), Variable(label), Variable(weight)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output = net(inputs)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % 200 == 199:    # print every 2000 mini-batches
                print "batch:  {}, loss: {}".format(i+1, running_loss/(i+1))

    print "Finished Training"
