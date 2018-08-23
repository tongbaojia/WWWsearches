import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from xgboost import XGBClassifier, XGBRegressor
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from sklearn import metrics
from collections import Counter
from tqdm import tqdm
import itertools
from sklearn.model_selection import train_test_split, GridSearchCV

#numpy settings
np.random.seed(14)

# Matplotlib settings
plt.style.use('ggplot')
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams.update({'font.size': 18})

class pRegion(object):
    def __init__(self):
        self.categorical_feats = ['l0_c', 'l1_c', 'l2_c', 'l0_isEl', 'l1_isEl', 'l2_isEl']
        self.xfeats = ['l0_pt', 'l1_pt', 'l2_pt', 'l0_eta', 'l1_eta', 'l2_eta', 'l0_phi', 'l1_phi', 'l2_phi']
        self.param_grid = {'learning_rate': [.1],
                            'max_depth': [5,7],
                            'min_child_weight': [1, 3],
                            'gamma': [0],
                            'subsample': [.8],
                            'colsample_bytree': [.8]}
        self.model = GridSearchCV(XGBClassifier(), param_grid = self.param_grid, n_jobs=1, cv=3)
        self.bin_probs = {}

    def fit(self, train_df):
        print 'training xfeats model...'
        self.model.fit(train_df[self.xfeats].values, train_df.is_sig.values)

        print 'calculating categorical probabilities...'
        gb = train_df.groupby(self.categorical_feats).groups
        for key in gb:
            df_short = train_df.loc[gb[key]]
            p = sum(df_short[df_short.is_sig == 1].weight) / float(sum(df_short.weight))
            self.bin_probs[key] = p

        print 'finding best combination of bins...'
        train_df = self.evaluate(train_df)
        self.bins = np.linspace(0.01, max(train_df.bin_p.values * train_df.preds.values), 22)

        # figure just for histogram binning
        fig = plt.figure(figsize=(10,6))
        gs = gridspec.GridSpec(2, 1)
        ax1 = plt.subplot(gs[0])
        nsig,_,_ = plt.hist(train_df[train_df.is_sig == 1].bin_p.values * train_df[train_df.is_sig == 1].preds.values, bins=self.bins, weights=train_df[train_df.is_sig == 1].weight.values, color='g')
        ax2 = plt.subplot(gs[1])
        nbkg,_,_ = plt.hist(train_df[train_df.is_sig == 0].bin_p.values * train_df[train_df.is_sig == 0].preds.values, bins=self.bins, weights=train_df[train_df.is_sig == 0].weight.values, color='b')
        plt.close(fig)

        self.best_signif = []
        for r in range(1, nsig.shape[0] + 1):
            for combo in tqdm(itertools.combinations(range(nsig.shape[0]), r)):
                ixes = np.array(combo)
                if sum(nbkg[ixes]) == 0:
                    continue
                signif = sum(nsig[ixes]) / np.sqrt(sum(nbkg[ixes]))
                if signif > 1.2:
                    self.best_signif.append((signif, ixes, sum(nsig[ixes]) + sum(nbkg[ixes])))
        self.best_signif = sorted(self.best_signif, key=lambda x: -x[0])

    def evaluate(self, train_df):
        train_df['preds'] = self.model.predict_proba(train_df[self.xfeats].values)[:,1]
        gb = train_df.groupby(self.categorical_feats).groups
        for key in gb:
            if key in self.bin_probs:
                train_df.loc[gb[key], 'bin_p'] = self.bin_probs[key]
            else:
                train_df.loc[gb[key], 'bin_p'] = 0.0
        return train_df

    def get_signif(self, train_df):
        fig = plt.figure(figsize=(10,6))
        gs = gridspec.GridSpec(2, 1)
        ax1 = plt.subplot(gs[0])
        nsig,_,_ = plt.hist(train_df[train_df.is_sig == 1].bin_p.values * train_df[train_df.is_sig == 1].preds.values, bins=self.bins, weights=train_df[train_df.is_sig == 1].weight.values, color='g')
        ax2 = plt.subplot(gs[1])
        nbkg,_,_ = plt.hist(train_df[train_df.is_sig == 0].bin_p.values * train_df[train_df.is_sig == 0].preds.values, bins=self.bins, weights=train_df[train_df.is_sig == 0].weight.values, color='b')
        plt.close(fig)

        return sum(nsig[self.best_signif[0][1]]) / np.sqrt(sum(nbkg[self.best_signif[0][1]])), sum(nsig[self.best_signif[0][1]]), sum(nbkg[self.best_signif[0][1]])


def get_data(data_dir):
    fs = [data_dir + f for f in os.listdir(data_dir) if f[0] != '.']
    df = pd.DataFrame()

    for f in fs:
        print 'reading', f
        new_df = pd.read_csv(f)
        df = pd.concat([df, new_df], ignore_index = True)
        df.index = range(len(df))

    return df

def add_cl_ix(df):
    df['is_sig'] = [1 if 'signal' in val else 0 for val in df.cl.values]
    return df


data_dir = 'data/correct/'
model = pRegion()

df = get_data(data_dir)
df = add_cl_ix(df)
train_df, test_df = train_test_split(df, test_size=0.25, random_state=42)

model.fit(train_df)
train_df = model.evaluate(train_df)
test_df = model.evaluate(test_df)

train_sig, train_sig_size, train_bkg_size = model.get_signif(train_df)
test_sig, test_sig_size, test_bkg_size = model.get_signif(test_df)

# plots
print 'train plots'

bins = model.bins

fig = plt.figure(figsize=(10,6))
nbkg,_,_ = plt.hist(train_df[train_df.is_sig == 0].bin_p.values * train_df[train_df.is_sig == 0].preds.values, bins=bins, weights=4./3.*train_df[train_df.is_sig == 0].weight.values, color='b')
nsig,_,_ = plt.hist(train_df[train_df.is_sig == 1].bin_p.values * train_df[train_df.is_sig == 1].preds.values, bins=bins, weights=4./3.*train_df[train_df.is_sig == 1].weight.values, color='g')
plt.ylabel('Weighted Counts')
plt.xlabel(r'$p_{signal}$')
plt.annotate(r'$\sigma$: ' + str(round(train_sig * np.sqrt(4./3.), 3)), xy=(.7,.6), xycoords='axes fraction')
plt.legend(handles = [mpatches.Patch(color='b', label='Background'),
                        mpatches.Patch(color='g', label='Signal')])
plt.title('Train')
plt.tight_layout()
plt.savefig('plots/train.pdf')
plt.close(fig)



print 'test plots'

fig = plt.figure(figsize=(10,6))
nbkg,_,_ = plt.hist(test_df[test_df.is_sig == 0].bin_p.values * test_df[test_df.is_sig == 0].preds.values, bins=bins, weights=4*test_df[test_df.is_sig == 0].weight.values, color='b')
nsig,_,_ = plt.hist(test_df[test_df.is_sig == 1].bin_p.values * test_df[test_df.is_sig == 1].preds.values, bins=bins, weights=4*test_df[test_df.is_sig == 1].weight.values, color='g')
plt.ylabel('Weighted Counts')
plt.xlabel(r'$p_{signal}$')
plt.annotate(r'$\sigma$: ' + str(round(test_sig * np.sqrt(4.), 3)), xy=(.7,.6), xycoords='axes fraction')
plt.legend(handles = [mpatches.Patch(color='b', label='Background'),
                        mpatches.Patch(color='g', label='Signal')])
plt.title('Test')
plt.tight_layout()
plt.savefig('plots/test.pdf')
plt.close(fig)






