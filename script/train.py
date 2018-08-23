import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from xgboost import XGBClassifier
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from sklearn import metrics


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

def train(df, x_cols):
    y_col = 'is_sig'
    model = XGBClassifier()

    print 'training...'
    model.fit(df[x_cols].values, df[y_col].values, sample_weight = df.weight.values)
    df['preds'] = model.predict_proba(df[x_cols].values)[:,1]
    print 'done training'

    return df, model


data_dir = '../data/' # Modify this

df = get_data(data_dir)
df = add_cl_ix(df)

df = df[df.SOFS == 0]
x_cols = [col for col in df.columns if not col in ['runNumber', 'lbNumber', 'eventNumber', 'SFOS', 'is_sig', 'weight', 'cl', 'preds']]
df,model = train(df, x_cols)

x_bins = np.linspace(0, max(df.preds), 30)
fig = plt.figure(figsize=(10,6))
gs = gridspec.GridSpec(3, 1)
ax = plt.subplot(gs[0,:])
plt.title('XGBClassifier Separability')
n_bkg,bins,paint = plt.hist(df[df.is_sig == 0].preds, bins=x_bins, weights=df[df.is_sig == 0].weight, color='r')
plt.yscale('log')
plt.ylabel(r'Weighted Background Counts', size=9)
plt.legend(handles=[mpatches.Patch(color='red', label='Background')])
ax1 = plt.subplot(gs[1,:])
n_sig,bins,paint = plt.hist(df[df.is_sig == 1].preds, bins=x_bins, weights=df[df.is_sig == 1].weight, color='g')
plt.yscale('log')
plt.ylabel(r'Weighted Signal Counts', size=9)
plt.legend(handles=[mpatches.Patch(color='green', label='Signal')])
ax2 = plt.subplot(gs[2,:])
plt.bar((x_bins[:-1] + x_bins[1:]) / 2., n_sig / np.sqrt(n_bkg), width=x_bins[1] - x_bins[0], color='k')
plt.ylabel(r'Significance ($S/\sqrt{B})$', size=9)
plt.xlabel('Probability Event is a Signal')

plt.tight_layout()
plt.savefig('../plots/preds_train.pdf')
plt.close(fig)


fpr, tpr, thresholds = metrics.roc_curve(df.is_sig.values, df.preds.values, pos_label=1)
fig = plt.figure(figsize=(6,6))
plt.plot(fpr, tpr)
plt.title('XGBClassifier ROC')
plt.annotate('Area: ' + str(round(metrics.auc(fpr, tpr), 2)), xy=(.8,.2), xycoords='axes fraction')
plt.xlim((0,1))
plt.ylim((0,1))
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.savefig('../plots/roc_curve.pdf')
plt.close(fig)



pred_variable = 'deep_pred'

figures = []
for var in df.dataset.columns:
    colors = ['r', 'b']
    title  = {0: 'background', 1: 'signal'}
    
    f, axes = plt.subplots(1, 2, sharey=True)
    for sig in [0, 1]:        
        x = df[df['is_sig'] == sig][var]
        y = df[df['is_sig'] == sig][pred_variable]
        weight = df[df['is_sig'] == sig]['weight']
        
        h = axes[sig].hist2d(x, y, weights=weight, bins=20, norm=LogNorm())
        axes[sig].set_ylabel('classifier output')
        axes[sig].set_xlabel(var)
        axes[sig].set_xlim([-3, 20])
        axes[sig].set_title(title[sig] + ', corr: {:.2f}'.format(corr(x, y, weight)))
    #f.colorbar(h[3]) 
    # plt.show()
    figures.append(f)

import matplotlib.backends.backend_pdf
pdf = matplotlib.backends.backend_pdf.PdfPages("2D_plots.pdf")
for fig in figures:
    pdf.savefig( fig )
pdf.close()








