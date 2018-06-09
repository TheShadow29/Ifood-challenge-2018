import sys
sys.path.append("/home/SharedData/anaconda2/external/fastai/")

from tqdm import tqdm
tqdm.monitor_interval = 0

from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

PATH = "../"
sz = 299
arch = inception_4
bs = 48

torch.cuda.is_available()

torch.backends.cudnn.enabled

label_csv = '../train.csv'
n = len(list(open(label_csv)))-1
label_df = pd.read_csv(label_csv)

x = []
y = []
for i in range(n):
    x.append(i)
    y.append(label_df.iloc[i]['class'])

x = np.array(x)
y = np.array(y)

def get_data(sz, bs,val_idxs):
    tfms = tfms_from_model(arch, sz, aug_tfms=[RandomCrop(sz), RandomFlip()], max_zoom=1.2)
    data = ImageClassifierData.from_csv(PATH,'train_full',label_csv,val_idxs=val_idxs,suffix='.JPG', tfms=tfms,bs=bs)
    return data

def accuracytop3(preds, targs):
    preds_3 = preds.sort(dim=1, descending=True)[1][:, :3]
    return ((preds_3[:, 0] == targs) + (preds_3[:, 1] == targs) + (preds_3[:, 2] == targs)).float().mean()

skf = StratifiedKFold(n_splits=5)
skf.get_n_splits(x, y)
fold_no = 1

for train_idxs,val_idxs in skf.split(x,y):
    print("Fold No = " + str(fold_no))
    print("Val len = " + str(len(val_idxs)))
    data = get_data(sz, bs, val_idxs)
    learn = ConvLearner.pretrained(arch, data, precompute=False,xtra_fc=[512])

    #learn.load('inception_ifungi_1_sc')

    learn.unfreeze()

    learn.fit([1e-2,1e-2,1e-2], 1, wds=1e-4, cycle_len=30, best_save_name='inception_ifungi_512h_fold'+str(fold_no), metrics=[accuracy, accuracytop3])

    fold_no+=1


