import sys
sys.path.append("/scratch/arka/miniconda3/external/fastai/")
import matplotlib
matplotlib.use('Agg')

from tqdm import tqdm
tqdm.monitor_interval = 0

from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *
from fastai.models.senet import *

class new_senet(nn.Module):
    def __init__(self, md, nout):
        super().__init__()
        self.md = md
        self.lin2 = nn.Linear(1000, 512)
        self.lin3 = nn.Linear(512, nout)
        self.d1 = nn.Dropout(0.5)
    def forward(self, inp):
        out1 = F.relu(self.md(inp))
        out2 = self.d1(F.relu(self.lin2(out1)))
        out3 = self.lin3(out2)
        return F.log_softmax(out3, dim=-1)

PATH = "/scratch/arka/Ark_git_files/iFood/"
sz = 224
arch = inception_4
bs = 32


torch.cuda.is_available()

torch.backends.cudnn.enabled

train_label_csv = f'{PATH}train3_info.csv'
train2_label_csv = f'{PATH}train4_info.csv'
valid_label_csv = f'{PATH}val_info.csv'
test_label_csv = f'{PATH}test_info.csv'

train_label_df = pd.read_csv(train_label_csv, header=None)
valid_label_df = pd.read_csv(valid_label_csv, header=None)
test_label_df = pd.read_csv(test_label_csv, header=None)
train2_label_df = pd.read_csv(train2_label_csv, header=None)

train2_label_df.pivot_table(index=1, aggfunc=len).sort_values(0, ascending=True)
# len(train_label_df)

val_idxs = np.arange(len(train_label_df), len(train2_label_df))


def get_data(sz, bs):
    tfms = tfms_from_model(arch, sz, aug_tfms=[RandomCrop(sz), RandomFlip()], max_zoom=1.1)
    data = ImageClassifierData.from_csv(PATH, 'train_all', f'{PATH}train4_info.csv',
                                        test_name='test_set', val_idxs=val_idxs,
                                        tfms=tfms, bs=bs, skip_header=False)
    return data

def accuracytop3(preds, targs):
    preds_3 = preds.sort(dim=1, descending=True)[1][:, :3]
    return ((preds_3[:, 0] == targs) + (preds_3[:, 1] == targs) + (preds_3[:, 2] == targs)).float().mean()

data = get_data(sz, bs)
md = se_resnext101_32x4d()
md3 = new_senet(md,211)
learn = ConvLearner.from_model_data(md3, data)
learn.unfreeze()

learn.fit(1e-2,1,cycle_len=30, best_save_name='se_resnext101_512h',metrics=[accuracy, accuracytop3])


