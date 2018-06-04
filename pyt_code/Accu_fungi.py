
# coding: utf-8

# In[13]:


import sys
sys.path.append("/home/paperspace/fastai/")
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

PATH = "/home/paperspace/iFood/ifungi_data/fungi_data/"
sz = 299
arch = resnext101_64
# arch = inception_4
# arch = resnet50
# arch = resnext50
bs = 32

torch.cuda.is_available()

torch.backends.cudnn.enabled

# files = os.listdir(f'{PATH}train_all/')
# files.sort()
# files[-5:-1]

# train_label_csv = f'{PATH}train3_info.csv'
# train2_label_csv = f'{PATH}train4_info.csv'
# valid_label_csv = f'{PATH}val_info.csv'
# test_label_csv = f'{PATH}test_info.csv'

# train_label_df = pd.read_csv(train_label_csv, header=None)
# valid_label_df = pd.read_csv(valid_label_csv, header=None)
# test_label_df = pd.read_csv(test_label_csv, header=None)
# train2_label_df = pd.read_csv(train2_label_csv, header=None)

# train2_label_df.pivot_table(index=1, aggfunc=len).sort_values(0, ascending=True) # 
# len(train_label_df)

# val_idxs = np.arange(len(train_label_df), len(train2_label_df))
test_json_fname = f'{PATH}test.json'
val_json_fname = f'{PATH}val.json'

def get_data(sz, bs):
    # tfms = tfms_from_model(arch, sz, aug_tfms=transforms_top_down, max_zoom=1.1) # -
    tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
    data = ImageClassifierData.from_paths(PATH, tfms=tfms, bs=bs, test_name='test')
    # data = ImageClassifierData.from_csv(PATH, 'train_all', f'{PATH}train4_info.csv',
    #                                     test_name='test_set', val_idxs=val_idxs,
    #                                     tfms=tfms, bs=bs, skip_header=False)
    return data


data = get_data(sz, bs)

learn = ConvLearner.pretrained(arch, data, precompute=False, xtra_fc=[512])

# lrf = learn.lr_find()

# learn.sched.plot_lr()

# learn.sched.plot()


# def one_to_one_map(inp_label):
#     import json
#     big_label_dict = json.load(open(f'{PATH}big_label_dict.json', 'r'))
#     return big_label_dict[inp_label]

# def accuracytop3(preds, targs):
# #     pdb.set_trace()
#     preds1 = torch.Tensor(preds)
#     targs1 = torch.LongTensor(targs)
#     preds_3 = preds1.sort(dim=1, descending=True)[1][:, :3]
#     return ((preds_3[:, 0] == targs1) + (preds_3[:, 1] == targs1) +
#             (preds_3[:, 2] == targs1)).float().mean()

def accuracytop3(preds, targs):
    preds_3 = preds.sort(dim=1, descending=True)[1][:, :3]
    return ((preds_3[:, 0] == targs) + (preds_3[:, 1] == targs) +
            (preds_3[:, 2] == targs)).float().mean()

def accuracytop3_np(preds, targs):
    preds_3 = np.argsort(preds, axis=1)[:, -3:]
    return ((preds_3[:, 0] == targs) + (preds_3[:, 1] == targs) +
            (preds_3[:, 2] == targs)).mean()

def test_acc_csv():
    test_d = json.load(open(test_json_fname, 'r'))
    test_l = test_d['images']
    log_preds, y = learn.TTA(is_test=True)
    probs = np.exp(log_preds)
    probs_1 = np.mean(probs, axis=0)
    preds = np.argsort(-probs_1, axis=1)[:, :3]
    sub_ds = pd.DataFrame(preds)
    sub_ds.insert(0, 'id', [o['id'] for o in test_l])
    sub_ds_ids = sub_ds['id']
    sub = 'id,predicted\n'
    str_format = '{}, {} {} {}\n'
    for i in range(len(sub_ds)):
        pred_test_fin = preds[i, :]
        sub += str_format.format(sub_ds_ids[i],
                                 pred_test_fin[0], pred_test_fin[1], pred_test_fin[2])
    with open('sub_fungi1.csv', 'w') as f:
        f.write(sub)
        
def val_acc_csv():
    val_d = json.load(open(val_json_fname, 'r'))
    val_l = val_d['images']
    log_preds, y = learn.TTA(is_test=False)
    probs = np.exp(log_preds)
    probs_1 = np.mean(probs, axis=0)
    preds = np.argsort(-probs_1, axis=1)[:, :3]
    sub_ds = pd.DataFrame(preds)
    sub_ds.insert(0, 'id', [o['id'] for o in val_l])
    sub_ds_ids = sub_ds['id']
    sub = 'id,predicted\n'
    str_format = '{}, {} {} {}\n'
    for i in range(len(sub_ds)):
        pred_test_fin = preds[i, :]
        sub += str_format.format(sub_ds_ids[i],
                                 pred_test_fin[0], pred_test_fin[1], pred_test_fin[2])
    with open('sub_fungi1_val.csv', 'w') as f:
        f.write(sub)


# def accuracytop3(preds, targs):
#     preds_3 = preds.sort(dim=1, descending=True)[1][:, :3]
#     return ((preds_3[:, 0] == targs) + (preds_3[:, 1] == targs) +
#             (preds_3[:, 2] == targs)).float().mean()

# def test_acc_csv():
#     log_preds, y = learn.TTA(is_test=True)
#     probs = np.exp(log_preds)

#     probs_1 = np.mean(probs, axis=0)
#     # probs_1.shape

#     sub_ds = pd.DataFrame(probs_1)
#     sub_ds.columns = data.classes
#     sub_ds.insert(0, 'id', [o[9:] for o in data.test_ds.fnames])

#     SUBM = f'{PATH}sub/'
#     os.makedirs(SUBM, exist_ok=True)
#     sub = 'id,predicted\n'
#     str_format = '{},{} {} {}\n'
#     ds_class_array = np.array(data.classes)
#     for i in range(len(sub_ds)):
#         r1 = sub_ds.iloc[i]
#         r11 = np.array(r1[1:])
#         r12 = -r11
#         r2 = np.argsort(r12)
#         r3 = np.array(data.classes)[r2]
#         pred_test_fin = r3[:3]
#         sub += str_format.format(data.test_ds.fnames[i][9:],
#                                  pred_test_fin[0], pred_test_fin[1], pred_test_fin[2])

#     with open('sub_fin1.csv', 'w') as f:
#         f.write(sub)


# In[14]:


learn.load('best_rxnet_fu1')


# In[15]:


learn.freeze()
learn.precompute = False


# In[16]:


learn.save_fc1()


# In[17]:


learn.precompute=True


# In[ ]:


learn.fit(0, 1, cycle_len=1, metrics=[accuracy, accuracytop3])


# In[18]:


val_acc_csv()


# In[6]:


test_acc_csv()


# In[ ]:


test_d = json.load(open(test_json_fname, 'r'))


# In[ ]:


test_d['images']


# In[ ]:


log_preds, y = learn.TTA()
probs = np.exp(log_preds)


# In[ ]:


len(probs_1)


# In[ ]:


probs_1 = np.mean(probs, axis=0)


# In[ ]:


probs_1


# In[ ]:


preds = np.argsort(-probs_1, axis=1)


# In[ ]:


preds1 = preds[:, :3]


# In[ ]:


import json


# In[ ]:


get_ipython().system("ls '{PATH}'")


# In[ ]:


valid_fname = "/home/paperspace/iFood/ifungi_data/val.json"


# In[ ]:


val_d = json.load(open(valid_fname, 'r'))


# In[ ]:


val_l = val_d['images']


# In[ ]:


val_d['annotations'][0]


# In[ ]:


val_l[0]


# In[ ]:


accuracytop3_np(probs_1, y)


# In[ ]:


ds = pd.DataFrame(probs_1)


# In[ ]:


ds.columns = data.classes


# In[ ]:


d


# In[ ]:


data.val_ds.fnames


# In[ ]:


ds.insert(0, 'id', [o['id'] for o in val_l])


# In[ ]:


ds


# In[ ]:


ds.columns[0]


# In[ ]:


cat_dict = {}
for v in val_d['categories']:
    cat_dict[v['name']] = v['id']


# In[ ]:


cat_dict['Achroomyces disciformis']


# In[ ]:


sorted(cat_dict.keys())


# In[ ]:


val_d = json.load(open(valid_fname, 'r'))
log_preds, y = learn.TTA()
probs = np.exp(log_preds)
probs_1 = np.mean(probs, axis=0)
preds = np.argsort(-probs_1, axis=1)[:, :3]
sub_ds = pd.DataFrame(preds)
sub_ds.insert(0, 'id', [o['id'] for o in val_l])
sub_ds_ids = sub_ds['id']
sub = 'id,predicted\n'
str_format = '{}, {} {} {}\n'
for i in range(len(sub_ds)):
    pred_test_fin = preds[i, :]
    sub += str_format.format(sub_ds_ids[i],
                             pred_test_fin[0], pred_test_fin[1], pred_test_fin[2])


# In[ ]:


sub


# In[ ]:


ds['id']


# In[ ]:


probs.mean(axis=0)

