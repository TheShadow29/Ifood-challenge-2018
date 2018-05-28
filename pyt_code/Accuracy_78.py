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

PATH = "../data/"
sz = 299
arch = resnext101_64
# arch = resnet50
# arch = resnext50
bs = 64

torch.cuda.is_available()

torch.backends.cudnn.enabled

files = os.listdir(f'{PATH}train_all/')
files.sort()
files[-5:-1]

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
    tfms = tfms_from_model(arch, sz, aug_tfms=transforms_top_down, max_zoom=1.1)
    data = ImageClassifierData.from_csv(PATH, 'train_all_compressed', f'{PATH}train4_info.csv',
                                        test_name='test_set', val_idxs=val_idxs,
                                        tfms=tfms, bs=bs, skip_header=False)
    return data


data = get_data(sz, bs)

learn = ConvLearner.pretrained(arch, data, precompute=False, opt_fn=optim.Adam)

# lrf = learn.lr_find()

# learn.sched.plot_lr()

# learn.sched.plot()


def one_to_one_map(inp_label):
    import json
    big_label_dict = json.load(open(f'{PATH}big_label_dict.json', 'r'))
    return big_label_dict[inp_label]


def accuracytop3(preds, targs):
    preds_3 = preds.sort(dim=1, descending=True)[1][:, :3]
    return ((preds_3[:, 0] == targs) + (preds_3[:, 1] == targs) +
            (preds_3[:, 2] == targs)).float().mean()


# learn.unfreeze()
# learn.load('acc_rxnet_79')
# learn.load('best_rxnet_1')
# learn.fit(0.005, 3, cycle_len=2, metrics=[accuracy, accuracytop3])

# learn.fit(0.001, 2, cycle_len=2, cycle_mult=2, metrics=[accuracy, accuracytop3])

# learn.fit(0.001, 1, cycle_len=4, cycle_mult=2, metrics=[accuracy, accuracytop3])

# learn.fit(0.001, 1, cycle_len=4, cycle_mult=2, metrics=[accuracy, accuracytop3])

# log_preds, y = learn.TTA(is_test=True)
# probs = np.exp(log_preds)

# probs_1 = np.mean(probs, axis=0)
# probs_1.shape

# sub_ds = pd.DataFrame(probs_1)
# sub_ds.columns = data.classes
# sub_ds.insert(0, 'id', [o[9:] for o in data.test_ds.fnames])


def test_acc_csv():
    log_preds, y = learn.TTA(is_test=True)
    probs = np.exp(log_preds)

    probs_1 = np.mean(probs, axis=0)
    # probs_1.shape

    sub_ds = pd.DataFrame(probs_1)
    sub_ds.columns = data.classes
    sub_ds.insert(0, 'id', [o[9:] for o in data.test_ds.fnames])

    SUBM = f'{PATH}sub/'
    os.makedirs(SUBM, exist_ok=True)
    sub = 'id,predicted\n'
    str_format = '{},{} {} {}\n'
    ds_class_array = np.array(data.classes)
    for i in range(len(sub_ds)):
        r1 = sub_ds.iloc[i]
        r11 = np.array(r1[1:])
        r12 = -r11
        r2 = np.argsort(r12)
        r3 = np.array(data.classes)[r2]
        pred_test_fin = r3[:3]
        sub += str_format.format(data.test_ds.fnames[i][9:],
                                 pred_test_fin[0], pred_test_fin[1], pred_test_fin[2])

    with open('sub_fin1.csv', 'w') as f:
        f.write(sub)


# def val_conf_mtx():
# log_preds, y = learn.TTA()
# probs = np.exp(log_preds)
# probs_1 = np.mean(probs, axis=0)
# ds_class_array = np.array(data.classes)
# pred_validations = np.zeros((probs_1.shape[0], 3))
# cm = np.zeros((211, 211))
# actual_classes = valid_label_df[1]
# for i in range(probs_1.shape[0]):
#     actual_class = actual_classes[i]
#     r11 = probs_1[i, :]
#     r12 = -r11
#     r2 = np.argsort(r12)
#     r3 = ds_class_array[r2]
#     pred_val_fin = np.array(r3[:3], dtype=np.int_)
#     pred_validations[i, :] = pred_val_fin
#     if np.any(pred_val_fin == actual_class):
#         cm[actual_class, actual_class] += 1
#     else:
#         cm[actual_class, pred_val_fin[0]] += 1

# return cm


# In[77]:


# from fastai.conv_learner import *


# # In[78]:


# ## 19
# learn.unfreeze()
# learn.bn_freeze(True)

# lr = np.array([0.00001, 0.0001, 0.001])
# # In[79]:


# ## 20
# lr = np.array([0.00001, 0.0001, 0.001])


# # In[83]:


# learn.set_data(get_data(224,bs//2))


# # In[95]:


# ## 21
# learn.fit(lr, 1, cycle_len=1, metrics=[accuracy, accuracytop3])


# # In[ ]:


# learn.fit(lr, 1, cycle_len=3, metrics=[accuracy, accuracytop3])


# # In[ ]:


# ## 22
# learn.fit(lr, 1, cycle_len=3, metrics=[accuracy, accuracytop3])


# # In[ ]:


# ## 23
# learn.save('resNext101_64_RanUnfreezeNicely')


# # In[ ]:


# ## 24
# learn.freeze()


# # In[ ]:


# ## 25
# learn.precompute=True


# # In[ ]:


# ## 26
# def accuracytop5(preds, targs):
#     preds_5 = preds.sort(dim=1, descending=True)[1][:, :5]
#     return ((preds_5[:, 0] == targs) + (preds_5[:, 1] == targs) + (preds_5[:, 2] == targs) + (preds_5[:, 3] == targs) + (preds_5[:, 4] == targs)).float().mean()


# # In[ ]:


# ## 27
# learn.fit(1e-4, 1, cycle_len=1, metrics=[accuracy, accuracytop3, accuracytop5])


# # In[ ]:


# ## 28 -- STOP
# log_preds,y = learn.TTA()
# probs = np.mean(np.exp(log_preds),0)
# b = torch.from_numpy(probs)
# c = torch.from_numpy(y)
# accuracytop5(b, c)


# # In[ ]:


# learn.unfreeze()
# learn.bn_freeze(True)


# # In[ ]:


# ## DONT :P Will Regret
# lr = np.array([0.0005, 0.005, 0.05])


# # In[ ]:


# learn.fit(lr, 1, cycle_len=3, metrics=[accuracy, accuracytop3, accuracytop5])


# # In[ ]:


# learn.load('resNext101_64_bs64_sz224_all_Freezed')


# # In[ ]:


# learn.freeze()


# # In[ ]:


# learn.precompute=True


# # In[ ]:


# learn.fit(0, 1, cycle_len=1, metrics=[accuracy, accuracytop3])


# # In[ ]:


# lrf=learn.lr_find()


# # In[ ]:


# learn.sched.plot()


# # In[ ]:


# learn.unfreeze()


# # In[ ]:


# lrf=learn.lr_find()


# # In[ ]:


# learn.fit(0, 1, cycle_len=1, metrics=[accuracy, accuracytop3])


# # In[ ]:


# learn.fit(0.05, 1, cycle_len=2, metrics=[accuracytop3])


# # In[ ]:


# learn.fit(0.001, 1, cycle_len=2, metrics=[accuracy, accuracytop3])


# # In[ ]:


# get_ipython().system('ls ../data/tmp/')


# # In[ ]:


# learn.load('resNext101_64_bs64_sz224_all')


# # In[ ]:


# learn.freeze()


# # In[ ]:


# learn.save('resNext101_64_bs64_sz224_all_Freezed')


# # In[ ]:


# learn.fit(0.001, 1, cycle_len=2, metrics=[accuracy, accuracytop3])


# # In[ ]:


# learn.unfreeze()


# # In[ ]:


# lr=np.array([0.0005, 0.005, 0.05])


# # In[ ]:


# learn.fit(lr, 2, cycle_len=1)


# # In[ ]:


# learn.save('resNext101_64_bs64_sz224_all')


# # In[ ]:


# get_ipython().system('ls ../data/models/')
