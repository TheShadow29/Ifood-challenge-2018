
# coding: utf-8

# In[1]:


# Put these at the top of every notebook, to get automatic reloading and inline plotting
## 1
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


## 2
import sys
sys.path.append("/home/SharedData/anaconda2/external/fastai/")


# In[3]:


# !pip install torchtext


# In[3]:


from tqdm import tqdm
tqdm.monitor_interval = 0


# In[4]:


# This file contains all the main external libs we'll use
## 3
from fastai.imports import *


# In[5]:


## 4
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *


# In[6]:


## 5
PATH = "../data/"
sz=224
arch = resnext101_64
bs=64


# In[7]:


## 6
torch.cuda.is_available()


# In[8]:


## 7
torch.backends.cudnn.enabled


# In[10]:


os.listdir(PATH)


# In[11]:


f'{PATH}train_set1'


# In[12]:


files = os.listdir(f'{PATH}train_set1/')
files.sort()
files[-5:-1]


# In[12]:


img = plt.imread(f'{PATH}train_set1/{files[4]}')
plt.imshow(img);


# In[13]:


img.shape


# In[14]:


img[:4,:4]


# In[15]:


get_ipython().system('ls ../data')


# In[9]:


## 8
train_label_csv = f'{PATH}train_info.csv'
train2_label_csv = f'{PATH}train2_info.csv'
valid_label_csv = f'{PATH}val_info.csv'
test_label_csv = f'{PATH}test_info.csv'


train_label_df = pd.read_csv(train_label_csv, header=None)
valid_label_df = pd.read_csv(valid_label_csv, header=None)
test_label_df = pd.read_csv(test_label_csv, header=None)
train2_label_df = pd.read_csv(train2_label_csv, header=None)


# In[17]:


print(train2_label_df.tail())
print("Now Valid")
print(valid_label_df.tail())


# In[14]:


train_label_df.pivot_table(index=1, aggfunc=len).sort_values(0, ascending=True)


# In[15]:


valid_label_df.pivot_table(index=1, aggfunc=len).sort_values(0, ascending=True)


# In[16]:


train2_label_df.pivot_table(index=1, aggfunc=len).sort_values(0, ascending=True)


# In[21]:


len(train_label_df)


# In[10]:


val_idxs = np.arange(len(train_label_df), len(train2_label_df))
val_idxs


# In[23]:


get_ipython().system("wc -l '{PATH}train2_info.csv'")


# In[24]:


print(len(val_idxs))
print(len(valid_label_df))


# In[11]:


## 9
val_idxs = np.arange(len(train_label_df), len(train2_label_df))
def get_data(sz, bs):
    tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
    data = ImageClassifierData.from_csv(PATH, 'train_set1', f'{PATH}train2_info.csv', test_name='test_set', val_idxs=val_idxs, tfms=tfms, bs=bs, skip_header=False)
    return data 

data = get_data(sz,bs)


# In[12]:



fn = PATH + data.val_ds.fnames[-1]; fn


# In[27]:


img = PIL.Image.open(fn); img


# In[28]:


img.size


# In[29]:


len(data.trn_ds), len(data.test_ds)


# In[30]:


len(data.classes), data.classes[:7]


# In[ ]:


# !pip install torch==0.3.1


# In[13]:


## 10
learn = ConvLearner.pretrained(arch,data, precompute=True, opt_fn=optim.Adam)


# In[32]:


## 11
lrf=learn.lr_find()


# In[34]:


learn.sched.plot_lr()


# In[35]:


## 12
learn.sched.plot()


# In[14]:


## 13
def accuracytop3(preds, targs):
    preds_3 = preds.sort(dim=1, descending=True)[1][:, :3]
    return ((preds_3[:, 0] == targs) + (preds_3[:, 1] == targs) + (preds_3[:, 2] == targs)).float().mean()
    


# In[15]:


## 14
learn.fit(0.005, 3, cycle_len=2, metrics=[accuracy, accuracytop3])


# In[16]:


## 15
learn.fit(0.001, 2, cycle_len=2, cycle_mult=2, metrics=[accuracy, accuracytop3])


# In[40]:


lrf=learn.lr_find()


# In[41]:


learn.sched.plot()


# In[17]:


## 16_1
learn.fit(0.001, 1, cycle_len=4, cycle_mult=2, metrics=[accuracy, accuracytop3])


# In[18]:


## 16_2
learn.fit(0.001, 1, cycle_len=4, cycle_mult=2, metrics=[accuracy, accuracytop3])


# In[19]:


# 16_3 Testing1
log_preds, y = learn.TTA(is_test=True)
probs = np.exp(log_preds)


# In[22]:


#16_3_2 Testing2
probs_1 = np.mean(probs, axis=0)
probs_1.shape


# In[25]:


#16_3_3 Testing3
sub_ds = pd.DataFrame(probs_1)
sub_ds.columns = data.classes
sub_ds.insert(0, 'id', [o[9:] for o in data.test_ds.fnames])
sub_ds.head()


# In[64]:


sub_ds.iloc[0]


# In[77]:


img_test = PIL.Image.open(f'{PATH}test_set/test_007350.jpg')
plt.imshow(img_test)


# In[67]:


r1 = sub_ds.iloc[1]
r11 = np.array(r1[1:])
r12 = -r11
r2 = np.argsort(r12)
r3 = np.array(data.classes)[r2]
r3
r3[:3]
r1[r3[:3]]


# In[75]:


# 16_3_4 Testing4
SUBM = f'{PATH}sub/' 
os.makedirs(SUBM, exist_ok=True)
sub = 'id, predicted\n'
str_format = '{},{} {} {}\n'
ds_class_array = np.array(data.classes)
for i in range(len(sub_ds)):
    r1 = sub_ds.iloc[i]
    r11 = np.array(r1[1:])
    r12 = -r11
    r2 = np.argsort(r12)
    r3 = np.array(data.classes)[r2]
    pred_test_fin = r3[:3]
    sub += str_format.format(data.test_ds.fnames[i][9:], pred_test_fin[0], pred_test_fin[1], pred_test_fin[2])
# with open('sub1.csv', 'w') as f:
#     f.write(sub)


# In[76]:


# 16_3_5 Testing5
with open('sub_fin1.csv', 'w') as f:
    f.write(sub)


# In[ ]:


## Optional. Doesnt Help Much
learn.set_data(get_data(299,bs))


# In[ ]:


learn.fit(0.001, 1, cycle_len=4, cycle_mult=2, metrics=[accuracy, accuracytop3])


# In[44]:


## 17
import torch


# In[45]:


pred_test = learn.predict(is_test=True)


# In[ ]:


learn


# In[108]:


log_preds = learn.predict(is_test=True)
probs = np.exp(log_preds)  


# In[109]:


probs.shape


# In[103]:


data.test_ds.fnames[0][9:]


# In[111]:


probs.shape


# In[1]:


ds = pd.DataFrame(probs)
ds.columns = data.classes
ds.insert(0, 'id', [o[9:] for o in data.test_ds.fnames])


# In[114]:


ds.head()


# In[115]:


SUBM = f'{PATH}sub/' 
os.makedirs(SUBM, exist_ok=True) 
ds.to_csv(f'{SUBM}subm.csv', index=False)


# In[97]:


pred_test


# In[48]:


len(test_label_df)


# In[61]:


pred_test_3 = (-pred_test).argsort(axis=1)
pred_test_3 = pred_test_3[:, :3]


# In[72]:


test_label_df[0][0]


# In[99]:


fn = PATH + data.test_ds.fnames[0]; fn


# In[120]:


sub = 'id, predicted\n'
str_format = '{},{} {} {}\n'
for i in range(len(ds)):
    row = ds.
    sub += str_format.format(data.test_ds.fnames[i][9:], pred_test_3[i, 0], pred_test_3[i, 1], pred_test_3[i, 2])
with open('sub1.csv', 'w') as f:
    f.write(sub)


# In[96]:


## 18
log_preds,y = learn.TTA()
probs = np.mean(np.exp(log_preds),0)
b = torch.from_numpy(probs)
c = torch.from_numpy(y)
accuracytop3(b, c)


# In[ ]:


## Dont Run
def unfreeze_new(self, num_from_last):
    """ Unfreeze all layers.

    Make all layers trainable by unfreezing. This will also set the `precompute` to `False` since we can
    no longer pre-calculate the activation of frozen layers.

    Returns:
        None
    """
    self.freeze_to(-1*num_from_last)
    self.precompute = False


# In[77]:


from fastai.conv_learner import *


# In[78]:


## 19
learn.unfreeze()
learn.bn_freeze(True) 

lr = np.array([0.00001, 0.0001, 0.001])
# In[79]:


## 20
lr = np.array([0.00001, 0.0001, 0.001])


# In[83]:


learn.set_data(get_data(224,bs//2))


# In[95]:


## 21
learn.fit(lr, 1, cycle_len=1, metrics=[accuracy, accuracytop3])


# In[ ]:


learn.fit(lr, 1, cycle_len=3, metrics=[accuracy, accuracytop3])


# In[ ]:


## 22
learn.fit(lr, 1, cycle_len=3, metrics=[accuracy, accuracytop3])


# In[ ]:


## 23
learn.save('resNext101_64_RanUnfreezeNicely')


# In[ ]:


## 24
learn.freeze()


# In[ ]:


## 25
learn.precompute=True


# In[ ]:


## 26
def accuracytop5(preds, targs):
    preds_5 = preds.sort(dim=1, descending=True)[1][:, :5]
    return ((preds_5[:, 0] == targs) + (preds_5[:, 1] == targs) + (preds_5[:, 2] == targs) + (preds_5[:, 3] == targs) + (preds_5[:, 4] == targs)).float().mean()


# In[ ]:


## 27
learn.fit(1e-4, 1, cycle_len=1, metrics=[accuracy, accuracytop3, accuracytop5])


# In[ ]:


## 28 -- STOP
log_preds,y = learn.TTA()
probs = np.mean(np.exp(log_preds),0)
b = torch.from_numpy(probs)
c = torch.from_numpy(y)
accuracytop5(b, c)


# In[ ]:


learn.unfreeze()
learn.bn_freeze(True) 


# In[ ]:


## DONT :P Will Regret
lr = np.array([0.0005, 0.005, 0.05])


# In[ ]:


learn.fit(lr, 1, cycle_len=3, metrics=[accuracy, accuracytop3, accuracytop5])


# In[ ]:


learn.load('resNext101_64_bs64_sz224_all_Freezed')


# In[ ]:


learn.freeze()


# In[ ]:


learn.precompute=True


# In[ ]:


learn.fit(0, 1, cycle_len=1, metrics=[accuracy, accuracytop3])


# In[ ]:


lrf=learn.lr_find()


# In[ ]:


learn.sched.plot()


# In[ ]:


learn.unfreeze()


# In[ ]:


lrf=learn.lr_find()


# In[ ]:


learn.fit(0, 1, cycle_len=1, metrics=[accuracy, accuracytop3])


# In[ ]:


learn.fit(0.05, 1, cycle_len=2, metrics=[accuracytop3])


# In[ ]:


learn.fit(0.001, 1, cycle_len=2, metrics=[accuracy, accuracytop3])


# In[ ]:


get_ipython().system('ls ../data/tmp/')


# In[ ]:


learn.load('resNext101_64_bs64_sz224_all')


# In[ ]:


learn.freeze()


# In[ ]:


learn.save('resNext101_64_bs64_sz224_all_Freezed')


# In[ ]:


learn.fit(0.001, 1, cycle_len=2, metrics=[accuracy, accuracytop3])


# In[ ]:


learn.unfreeze()


# In[ ]:


lr=np.array([0.0005, 0.005, 0.05])


# In[ ]:


learn.fit(lr, 2, cycle_len=1)


# In[ ]:


learn.save('resNext101_64_bs64_sz224_all')


# In[ ]:


get_ipython().system('ls ../data/models/')

