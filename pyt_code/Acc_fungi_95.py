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
import json

PATH = "/home/paperspace/iFood/ifungi_data/fungi_data/"
sz = 299
arch = resnext101_64
# arch = inception_4
# arch = resnet50
# arch = resnext50
bs = 32

torch.cuda.is_available()

torch.backends.cudnn.enabled

test_json_fname = f'{PATH}test.json'
val_json_fname = f'{PATH}val.json'

#test_json_fname = f'{PATH}test.json'

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

def save_test():
    test_d = json.load(open(test_json_fname, 'r'))
    test_l = test_d['images']
    log_preds, y = learn.TTA(is_test=True)
    probs = np.exp(log_preds)
    probs_1 = np.mean(probs, axis=0)
    # preds = np.argsort(-probs_1, axis=1)[:, :3]
    # sub_ds = pd.DataFrame(preds)
    sub_ds = pd.DataFrame(probs_1)
    sub_ds.insert(0, 'id', [o['id'] for o in test_l])
    sub_ds.to_pickle('ifungi_test.pkl')
    return

def get_test_img_names():
    te_img_dict = {}
    test_d = json.load(open(test_json_fname, 'r'))
    test_l = test_d['images']
    for t in test_l:
        te_img_dict[t['file_name']] = t['id']
    test_fname_list = []
    for t1 in data.test_ds.fnames:
        test_fname_list.append(te_img_dict[t1])

    return test_fname_list

def test_acc_csv():
    # test_d = json.load(open(test_json_fname, 'r'))
    # test_l = test_d['images']
    test_img_names = get_test_img_names()
    log_preds, y = learn.TTA(is_test=True)
    probs = np.exp(log_preds)
    probs_1 = np.mean(probs, axis=0)
    preds = np.argsort(-probs_1, axis=1)[:, :3]
    sub_ds = pd.DataFrame(preds)
    sub_ds.insert(0, 'id', [o for o in test_img_names])
    sub_ds_ids = sub_ds['id']
    sub = 'id,predicted\n'
    str_format = '{},{} {} {}\n'
    for i in range(len(sub_ds)):
        pred_test_fin = preds[i, :]
        sub += str_format.format(sub_ds_ids[i],
                                 pred_test_fin[0], pred_test_fin[1], pred_test_fin[2])
    with open('sub_fungi1.csv', 'w') as f:
        f.write(sub)


# def accuracytop3(preds, targs):
#     preds_3 = preds.sort(dim=1, descending=True)[1][:, :3]
#     return ((preds_3[:, 0] == targs) + (preds_3[:, 1] == targs) +
#             (preds_3[:, 2] == targs)).float().mean()

# def test_acc_csv():
#     test_d = json.load(open(
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
