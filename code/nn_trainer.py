import torch
import visdom
from tqdm import tqdm
import os
import shutil


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class BaseTrainer:
    def __init__(self, config, nn_model, optimizer, other_params):
        # other_params is a list of parameters
        # this is to avoid adding arguments here
        # Can use visdom for visualization
        self.config = config
        self.nn_model = nn_model
        self.optimizer = optimizer
        self.other_params = other_params
        return

    def load_model(self, load_path='model_best.pth.tar'):
        if os.path.isfile(load_path):
            print("=> loading checkpoint '{}'".format(load_path))
            checkpoint = torch.load(load_path)
            self.start_epoch = checkpoint['epoch']
            # best_prec1 = checkpoint['best_prec1']
            self.nn_model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(load_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(load_path))

    def train_model(self, num_epochs=10):
        is_best = False
        best_acc = 0
        for i in tqdm(range(num_epochs)):
            self.train_single_step()

        curr_acc = self.test_model()
        is_best = curr_acc > best_acc
        if is_best:
            best_acc = curr_acc
            is_best = True
            save_checkpoint({
                'epoch': self.curr_epoch + 1,
                # 'arch': args.arch,
                'state_dict': self.nn_model.state_dict(),
                # 'best_prec1': best_prec1,
                'optimizer': self.optimizer.state_dict(),
            }, is_best)

        return

    def train_single_step(self):
        raise NotImplementedError

    def test_model(self):
        raise NotImplementedError
