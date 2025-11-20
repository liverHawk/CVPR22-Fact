import abc
import torch
import os.path as osp
from dataloader.data_utils import *

from utils import (
    ensure_path,
    Averager, Timer, count_acc,
    WandbLogger,
)


class Trainer(object, metaclass=abc.ABCMeta):
    def __init__(self, args):
        self.args = args
        self.args = set_up_datasets(self.args)
        self.dt, self.ft = Averager(), Averager()
        self.bt, self.ot = Averager(), Averager()
        self.timer = Timer()
        self.global_step = 0
        self.wandb = WandbLogger(self.args)

        # train statistics
        self.trlog = {}
        self.trlog['train_loss'] = []
        self.trlog['val_loss'] = []
        self.trlog['test_loss'] = []
        self.trlog['train_acc'] = []
        self.trlog['val_acc'] = []
        self.trlog['test_acc'] = []
        self.trlog['max_acc_epoch'] = 0
        self.trlog['max_acc'] = [0.0] * args.sessions


    @abc.abstractmethod
    def train(self):
        pass

    def finalize(self):
        if hasattr(self, 'wandb') and self.wandb is not None:
            self.wandb.finish()