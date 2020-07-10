import sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
if os.path.dirname(__file__)!="":
    os.chdir(os.path.dirname(__file__)) # set current .py file as working directory
from torch.optim.lr_scheduler import MultiStepLR
from easydict import EasyDict as edict
import torch.nn.functional as F
import pdb
import numpy.ma as ma
import subprocess
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateLogger
from pytorch_lightning import Trainer

from unet import *
from subvolume_dataset import *
from dice_cost_functions import *
from figure_generation import make_figure
import gc
import config.settings

class Unet3D(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()

        # load default settings
        with open('training_parameters.json') as fp:
            default_params = edict(json.load(fp))
        # update
        if not isinstance(hparams,dict) :
            hparams = edict(vars(hparams))
        default_params.update(hparams)
        self.hparams = default_params

        torch.manual_seed(self.hparams.manual_seed)
        torch.backends.cudnn.enabled = False

        # getting model
        if hparams.unet_type == "3-1-3":
            self.cnn = unet3(self.hparams.n_channels, self.hparams.n_classes, drop_rate=self.hparams.dropRate)
        elif hparams.unet_type == "4-1-4":
            self.cnn = unet3_4L(self.hparams.n_channels, self.hparams.n_classes, drop_rate=self.hparams.dropRate)
        else:
            raise ValueError("wrong unet-type")
        # print(self.cnn)
        self.label_weights = self.hparams.initial_label_weights

        with open(self.hparams.test_train_split_fpath) as fp:
            self.train_val_split = json.load(fp)

        self.current_best_mDICE = -1

    def forward(self, x):
        x = self.cnn(x)
        return x

    def configure_optimizers(self):
        optimizer = optim.SGD(self.cnn.parameters(), lr=self.hparams.initial_lr, momentum=self.hparams.momentum)
        scheduler = MultiStepLR(optimizer, milestones=self.hparams.milestones, gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def train_dataloader(self):
        training_dataset = subvolume_dataset(data_root=self.hparams.data_root, use_subjects=self.train_val_split['training'])
        train_loader = DataLoader(training_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers)
        print('Training dataset size')
        print(len(training_dataset))
        print('====================')
        return train_loader


    def val_dataloader(self):
        testing_dataset = subvolume_dataset(data_root=self.hparams.data_root, use_subjects=self.train_val_split['testing'])
        test_loader = DataLoader(testing_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
        print('Testing dataset size')
        print(len(testing_dataset))
        print('====================')
        return test_loader

    def training_step(self, batch, batch_idx):
        input_data, truth, subjects = batch

        predicted = self.forward(input_data)
        loss, dices = dice_coef_multilabel(truth, predicted, class_labels=self.hparams.class_labels,
                                           smooth=0.01, weights=self.label_weights)

        # add logging
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs, "dices":dices}

    def training_epoch_end(self,outputs):
        # print(outputs)
        # print("call training_epoch_end")
        dices_list = [x['dices'] for x in outputs]
        # print("\n ","dices_list",dices_list)
        # update class weights according to last epoch's performance
        self.label_weights = self.__get_weights__(dices_list, self.hparams.class_labels)
        # add logging: compute training stats
        dices_mat = np.array(dices_list)
        self.logger.experiment.add_text("training/dice_each_class",
                                        str(dices_mat.mean(axis=0))+" || mDICE: {}".format(dices_mat.mean()),
                                        self.current_epoch)
        logs = {'training_mDICE': torch.tensor(dices_mat.mean())}
        return {"log":logs}

    def on_epoch_start(self):

        if self.hparams.verbose:
            print( 'Learning rate: %s' % str( self.trainer.lr_schedulers[0]["scheduler"].get_last_lr() )) #???
            print( 'Weights: %s' % (str(self.label_weights)) )

    def validation_step(self, batch, batch_idx):
        input_data, truth, subjects = batch
        predicted = self.forward(input_data)

        cur_loss, dices = dice_coef_multilabel(truth, predicted, class_labels=self.hparams.class_labels,
                                               smooth=0.01, weights=None)
        return {"dices": dices,"loss":cur_loss}

    def validation_epoch_end(self, outputs):
        dices_list = [x['dices'] for x in outputs]
        weighted_loss = torch.stack([x['loss'] for x in outputs]).mean()
        # add logging: compute val stats
        dices_mat = np.array(dices_list)
        dice_each_class = np.mean(dices_mat,axis=0)
        self.logger.experiment.add_text("val/dice_each_class",
                                        str(dice_each_class) + " || mDICE: {}".format(dice_each_class.mean()),
                                        self.current_epoch)
        logs = {'val_mDICE': torch.tensor(dices_mat.mean()),
                "weighted_loss": weighted_loss}

        if self.hparams.save_additional_checkpoint not in ["", None]:
            if logs["val_mDICE"].numpy() > self.current_best_mDICE:
                self.current_best_mDICE = logs["val_mDICE"].numpy() # update
                now = datetime.now() # get time
                dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
                tarfname = "{}-{}-mDICE-{}.tar.gz".\
                    format(self.hparams.ID, dt_string, np.round(logs["val_mDICE"].numpy(),decimals = 3))
                subprocess.run('tar -czf {} {}'.format(tarfname, self.hparams.cur_ckpt_loc).split())
                subprocess.run(["cp", "-r", tarfname, self.hparams.save_additional_checkpoint])
                subprocess.run(["rm", tarfname])
                print("Save training checkpoint {} to {}".
                      format(tarfname, self.hparams.save_additional_checkpoint))
            else:
                print("Not best metric. Did not save the additional checkpoint.")

        return {"log": logs}

    def __get_weights__(self, dice_list, class_labels):

        weights = len(class_labels) * [0.0]
        total = float(len(dice_list))
        for dices in dice_list:
            for i, d in enumerate(dices):
                weights[i] += d

        weights = [w / total for w in weights]
        weights = [1.0 - w for w in weights]
        sum_weights = 0.0
        for w in weights:
            sum_weights += w

        weights = [round(w / sum_weights, 3) for w in weights]

        return weights

if __name__ == '__main__':
    # settings
    hparams = config.settings.parse_opts()

    Sys = Unet3D(hparams=hparams)

    checkpoint_callback = ModelCheckpoint(
        filepath=None,
        monitor='val_mDICE',
        save_top_k=1,
        verbose=True,
        mode='max'
    )

    lr_logger = LearningRateLogger()

    if hparams.debug:
        limit_train_batches = 5
        limit_val_batches = 2
    else:
        limit_train_batches = 1.0
        limit_val_batches = 1.0

    if hparams.resume_path not in ["",None]:
        print("Resume Training Process from {}".format(hparams.resume_path))
        resume_path = hparams.resume_path
    else:
        resume_path = None

    trainer = Trainer(resume_from_checkpoint=resume_path,
                      checkpoint_callback=checkpoint_callback,
                      callbacks=[lr_logger],
                      gpus=hparams.gpu_id,
                      default_root_dir='results/{}'.format(os.path.basename(__file__)[:-3]),
                      max_epochs = hparams.num_epochs,
                      check_val_every_n_epoch=2,
                      limit_train_batches=limit_train_batches,
                      limit_val_batches=limit_val_batches
                      )

    trainer.fit(Sys)

