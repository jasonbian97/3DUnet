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

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateLogger
from pytorch_lightning import Trainer

from unet import *
from subvolume_dataset import *
from dice_cost_functions import *
from figure_generation import make_figure
import gc
import config.settings
from train_pytorchlightning import *
import yaml
import nibabel
from scipy.ndimage import zoom as spzoom

hp_path = "/mnt/ssd2/Projects/AortaSeg/chuckUnet/results/train_pytorchlightning/lightning_logs/version_0/hparams.yaml"
test_img_path = "data/raw/NiiBiMask/102_20120601v2_VOI.nii.gz"
test_img_ann_path = "data/raw/NiiAnnReg/102_20120601v2_VOI_labelled.nii.gz"

with open(hp_path) as file:
    hparams = yaml.load(file,Loader=yaml.FullLoader)

pretrained_model = Unet3D.load_from_checkpoint(
    checkpoint_path="/mnt/ssd2/Projects/AortaSeg/chuckUnet/results/train_pytorchlightning/lightning_logs/version_0/checkpoints/epoch=31.ckpt",
    hparams = hparams
)
pretrained_model.freeze()

ct = nibabel.load(test_img_path)
seg = nibabel.load(test_img_ann_path)

T = ct.affine

ct = ct.get_data()
seg = seg.get_data()

res = [1., 1., 1.]
zoom_factor = []
for k in range(3):
    zoom_factor.append(abs(T[k, k]) / res[k])

zoom_factor = tuple(zoom_factor)

ct = np.round( spzoom(  ct.astype(float), zoom_factor, order=1 ) ).astype('int16')
seg = np.round( spzoom( seg.astype(float), zoom_factor, order=0 ) ).astype('uint8')

W=64

xmax = ct.shape[0] - W
ymax = ct.shape[1] - W
zmax = ct.shape[2] - W

stride = W // 2
# Calculating indices at which to extract patches
z_list = list(range(0, zmax, stride))
z_list = [z for z in z_list if (z + W) < (ct.shape[2] - 1)]
if z_list:
    z_list[-1] = zmax

y_list = list(range(0, ymax, stride))
y_list = [y for y in y_list if (y + W) < (ct.shape[1] - 1)]
if y_list:
    y_list[-1] = ymax

x_list = list(range(0, xmax, stride))
x_list = [x for x in x_list if (x + W) < (ct.shape[0] - 1)]
if x_list:
    x_list[-1] = xmax

num_class = 7
vote_mat = torch.zeros((num_class,*ct.shape))
for z in z_list:
    for y in y_list:
        for x in x_list:
            ct2 = ct[x:(x+W), y:(y+W), z:(z+W)]

            ct2 = np.expand_dims(ct2,0)
            ct2 = np.expand_dims(ct2,0)
            ct2 = torch.from_numpy(ct2).float()
            logits = pretrained_model(ct2)
            logits = torch.squeeze(logits)
            predicted = np.argmax(logits,axis=0)

            for c in range(num_class):
                vote_mat[c,x:(x+W), y:(y+W), z:(z+W)] += (predicted==c).float()

final_pred = np.argmax(vote_mat.numpy(),axis=0)
final_pred = final_pred.astype(np.uint8)
new_img = nibabel.Nifti1Image(final_pred, T)
nibabel.save(new_img, "102_20120601v2_VOI_pred.nii.gz")
print(np.unique(final_pred))



