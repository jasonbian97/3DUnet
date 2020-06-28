from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.utils.data.sampler import WeightedRandomSampler
from scipy.ndimage import zoom
from scipy.ndimage import rotate

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pdb
import json
import time
import math

import pdb

import random


def f2s(f):
    #This function converts a file name to a subjects name
    return f.split('_')[0] 

class subvolume_dataset(Dataset):

    def __init__(self, data_root, subset_size=None, ignore_subjects=None, use_subjects=None):

        files = os.listdir(data_root)
        self.files = files

        all_subjects = list(set([f2s(f) for f in files]))

        if use_subjects is not None:
            self.files = [f for f in self.files if f2s(f) in use_subjects]

        
        if ignore_subjects is not None:
            self.files = [f for f in self.files if f2s(f) not in ignore_subjects]

        
        if subset_size is not None:
            subset_subjects = np.random.choice( all_subjects , subset_size, replace=False) # ???
            self.files = [f for f in files if f2s(f) in subset_subjects]
        
        self.data_root = data_root
        
        self.subjects = list( set( [ f2s(f) for f in self.files] ) )

    def update_subjects( self, new_subjects ):
        self.subjects = new_subjects
        self.files = [f for f in self.files if f2s(f) in self.subjects]

    def __len__(self):
        return len( self.files )
        
    def __getitem__(self, idx):
        f = self.files[idx]
        filename = os.path.join( self.data_root, f)
        subjects = f

        np_data = np.load( filename )

        image, segmentation = np_data['ct'], np_data['segmentation']

        image = image.astype('float32')
        segmentation = segmentation.astype('float32')

        image = np.expand_dims(image,0)
        segmentation = np.expand_dims(segmentation,0)
        
        rz = np.random.uniform(low=-15.0, high=15.0)
        rd = random.randint(0,2)

        if rd==0:
            image = rotate(image, rz, axes=(0,1), reshape=False, order=1, mode='nearest', cval=0.0, prefilter=True)
            segmentation = rotate(segmentation, rz, axes=(0,1), reshape=False, order=0, mode='nearest', cval=0.0, prefilter=True)
        elif rd==1:
            image = rotate(image, rz, axes=(1,2), reshape=False, order=1, mode='nearest', cval=0.0, prefilter=True)
            segmentation = rotate(segmentation, rz, axes=(1,2), reshape=False, order=0, mode='nearest', cval=0.0, prefilter=True)
        elif rd==2:
            image = rotate(image, rz, axes=(0,2), reshape=False, order=1, mode='nearest', cval=0.0, prefilter=True)
            segmentation = rotate(segmentation, rz, axes=(0,2), reshape=False, order=0, mode='nearest', cval=0.0, prefilter=True)

        
        
        return image, segmentation, subjects

