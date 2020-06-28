import torch
import numpy as np
import pdb

def dice_coef(true_label, predicted_label, smooth=0):

    intersection = (true_label* predicted_label).sum()
    union = (true_label).sum() + (predicted_label).sum()
    return (2. * intersection + smooth) / ( union  + smooth)


def dice_coef_multilabel(true_labels, predicted_labels, class_labels, smooth=0, weights=None):
    dice=0.0
    true_labels=  torch.squeeze( true_labels )

    dice_list = []

    for i,label in enumerate(class_labels):
        predicted_label=  torch.squeeze( predicted_labels[:,i,:,:,:] )
        true_label =  (true_labels==label).float()
        
        cur_dice = dice_coef( true_label, predicted_label, smooth=smooth)
        dice_list.append( cur_dice.cpu().item() )
        if weights is not None:
            dice += weights[i]*cur_dice
        else:
            dice += cur_dice

    return -dice, dice_list
