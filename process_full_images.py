import numpy as np
import numpy.ma as ma
import json
import os
import nibabel
import random
from scipy.ndimage import zoom as spzoom
from subvolume_dataset import *
from unet import *
import nibabel
import torch


def numpy_dice_coef(true_label, predicted_label, smooth=0.01):
    intersection = np.sum(true_label * predicted_label)
    union = np.sum(true_label) + np.sum(predicted_label)
    return (2. * intersection + smooth) / ( union + smooth)


in_root = 'C:/dl_aorta_segmentation/data/raw_nifti'
out_dir = 'C:/dl_aorta_segmentation/aorta_segmentation/full_image_test'

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

# Load parameters
with open('training_parameters_original.json') as fp:
    training_parameters = json.load(fp)
model_file_name = training_parameters['model_file_name']
n_channels = training_parameters['n_channels']
n_classes = training_parameters['n_classes']
dropRate = training_parameters['dropRate']
W = training_parameters['patch_width']

cnn = unet3(n_channels, n_classes, drop_rate=dropRate)
cnn.cuda(0)

# Load trained UNet model
cnn.load(model_file_name)

# Get testing data
with open('test_train_split.json') as fp:
    test_split = json.load(fp)

testing_dataset = subvolume_dataset(data_root=in_root, use_subjects=test_split['testing'])
files = testing_dataset.files
files = [f for f in files if not 'VOI' in f]

# Store all test dice scores
all_dices = []
dice_stats = {'min': {'name': '', 'dice': 1}, 'max': {'name': '', 'dice': 0}}

# Store filenames of scans that were too small after zooming
small_files = []

# Get list of all files in out_dir, to prevent repeats
repeat = True
already_outputted = os.listdir(out_dir)
no_repeat_files = files if repeat else [f for f in files if (f.replace('.nii.gz', '_zoomed_in.nii.gz')) not in already_outputted]

# JSON dictionary
all_output = {}
json_fname = 'full_image_test_summary.json'

for f in no_repeat_files:
    print(f)

    c1 = os.path.join(in_root,f)
    c2 = os.path.join(in_root,f.replace('.nii.gz', '_VOI.nii.gz'))

    ct = nibabel.load(c1)
    seg = nibabel.load(c2)

    T = ct.get_sform()

    ct = ct.get_data()
    seg = seg.get_data()

    res = [1.5, 1.5, 1.5]
    zoom_factor = []
    for k in range(3):
        zoom_factor.append(abs(T[k, k]) / res[k])

    zoom_factor = tuple(zoom_factor)

    ct = np.round( spzoom(  ct.astype(float), zoom_factor, order=1 ) ).astype('int16')
    seg = np.round( spzoom( seg.astype(float), zoom_factor, order=0 ) ).astype('uint8')

    print ('Size after zoom: ', ct.shape)

    if ct.shape[0] < W or ct.shape[1] < W or ct.shape[2] < W:
        print ("Patch size too big. Skipping ...")
        small_files.append(f)
        continue

    '''
    if ct.shape[0] < W:
        p = W-ct.shape[0]
        ct = np.pad(ct,( (p,0), (0,0), (0,0) ),  mode='minimum')
        seg = np.pad(seg,( (p,0), (0,0), (0,0) ), mode='minimum')

    if ct.shape[1]<W:
        print ("Patch size too big; need to pad shape[1]!!!")
        p = W-ct.shape[1]
        ct = np.pad(ct,( (0,0), (p,0), (0,0) ),  mode='minimum')
        seg = np.pad(seg,( (0,0), (p,0), (0,0) ), mode='minimum')

    if ct.shape[2]<W:
        print ("Patch size too big; need to pad shape[2]!!!")
        p = W-ct.shape[2]
        ct = np.pad(ct,( (0,0), (0,0), (p,0) ),  mode='minimum')
        seg = np.pad(seg,( (0,0), (0,0), (p,0) ), mode='minimum')
    '''

    xmax = ct.shape[0] - W
    ymax = ct.shape[1] - W
    zmax = ct.shape[2] - W

    estimated_segmentation = np.zeros(ct.shape)

    stride = W // 8
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

    for z in z_list:
        for y in y_list:
            for x in x_list:
                ct2 = ct[x:(x+W), y:(y+W), z:(z+W)]

                ct2 = np.expand_dims(ct2,0)
                ct2 = np.expand_dims(ct2,0)
                ct2 = torch.from_numpy(ct2).float().to(0)

                predicted = cnn(ct2)
                # Perform majority vote by setting non-aorta labels as -1
                # and aorta labels as 1, and then summing the results.
                # If the final result for a voxel is positive, the label should be aorta.
                # Else, the label should be non-aorta.
                predicted = np.squeeze(predicted.cpu().detach().numpy())
                predicted_mask = (np.argmax(predicted[:,:,:,:], axis=0)).astype(int)
                predicted_mask -= ((predicted_mask == 0).astype(int))

                estimated_segmentation[x:(x+W), y:(y+W), z:(z+W)] += predicted_mask

    estimated_segmentation = (estimated_segmentation > 0).astype(int)

    cur_dice = (0.5 * numpy_dice_coef((seg == 0).astype(float), (estimated_segmentation == 0).astype(float), smooth=0.01)) + \
               (0.5 * numpy_dice_coef((seg == 1).astype(float), (estimated_segmentation == 1).astype(float), smooth=0.01))
    print ("Dice Score = ", cur_dice)

    # Update dice score stats
    if cur_dice < dice_stats['min']['dice']:
        dice_stats['min']['name'] = f
        dice_stats['min']['dice'] = cur_dice
    if cur_dice > dice_stats['max']['dice']:
        dice_stats['max']['name'] = f
        dice_stats['max']['dice'] = cur_dice

    all_dices.append(cur_dice)

    # Zoom back to original resolution
    res = [1.5, 1.5, 1.5]
    zoom_factor = []
    for k in range(3):
        zoom_factor.append(res[k] / abs(T[k, k]))

    zoom_factor = tuple(zoom_factor)

    estimated_segmentation = np.round(spzoom( estimated_segmentation.astype(float), zoom_factor, order=0)).astype('uint8')
    ct_unzoom = np.round(spzoom(ct.astype(float), zoom_factor, order=1)).astype('int16')

    # Save estimated segmentation as a zipped NIFTI file
    orig_seg = nibabel.load(c2)
    save_seg_fname = os.path.join(out_dir, f.replace('.nii.gz', '_VOI_estimated.nii.gz'))
    estimated_seg = nibabel.Nifti1Image(estimated_segmentation, orig_seg.affine, orig_seg.header)
    nibabel.save(estimated_seg, save_seg_fname)

    # Save zoomed back (undid the zoom) ct as a zipped NIFTI file
    orig_ct = nibabel.load(c1)
    save_ct_fname = os.path.join(out_dir, f.replace('.nii.gz', '_zoomed_in.nii.gz'))
    ct_unzoom = nibabel.Nifti1Image(ct_unzoom, orig_ct.affine, orig_ct.header)
    nibabel.save(ct_unzoom, save_ct_fname)

print (100 * '=')
if all_dices:
    dice_mean = np.array(all_dices).mean()
    print('Testing DICE: %.3f' % dice_mean)
    print (100 * '=')
    print('Min Dice: ', dice_stats['min'])
    print('Max Dice: ', dice_stats['max'])
    all_output['Average Test Dice'] = dice_mean
    all_output['Min Test Dice'] = dice_stats['min']
    all_output['Max Test Dice'] = dice_stats['max']
else:
    no_dice_msg = 'No dice scores computed: Either all estimates are already in the output directory or all files have been skipped.'
    print (no_dice_msg)
    all_output['Average Test Dice'] = no_dice_msg
print (100 * '=')
print('Skipped files: ', small_files)
all_output['Skipped Files'] = small_files

with open(os.path.join(out_dir, json_fname), 'w') as fp:
    fp.write(json.dumps(all_output, indent=4))
