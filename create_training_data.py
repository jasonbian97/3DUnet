import numpy as np
import os
import nibabel
import random
import json
from scipy.ndimage import zoom as spzoom
import shutil
import os
os.chdir(os.path.dirname(__file__)) # set current .py file as working directory
# img_in_root = 'data/raw/NiiCT'
img_in_root = "data/raw/NiiBiMask"
seg_in_root = "data/raw/NiiAnnReg"
out_root = 'data/cache/train_test_data_npy_BiMask_sp1'

if os.path.exists(out_root):
    shutil.rmtree(out_root)
    os.makedirs(out_root)
else:
    os.makedirs(out_root)

with open('training_parameters.json') as fp:
    training_parameters = json.load(fp)
W = training_parameters['patch_width']

files = os.listdir(img_in_root)
# files = [f for f in files if not 'VOI' in f]

for f in files:
    print(f)
    base_name = f.split("_")[0] + "_" + f.split("_")[1]
    c1 = os.path.join(img_in_root,f)
    c2 = os.path.join(seg_in_root, base_name + "_VOI_labelled.nii.gz")

    ct = nibabel.load(c1)
    seg = nibabel.load(c2)

    # T = ct.get_sform()
    T = ct.affine

    ct = ct.get_data()
    seg = seg.get_data()

    # res = [1.5, 1.5, 1.5]
    res = [1.0, 1.0, 1.0]
    zoom_factor = []
    for k in range(3):
        zoom_factor.append(abs(T[k, k]) / res[k]) # ???

    zoom_factor = tuple(zoom_factor)

    ct = np.round( spzoom(  ct.astype(float), zoom_factor, order=1 ) ).astype('int16')
    seg = np.round( spzoom( seg.astype(float), zoom_factor, order=0 ) ).astype('uint8')

    if ct.shape[0]<W:
        p = W-ct.shape[0]
        ct = np.pad(ct,( (p,0), (0,0), (0,0) ),  mode='minimum')
        seg = np.pad(seg,( (p,0), (0,0), (0,0) ), mode='minimum')

    if ct.shape[1]<W:
        p = W-ct.shape[1]
        ct = np.pad(ct,( (0,0), (p,0), (0,0) ),  mode='minimum')
        seg = np.pad(seg,( (0,0), (p,0), (0,0) ), mode='minimum')

    if ct.shape[2]<W:
        p = W-ct.shape[2]
        ct = np.pad(ct,( (0,0), (0,0), (p,0) ),  mode='minimum')
        seg = np.pad(seg,( (0,0), (0,0), (p,0) ), mode='minimum')

    xmax = ct.shape[0] - W 
    ymax = ct.shape[1] - W 
    zmax = ct.shape[2] - W
    for i in range(50):

        o = os.path.join(out_root,f.replace('.nii.gz', '_%d.npz' % i))

        x = random.randint(0,xmax)
        y = random.randint(0,ymax)
        z = random.randint(0,zmax)


        ct2 = ct[x:(x+W), y:(y+W), z:(z+W)]
        seg2 = seg[x:(x+W), y:(y+W), z:(z+W)]
        if ct2.shape != (W,W,W):
            print('Incorrect shape')
            exit()

        if seg2.shape != (W,W,W):
            print('Incorrect shape')
            exit()
        np.savez_compressed(o, ct=ct2, segmentation=seg2)
