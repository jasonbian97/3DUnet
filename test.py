
import sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


from UNet3D import *
from SubVolumeDataset import *
from dice_cost_functions import *

from torch.optim.lr_scheduler import MultiStepLR

import torch.nn.functional as F
import pdb
import numpy.ma as ma

from figure_generation import make_figure

import clean

import gc

def memReport():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj), obj.size())

def create_figures(cnn, dataset, gpu_idx=1):

	batch_size=1
	test_loader = DataLoader(dataset, batch_size=batch_size)  
	test_loss = 0

	i=0
	with torch.no_grad():
		for images, truth, subjects in test_loader:
			images, truth = Variable( images.cuda(gpu_idx) ), Variable( truth.cuda(gpu_idx) )
			predicted = cnn(images)
			truth = np.squeeze( truth.cpu().detach().numpy() )
			predicted = np.squeeze( predicted.cpu().detach().numpy() )
			images = np.squeeze( images.cpu().detach().numpy() )

			make_figure('figures/test_figure_%d.png'%i, images, predicted, truth, figure_title=subjects[0])
			i+=1

def test(cnn, dataset, class_labels, weights=None, gpu_idx=1):

	batch_size=1
	test_loader = DataLoader(dataset, batch_size=batch_size)  
	all_subjects=[]
	all_dices=[]

	lines=[]
	with torch.no_grad():
		for input_data, truth, subjects in test_loader:

			input_data, truth = Variable(input_data.cuda(gpu_idx)), Variable(truth.cuda(gpu_idx))

			predicted = cnn(input_data)

			cur_loss, dices = dice_coef_multilabel(truth, predicted, class_labels=class_labels, smooth=0.01, weights=weights)

			all_subjects.append( subjects[0] )
			all_dices.append( cur_loss.item()  )


	print('Testing DICE: %.3f' % -np.array(all_dices).mean())
	return -np.array(all_dices).mean()

def get_weights(dice_list, class_labels):

	weights = len(class_labels)*[0.0]
	total = float(len(dice_list))
	for dices in dice_list:
		for i,d in enumerate(dices):
			weights[i]+=d

	weights = [w/total for w in weights]
	weights = [1.0-w for w in weights]
	sum_weights = 0.0
	for w in weights:
		sum_weights+=w
	
	weights = [round(w/sum_weights,3) for w in weights]

	return weights

if __name__ == "__main__":

	with open('training_parameters.json') as fp:
		training_parameters = json.load(fp)

	n_channels=training_parameters['n_channels']
	n_classes=training_parameters['n_classes']
	class_labels=training_parameters['class_labels']

	gpu_idx=1
	reuse_model = training_parameters['reuse_model']
	#############################

	#Training params
	initial_lr = training_parameters['initial_lr']
	milestones = training_parameters['milestones']
	num_epochs = training_parameters['num_epochs']
	batch_size = training_parameters['batch_size']
	dropRate = training_parameters['dropRate']
	momentum = training_parameters['momentum']
	in_plane_shape = training_parameters['in_plane_shape']
	slice_count = training_parameters['slice_count']

	print('Loading parameters...')
	#############################
	#############
	#  Load CNN
	cnn = unet3(n_channels, n_classes, drop_rate=dropRate)
	cnn.cuda(gpu_idx)
	cnn.load( training_parameters['model_file_name'] )

	cnn.eval()
	with open('test_train_split.json') as fp:
		test_split = json.load(fp)


	#Optimizer
	torch.backends.cudnn.enabled=False

	gpu_idx = 1

	data_root = '/mnt/data/copdgene/sub_volume_data'

	testing_dataset  = SubVolumeDataset(data_root=data_root, use_series=test_split['testing'] )
	testing_dataset.update_series( clean.clean(testing_dataset.series) )
	figure_dataset  = SubVolumeDataset(data_root=data_root, subset_size=100 )
	figure_dataset.update_series( clean.clean(figure_dataset.series) )


	weights = [0.0,0.5,0.5]
	create_figures(cnn, figure_dataset)
	current_dice = test(cnn, testing_dataset, class_labels, weights)
	
	


    	
