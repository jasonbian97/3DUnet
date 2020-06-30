
import sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from unet import *
from subvolume_dataset import *
from dice_cost_functions import *
import os
os.chdir(os.path.dirname(__file__)) # set current .py file as working directory
from torch.optim.lr_scheduler import MultiStepLR

import torch.nn.functional as F
import pdb
import numpy.ma as ma

from figure_generation import make_figure
import gc

def memReport():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj), obj.size())

def create_figures(cnn, dataset, figure_dir='figures', gpu_idx=0):

    batch_size=1
    test_loader = DataLoader(dataset, batch_size=batch_size)  
    test_loss = 0

    i=0
    with torch.no_grad():
        for images, truth, subjects in test_loader:
            images, truth = Variable( images.cuda(gpu_idx) ), Variable( truth.cuda(gpu_idx) )
            predicted = cnn(images)
            
            _, dices = dice_coef_multilabel( truth, predicted, class_labels=[0,1,2,3,4,5,6], smooth=0.01, weights=None )

            truth = np.squeeze( truth.cpu().detach().numpy() )
            predicted = np.squeeze( predicted.cpu().detach().numpy() )
            images = np.squeeze( images.cpu().detach().numpy() )

            if not os.path.isdir(figure_dir):
                os.mkdir(figure_dir)
            make_figure('%s/test_figure_%d.png' % (figure_dir,i), 
                        images, 
                        predicted, 
                        truth, 
                        figure_title='%s | %s' % (subjects[0], str(dices) )
            )
            i+=1

def test(cnn, dataset, class_labels, weights=None, gpu_idx=0):

    batch_size=8
    test_loader = DataLoader(dataset, batch_size=batch_size)
    all_subjects=[]
    all_dices=[]
    lines=[]
    dice_each_class = []
    with torch.no_grad():
        for input_data, truth, subjects in test_loader:

            input_data, truth = Variable(input_data.cuda(gpu_idx)), Variable(truth.cuda(gpu_idx))

            predicted = cnn(input_data)

            cur_loss, dices = dice_coef_multilabel(truth, predicted, class_labels=class_labels, smooth=0.01, weights=weights)

            all_subjects.append( subjects[0] )
            all_dices.append( cur_loss.item()  )
            dice_each_class.append(dices)

    # print('Testing DICE(weigted loss): %.3f' % -np.array(all_dices).mean())
    print('Dice_each_class = ', np.array(dice_each_class).mean(axis=0),
          " || ",
          "mDice = ", np.array(dice_each_class).mean())
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

def train_model(save_model=True, test_train_split_fpath='test_train_split.json', verbose=True, make_figures=True):

    with open('training_parameters.json') as fp:
       training_parameters = json.load(fp)

    n_channels=training_parameters['n_channels']
    n_classes=training_parameters['n_classes']
    class_labels=training_parameters['class_labels']

    gpu_idx=0
    reuse_model = training_parameters['reuse_model']
    #############################

    #Training params
    initial_lr = training_parameters['initial_lr']
    milestones = training_parameters['milestones']
    num_epochs = training_parameters['num_epochs']
    batch_size = training_parameters['batch_size']
    dropRate = training_parameters['dropRate']
    momentum = training_parameters['momentum']
    data_root = training_parameters['data_root']

    print('Loading parameters...')
    #############################
    #############
    #  Load CNN
    cnn = unet3(n_channels, n_classes, drop_rate=dropRate)
    cnn.cuda(gpu_idx)

    with open(test_train_split_fpath) as fp:
        test_split = json.load(fp)

    #Optimizer
    torch.backends.cudnn.enabled=False
    gpu_idx = 0

    training_dataset  = subvolume_dataset(data_root=data_root, use_subjects=test_split['training'] )
    testing_dataset  = subvolume_dataset(data_root=data_root, use_subjects=test_split['testing'] )

    print('Training dataset size')
    print(len(training_dataset))
    print('====================')
    print('Testing dataset size')
    print(len(testing_dataset))
    print('====================')

    figure_dataset  = subvolume_dataset(data_root=data_root, subset_size=training_parameters['figure_dataset_size'] )
    figure_dataset.files = np.random.choice(figure_dataset.files, training_parameters['figure_dataset_size'], replace=False )

    current_dice=0.0
    max_dice = training_parameters['max_dice']
    not_improving=False

    train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

    j=0
    #I had to give 85% weight to the right middle lobe for the first few epochs or it had trouble converging
    weights = training_parameters['initial_label_weights']

    #current_dice = test(cnn, testing_dataset, class_labels, weights)
    create_figures(cnn, figure_dataset)
    optimizer = optim.SGD(cnn.parameters(), lr=initial_lr, momentum=momentum)
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    # Store final training and testing dice scores
    final_training_dice = 0.0
    best_testing_dice = 0.0

    print('Starting training...')
    for epoch in range(num_epochs):
        start_time = time.time()
        if verbose:
            print( 'Epoch: %d' % epoch )
            print( 'Learning rate: %s' % str( scheduler.get_lr() ))
            print( 'Weights: %s' % (str(weights)) )
        dices_list = []
        all_dices =[]
        for i, data in enumerate(train_loader):
            # get the inputs

            input_data, truth, subjects = data
            #truth = torch.squeeze(truth)

            #Send the images to the gpu and convert them to Variables
            input_data, truth = Variable(input_data.cuda(gpu_idx)), Variable(truth.cuda(gpu_idx))

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            predicted = cnn(input_data)

            loss, dices = dice_coef_multilabel(truth, predicted, class_labels=class_labels, smooth=0.01, weights=weights)
            dices_list.append( dices )
            all_dices.append( loss.cpu().item() )
            #Do back-propagation of the gradients.
            loss.backward()
            optimizer.step()
            #del loss

            weights = get_weights(dices_list, class_labels) ###??? too frequently??
            if i%20 == 0:
                percent_complete = 100*(float(i*batch_size)/float(len(training_dataset)))
                percent_complete='%.2f'%percent_complete+'%'
                print(percent_complete,flush=True,end='')
            # print(percent_complete + len(percent_complete)*'\b')
            #i+=1 # Jeffrey: I do not think this is needed
        end_time = time.time()
        if verbose:
            print('Epoch took %.3f sec' % (end_time-start_time) )

        scheduler.step()

        if verbose:
            print('')
            print('Training Dice: %.3f' % -np.array(all_dices).mean())
        final_training_dice = -np.array(all_dices).mean()

        previous_dice = current_dice

        cnn.eval()
        if make_figures:
            print('Creating figures...')
            create_figures(cnn, figure_dataset)

        current_dice = test(cnn, testing_dataset, class_labels, weights=weights)
        cnn.train()

        if current_dice >= max_dice:
            max_dice = current_dice
            best_testing_dice = max_dice
            not_improving = False
        else:
            not_improving = True

        if save_model and not not_improving:
            cnn.save( training_parameters['model_file_name'] )

        if verbose:
            print( 'Dice score(val weighted loss): %.6f\t|\t Best DICE(val weighted loss): %.6f' % (current_dice, max_dice) )
            print(200*'=')
            print(200*'=')

    return final_training_dice, best_testing_dice

if __name__ == "__main__":
    final_training_dice, best_testing_dice = train_model(make_figures=False)
    print ("Finished Training")
    print(200*'=')
    print ("Final Training Dice: %.6f\t|\t Best Test DICE: %.6f" % (final_training_dice, best_testing_dice))
    print(200*'=')
