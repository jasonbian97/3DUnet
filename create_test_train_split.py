import json
from subvolume_dataset import *


with open('training_parameters.json') as fp:
	training_parameters = json.load(fp)
data_root = training_parameters['data_root']
training_dataset  = subvolume_dataset(data_root=data_root, 
									  subset_size=training_parameters['training_dataset_size'] )

testing_dataset  = subvolume_dataset(data_root=data_root, 
									 ignore_subjects=training_dataset.subjects, 
									 subset_size=training_parameters['testing_dataset_size'] )

test_split = {}
test_split['training'] = training_dataset.subjects
test_split['testing'] = testing_dataset.subjects

with open('test_train_split.json','w') as fp:
	fp.write( json.dumps(test_split,indent=4) )