import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pdb

def make_figure(figure_file, image, predicted, ground_truth, figure_title=''):

	cols = 2
	rows = 3
	N = image.shape[0]
	plt.figure( figsize=(8*cols,8*rows) )

	midpoint = 32
	
	for j,r in enumerate( [-16,0,16] ):
		cur_slice = midpoint+r
		img2d = np.squeeze(image[:,:,cur_slice]).transpose()
		gt = np.squeeze(ground_truth[:,:,cur_slice]).transpose()
		predicted_flat = np.argmax(predicted[:,:,:,cur_slice],axis=0 ).transpose()


		plt.subplot(rows,cols,1+cols*j)
		plt.imshow(img2d, cmap='gray',vmin=-1000,vmax=1000)
		gt=ma.masked_where(gt==0, gt)
		plt.imshow(gt, cmap='jet',vmin=0,vmax=6, alpha=0.2)
		plt.gca().get_xaxis().set_ticks([])
		plt.gca().get_yaxis().set_ticks([])
		plt.gca().invert_yaxis()

		
		plt.subplot(rows,cols,2+2*j)
		plt.imshow(img2d, cmap='gray',vmin=-1000,vmax=1000)
		predicted_flat=ma.masked_where(predicted_flat==0, predicted_flat)
		plt.imshow(predicted_flat, cmap='jet',vmin=0,vmax=6, alpha=0.2)
		plt.gca().get_xaxis().set_ticks([])
		plt.gca().get_yaxis().set_ticks([])
		plt.gca().invert_yaxis()
		plt.xlabel(figure_title)
		

		'''
		for k in range(2):
			plt.subplot(rows,cols,(k+2)+cols*j)
			img = predicted[k,:,:,cur_slice].transpose()
			plt.imshow(img, cmap='gray',vmin=0,vmax=1)
			plt.gca().get_xaxis().set_ticks([])
			plt.gca().get_yaxis().set_ticks([])
			plt.gca().invert_yaxis()

		plt.subplot(rows,cols,4+cols*j)
		plt.imshow(predicted_flat, cmap='gray', vmin=0, vmax=2)
		plt.gca().get_xaxis().set_ticks([])
		plt.gca().get_yaxis().set_ticks([])
		plt.gca().invert_yaxis()
		'''

	plt.savefig(figure_file, dpi=100, bbox_inches='tight', pad_inches = 0)
	plt.close()