import os

import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt

import config as cfg 


def comput_loss4onemap(y_pred,mode=0):
	'''
	Inputs:
		pred: 11x11 array
	'''

	y_pred = y_pred*0.02 + 0.98
	y_pred_tmp = 1 - y_pred
	# y_pred_tmp = np.min(1.0,y_pred_tmp)
	y_pred_tmp = y_pred_tmp*0.02+0.98  # preform normalization
	p_y_x = 1-np.prod(y_pred_tmp)
	print('p_y_x:',p_y_x)

	if mode ==0 :
		return(np.log(1.0-p_y_x))
	if mode ==1:
		return(np.log(p_y_x))


def compute_total_loss(a):
	# a = np.load('testout.npy')
	# a = a[0]

	loss = 0
	for i in range(14):
		current = a[:,:,i]

		mode = 1 if i==1 else 0
		_loss = comput_loss4onemap(current,mode)
		loss += _loss
		print(i,_loss)

	print(loss)

def visualize_feature_map(a,save_path):
	# a = np.load('testout.npy')
	# a = a[0]

	loss = 0
	for i in range(14):
		current = a[:,:,i]
		
		#####################
		# visualize
		#####################
		plt.subplots(1, 1)
		plt.imshow(current, cmap='jet')
		plt.clim(0, 1)
		plt.colorbar(cmap='jet')
		full_path = os.path.join(save_path,'fm%02d'%i)
		plt.savefig(full_path)


if __name__ == '__main__':

	file_path = os.path.join(cfg.DEBUG_PATH,'testout.npy')
	a = np.load(file_path)
	a = a[0]

	compute_total_loss(a)

	save_path = os.path.join(cfg.DEBUG_PATH,'feature')
	if not os.path.isdir(save_path):
		os.path.makedirs(save_path)
	
	visualize_feature_map(a,save_path)

