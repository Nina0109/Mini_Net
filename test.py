import keras
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler,EarlyStopping,ModelCheckpoint
from keras.callbacks import TensorBoard
from keras import backend as K
from keras.utils import generic_utils

import os
import glob
import sys
sys.path.insert(0, os.path.abspath('./'))


import time
import math
import numpy as np
from PIL import Image
import tensorflow as tf
import logging
logging.basicConfig(level=logging.DEBUG)

import matplotlib as mpl
mpl.use('agg')

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from thoracicnet.model import thoracic_model_deploy
from thoracicnet.datainput import read_img
import thoracicnet.config as cfg


disease2id = {'No Finding':0,'Atelectasis':1,'Cardiomegaly':2,
	'Effusion':3,'Infiltrate':4,'Mass':5,'Nodule':6,'Pneumonia':7,'Pneumothorax':8,'Consolidation':9,
	'Edema':10,'Emphysema':11,'Fibrosis':12,'Pleural_Thickening':13,'Hernia':14,'Infiltration':4}

logging.debug('Loading model...')
batch_size = 4
model = thoracic_model_deploy(batch_size)

weight_path = "weights/weights.20-50.05.hdf5"
# weight_path = "weights/weights.20-2.92.hdf5"
model.load_weights(weight_path)

logging.debug('Model loaded.')


def vis_heatmap(heatmaps,img,img_name):
	'''
	Input:
		heatmap: PxPxK array
		img: img in the from o h*w
	'''


	full_path = os.path.join(cfg.IMAGE_BASE_PATH,img_name)
	img_pic = Image.open(full_path)
	w,h = img_pic.size

	for i in range(heatmaps.shape[2]):
		heatmap = heatmaps[:,:,i]
		heatmap = Image.fromarray(heatmap)
		heatmap = heatmap.resize((w,h),Image.NEAREST)
		heatmap = np.array(heatmap,dtype=np.float32)

		fig, ax = plt.subplots(1, 1)
		# ax.imshow(img_pic)
		im = ax.imshow(heatmap, cmap='jet',alpha=0.2)
		plt.colorbar(im,cmap='jet')
		plt.savefig('results/imgs4/'+str(i)+'_'+img_name)

	# h = img.shape[0]
	# w = img.shape[1]

	# for i in range(heatmaps.shape[2]):
	# 	heatmap = heatmaps[:,:,i]
	# 	heatmap = Image.fromarray(heatmap)

	# 	heatmap = heatmap.resize((w,h),Image.NEAREST)
	# 	heatmap = np.array(heatmap,dtype=np.float32)


	# 	mycmap = plt.cm.Reds
	# 	mycmap._init()
	# 	mycmap._lut[:,-1] = np.linspace(0, 0.5, 255+4)

	# 	img_pic = Image.fromarray(img,'RGB')
	# 	fig, ax = plt.subplots(1, 1)
	# 	ax.imshow(img_pic)

	# 	y, x = np.mgrid[0:h, 0:w]
	# 	cb = ax.contourf(x, y, heatmap, 15, alpha=.75, cmap=mycmap)
	# 	plt.colorbar(cb)
	# 	plt.savefig('results/imgs/'+str(i)+'_'+img_name)


def test_for_one_image(img):
	pass

def gen_classification_label(line):
	img_name = ""
	label = ""
	return img_name,label

		
def test(test_file,threshold=0.2):
	suffix = int(time.time()*10%100000)
	test_list = []
	f_gt = open('results/gt.%d.txt'%suffix,'w')
	f_pred = open('results/pred.%d.txt'%suffix,'w')

	###########################
	# generate true label
	###########################
	logging.debug('generating gt files')

	with open(test_file,'r') as f:
		for line in f:
			img_name = line.strip()
			test_list.append(img_name)

	logging.debug('Total %d test imgs.'%len(test_list))

	gt = {}
	with open('data/Data_Entry_2017.csv','r') as f:
		for line in f:
			line  = line.strip().split(',')
			img_name = line[0]
			label = [0]*15
			diseases = line[1].split('|')
			for disease in diseases:
				idx = disease2id[disease]
				label[idx] = 1
			gt[img_name] = label

	logging.debug('gt length: %d'%len(gt))

	#########################
	# write gt file
	#########################
	for img in test_list:
		label = gt[img]
		label = [str(item) for item in label]
		print(img)
		f_gt.write(img+'  '+'  '.join(label)+'\n')

	f_gt.close()
	logging.debug('gt file generation completed.')


	########################
	# generate prediction
	########################
	for img_name in test_list:
		full_path = os.path.join('/workspace/data/',img_name)
		try:
			img,w,h = read_img(full_path, (512,512))
		except:
			continue
		res = model.predict(img)

		vis_heatmap(res[0], img[0], img_name)
		

		# logging.debug(str(res))
		res = 1-res
		logging.debug(res.shape)

		# normalize
		# res = 0.02*res+0.98

		res = np.prod(res,axis=1,keepdims=False) # product over --->PXP
		res = np.prod(res,axis=1,keepdims=False) # product over PXP<---
		res = 1-res
		pred = np.zeros_like(res)
		
		logging.debug('PRED:'+' '.join(['%.3f'%item for item in res.flatten().tolist()]))
		pred[res>threshold] = 1
		print(pred.shape)
		pred = pred.tolist()[0]
		pred = [str(int(item)) for item in pred]
		line = img_name+'  '+'  '.join([str(item) for item in res])
		logging.debug('PRED: '+line)
		f_pred.write(line+'\n')

	
	f_pred.close()

def find_lastest(file_dir='./results'):
	gt_list = glob.glob(file_dir+'/gt*.txt')
	pred_list = glob.glob(file_dir+'/pred*.txt')
	gt_list.sort(key=lambda fn: os.path.getmtime(fn))
	pred_list.sort(key=lambda fn: os.path.getmtime(fn))
	return gt_list[-1],pred_list[-1]

def evalute(gt_file,pred_file):

	gt = {}
	with open(gt_file,'r') as f:
		for line in f:
			line = line.strip().split('  ')
			img_name = line[0]
			label = line[1:]
			gt[img_name] = label

	for disease_id in range(1,15):
		tp = 0
		tn = 0
		fp = 0
		fn = 0
		total = 0

		with open(pred_file,'r') as f:
			for line in f:
				line = line.strip().split('  ')
				img_name = line[0]
				label = line[1+disease_id-1]
				gt_label = gt[img_name][disease_id]


				total += 1
				if gt_label=='1' and label=='1':
					tp += 1
				elif gt_label=='1' and label=='0':
					fn += 1
				elif gt_label=='0' and label=='1':
					fp += 1
				elif gt_label=='0' and label=='0':
					tn += 1

		# print('Diseaseid: %d--->'%disease_id,'TP: %d, FP: %d, TN: %d, FN: %d'%(tp,fp,tn,fn))
		precision = tp*1.0/(tp+fp+1e-8)
		recall = tp*1.0/(tp+fn+1e-8)

		print
		print('-'*30)
		print('Diseaseid: %d--->'%disease_id,'Precision = %.3f'%precision)
		print('Diseaseid: %d--->'%disease_id,'Recall = %.3f'%recall)


def convert(thre=0.5):
	gt_file, pred_file = find_lastest(file_dir='./results')
	fout = open('./results/pred.txt','w')
	with open(pred_file,'r') as f:
		context = f.read().splitlines()
		i = 0
		while i < len(context):
			line = context[i].strip() + ' ' + context[i+1].strip() + ' ' + context[i+2].strip()
			print(line)
			img_name = line[:16]
			tmp = line[19:-1].split(' ')
			print(tmp)
			while('') in tmp:
				tmp.remove('')
			tmp = np.array([float(item) for item in tmp])
			pred = np.zeros_like(tmp)
			pred[tmp>thre] = 1
			print(pred.tolist())
			new_line = img_name + '  '+ '  '.join([str(int(item)) for item in pred.tolist()]) + '\n'
			print(new_line)
			fout.write(new_line)
			i += 3
	fout.close()






if __name__ == '__main__':
	test_file = 'data/test_test.txt'
	test(test_file,threshold=0.65)

	convert(0.75)
	gt_file, pred_file = find_lastest('./results')
	logging.debug(gt_file)
	logging.debug(pred_file)

	evalute(gt_file, pred_file)

	# img_name = "00000013_002.png"
	# img = Image.open(img_name)
	# w,h = img.size
	# img = np.array(img,dtype=np.float32)

	# res = np.random.random((1,h,w,2))
	# res[res<0] = 0
	# vis_heatmap(res[0], img, img_name)


