import numpy as np 
import math
import os

from random import shuffle
from PIL import Image

import config as cfg

input_scale = cfg.INPUT_SCALE
class_num = cfg.CLASS_NUM

def read_img(path,target_size):
	try:
		# base = '/home/jianayang/data/PASCAL3D+_release1.1/Images/chair_imagenet'
		# path = '%s/%s.JPEG'%(base,path)
		img = Image.open(path).convert("RGB")
		img_rs = img.resize(target_size)
		width = img.width
		height = img.height
	except Exception as e:
		print(e)
		return False
	else:
		x = np.expand_dims(np.array(img_rs), axis=0)/255.0
		return x,width,height


# input file in the foramt of: img_path eta p xa,wa,ya,ha
def gen_all_inputs(d,training_sample,img_base_dir,num=1,random=False):
	img = []
	p = []
	eta = []
	bbox = []

	with open(training_sample) as f:
		ts = f.read().splitlines()

	if random:
		shuffle(ts)

	count = 0
	for sample in ts:
		if(count==num):
			break
		item = d[sample]
		print(item)
		img_path = os.path.join(img_base_dir,sample)
		print(img_path)
		# _img,w,h = read_img(img_path,(512,512))

		res = read_img(img_path,(input_scale,input_scale))
		if res is not False:
			_img,w,h = res
		
			img.append(_img)
			p.append(item['dl'][:2])
			eta.append(item['bbox'][0][:2])
			bbox.append(item['bbox'][1][:2])
			count += 1		
		else:
			continue

	img = np.concatenate(img)
	p = np.array(p)
	eta = np.array(eta)
	bbox = np.array(bbox)

	return img,eta,p,bbox

def gen_batch_inputs(d,training_sample,img_base_dir,batch_size,random=False):
	items = d.items()
	img =[]
	p = []
	eta = []
	bbox = []

	with open(training_sample) as f:
		ts = f.read().splitlines()

	if random:
		shuffle(ts)

	while True:
		for k,sample in enumerate(ts):
			item = d[sample]
			img_path = os.path.join(img_base_dir,sample)
			if not os.path.isfile(img_path):
				continue
			# _img,w,h = read_img(img_path,(512,512))
			res = read_img(img_path,(input_scale,input_scale))
			if res is not False:
				_img,w,h = res
				img.append(_img)
				p.append(item['dl'][:2])
				eta.append(item['bbox'][0][:2])
				bbox.append(item['bbox'][1][:2])
			else:
				continue		

			if len(img)==batch_size:
				img = np.concatenate(img)
				p = np.array(p)
				eta = np.array(eta)
				bbox = np.array(bbox)
				yield [img,eta,p,bbox],np.ones(img.shape[0])  # [x1,x2,x3,x4],y
				
				img,eta,p,bbox = [],[],[],[]


def gen_dummy_batch_inputs(batch_size,img_size):

	while True:
		img,eta,p,bbox = [],[],[],[]
		for i in range(batch_size):
			r = np.random.randint(2)
			if r==1:
				_img = np.ones((1,img_size,img_size,3))*10
				_p = [0,1]
				_eta = [0,0]
				_bbox = [[-1]*4 for _ in range(class_num)]
			else:
				_img = np.ones((1,img_size,img_size,3))*5
				_p = [1,0]
				_eta = [1,0]
				_bbox = [[0.5,0.1,0.5,0.1]]+[[-1]*4 for _ in range(class_num-1)]

			img.append(_img)
			p.append(_p)
			eta.append(_eta)
			bbox.append(_bbox)


		img = np.concatenate(img)
		p = np.array(p)
		eta = np.array(eta)
		bbox = np.array(bbox)
		yield [img,eta,p,bbox],np.ones(img.shape[0])  # [x0,x1,x2,x3],y


if __name__ == '__main__':
	training_sample = cfg.TRAIN_SAMPLE_LIST
	from generate_label import *
	d = gen(cfg.DATA_ENTRY_PATH,cfg.BBOX_PATH)
	# img_base_dir = '/data/users/jianayang/Thoracic_Disease/images'
	img_base_dir = cfg.IMAGE_BASE_PATH
	# img_base_dir = '/data/cxr8/images'
	# img,p,eta,bbox = gen_all_inputs(d,training_sample,img_base_dir,random=False)

	# for a in gen_batch_inputs(d, training_sample, img_base_dir, 1):
	for a in gen_dummy_batch_inputs(4,32):
		img,eta,p,bbox = a[0]
		break
	print(img.shape)
	print(p.shape)
	print(eta.shape)
	print(bbox.shape)

	# print(img)
	print(p)
	print(eta)
	print(bbox)




