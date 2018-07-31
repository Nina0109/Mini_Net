import keras
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler,EarlyStopping,ModelCheckpoint
from keras.callbacks import TensorBoard
from keras import backend as K
from keras.utils import generic_utils

import tensorflow as tf

import os
import time

import math
import numpy as np
from PIL import Image


from thoracicnet.model import thoracic_model
from thoracicnet.datainput import *
from thoracicnet.generate_label import *

import thoracicnet.config as cfg


try:
	from StringIO import StringIO  # Python 2.7
except ImportError:
	from io import BytesIO         # Python 3.x

# import matplotlib.pyplot as plt

def draw_train_curve(hist,img_prefix=''):

	plt.plot(hist['loss'])
	plt.plot(hist['val_loss'])

	plt.title('model train vs validation loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train','validation'],loc='upper right')
	plt.savefig(img_prefix+'_train_val.jpg')



class LogPredG(keras.callbacks.TensorBoard):
	def __init__(self,logdir,data):
		super(LogPredG, self).__init__(logdir)
		self.fun_lastfeaturemap = fun = K.function([model.layers[0].input,
			model.layers[7].input,model.layers[8].input,model.layers[9].input],[model.layers[6].output])
		# self.fun_losslayer = K.function([model.layers[0].input],[model.layers[6].output])
		self.iteration = 0
		self.writer2 = writer
		self.data = data

	def thoracic_loss(self,y_pred,eta,p,bbox,lambda_bbox=0.5):
		# Input shape: a list of 4 tensor
		# y_pred:(1,P,P)
		# eta:(1,1)
		# p:(1,1)
		# bbox:(1,4) = (xa,wa,ya,ha)
		# print(y_pred.shape)
		# print(eta.shape)
		# print(p.shape)
		# print(bbox.shape)
		# print

		lambda_bbox = 0.5

		y_pred = np.minimum(1.0,y_pred)

		y_pred_tmp = 1 - y_pred
		y_pred_tmp = np.minimum(1.0,y_pred_tmp)

		y_pred = y_pred*0.02+0.98   # preform normalization
		y_pred_tmp = y_pred_tmp*0.02+0.98  # preform normalization

		p_y_x = 1-np.prod(y_pred_tmp) 

		P = float(np.shape(y_pred)[1])

		x,y,w,h = bbox[0]*P,bbox[1]*P,bbox[2]*P,bbox[3]*P
		x,y,w,h = int(x),int(y),int(w),int(h)

		in_box = y_pred[y:y+h,x:x+w]

		p1 = np.prod(in_box)
		p3 = np.prod(y_pred_tmp[y:y+h,x:x+w])
		p2 = np.prod(y_pred_tmp)
		p_y_x_bbox = p1*p2/p3 + K.epsilon()

		loss1 = -eta*np.log(p_y_x_bbox)
		loss2 = -(1.0-eta)*p*np.log(p_y_x)
		loss3 = -(1.0-eta)*(1.0-p)*np.log(1.0-p_y_x)

		return loss1,loss2,loss3

	def total_loss(self,x):
		loss1,loss2,loss3 = 0,0,0
		bz = img_vis.shape[0]
		for i in range(bz):
			for j in range(14):
				_loss1,_loss2,_loss3 = self.thoracic_loss(x[0][i,:,:,j],x[1][i,j],x[2][i,j],x[3][i,j])
				loss1 += _loss1
				loss2 += _loss2
				loss3 += _loss3
		return loss1*1.0/bz,loss2*1.0/bz,loss3*1.0/bz

		

	def on_epoch_end(self,epoch,logs=None):
		logs = logs or {}

		if not self.validation_data and self.histogram_freq:
			raise ValueError("If printing histograms, validation_data must be "
                             "provided, and cannot be a generator.")
		if self.embeddings_data is None and self.embeddings_freq:
			raise ValueError("To visualize embeddings, embeddings_data must "
                             "be provided.")
		if self.validation_data and self.histogram_freq:
			if epoch % self.histogram_freq == 0:

				val_data = self.validation_data
				tensors = (self.model.inputs +
					self.model.targets +
					self.model.sample_weights)

				if self.model.uses_learning_phase:
					tensors += [K.learning_phase()]

				assert len(val_data) == len(tensors)
				val_size = val_data[0].shape[0]
				i = 0
				while i < val_size:
					step = min(self.batch_size, val_size - i)
					if self.model.uses_learning_phase:
						# do not slice the learning phase
						batch_val = [x[i:i + step] for x in val_data[:-1]]
						batch_val.append(val_data[-1])
					else:
						batch_val = [x[i:i + step] for x in val_data]
					assert len(batch_val) == len(tensors)
					feed_dict = dict(zip(tensors, batch_val))
					result = self.sess.run([self.merged], feed_dict=feed_dict)
					summary_str = result[0]
					self.writer.add_summary(summary_str, epoch)
					i += self.batch_size

		if self.embeddings_freq and self.embeddings_data is not None:
			if epoch % self.embeddings_freq == 0:
				# We need a second forward-pass here because we're passing
				# the `embeddings_data` explicitly. This design allows to pass
				# arbitrary data as `embeddings_data` and results from the fact
				# that we need to know the size of the `tf.Variable`s which
				# hold the embeddings in `set_model`. At this point, however,
				# the `validation_data` is not yet set.

				# More details in this discussion:
				# https://github.com/keras-team/keras/pull/7766#issuecomment-329195622

				embeddings_data = self.embeddings_data
				n_samples = embeddings_data[0].shape[0]

				i = 0
				while i < n_samples:
					step = min(self.batch_size, n_samples - i)
					batch = slice(i, i + step)

					if type(self.model.input) == list:
						feed_dict = {model_input: embeddings_data[idx][batch]
							for idx, model_input in enumerate(self.model.input)}
					else:
						feed_dict = {self.model.input: embeddings_data[0][batch]}

					feed_dict.update({self.batch_id: i, self.step: step})

					if self.model.uses_learning_phase:
						feed_dict[K.learning_phase()] = False

					self.sess.run(self.assign_embeddings, feed_dict=feed_dict)
					self.saver.save(self.sess,
								os.path.join(self.log_dir, 'keras_embedding.ckpt'),
								epoch)

					i += self.batch_size

		for name, value in logs.items():
			if name in ['batch', 'size']:
				continue
			summary = tf.Summary()
			summary_value = summary.value.add()
			summary_value.simple_value = value.item()
			summary_value.tag = name
			self.writer.add_summary(summary, epoch)


		val_x = self.data
		img_vis,eta_vis,p_vis,bbox_vis = val_x
		y_pred = self.fun_lastfeaturemap(val_x)[0]
		x = [y_pred,eta_vis,p_vis,bbox_vis]

		loss1,loss2,loss3 = self.total_loss(x)

		
		summary_l1 = tf.Summary(value=[tf.Summary.Value(tag='loss1(p=1,eta=1)', simple_value=loss1)])
		self.writer.add_summary(summary_l1, epoch)

		summary_l2 = tf.Summary(value=[tf.Summary.Value(tag='loss2(p=1,eta=0)', simple_value=loss2)])
		self.writer.add_summary(summary_l2, epoch)

		summary_l3 = tf.Summary(value=[tf.Summary.Value(tag='loss3(p=0,eta=0)', simple_value=loss3)])
		self.writer.add_summary(summary_l3, epoch)

		# summary_l1 = tf.Summary.Value(tag='loss1(p=1,eta=1)', simple_value=loss1)
		# summary_l2 = tf.Summary.Value(tag='loss2(p=1,eta=0)', simple_value=loss2)
		# summary_l3 = tf.Summary.Value(tag='loss3(p=0,eta=0)', simple_value=loss3)
		# scalar_summaries = [summary_l1,summary_l2,summary_l3]

		img_summaries = []

		# writer.add_summary(value=[tf.Summary.Value(tag='pred', simple_value=y_pred[0][:10])])
		for i in range(14):
			try:
				s = StringIO()
			except:
				s = BytesIO()
			img = y_pred[0][:,:,i]*255
			np.save('npout', img)
			image = Image.fromarray(img)
			image=image.convert('RGB')
			image.save(s, format='PNG')
			# scipy.misc.toimage(img).save(s, format="png")
			img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),height=img.shape[0],width=img.shape[1])
			s.close()
		
			# Create a Summary value
			img_summaries.append(tf.Summary.Value(tag='pred%d layer'%i, image=img_sum))

		# Create and write Summary
		summary = tf.Summary(value=img_summaries)
		self.writer.add_summary(summary, epoch)

		self.writer.flush()


if __name__ == '__main__':
	# K.clear_session()
	import tensorflow as tf

	# from keras.backend.tensorflow_backend import set_session
	# config = tf.ConfigProto()
	# config.gpu_options.per_process_gpu_memory_fraction = 0.4
	# set_session(tf.Session(config=config))
	
	batch_size = cfg.BATCH_SIZE
	model = thoracic_model(batch_size)
	# weight_path = "/home/jianayang/project/weights/w.h5"
	# model.load_weights(weight_path,by_name=True)
	optimizer = Adam(lr=0.001,beta_1=0.9,beta_2=0.999,epsilon=None,decay=1e-6,amsgrad=False)
	model.compile(loss=lambda y_true,y_pred: y_pred, optimizer=optimizer)
	print(model.summary())

	#------------------------------------------------------------------------------
	epoch_num = cfg.EPOCH_NUM
	learning_rate = np.linspace(0.001, 0.001, epoch_num)
	change_lr = LearningRateScheduler(lambda epoch: 0.1**int(epoch/10)*0.001)
	early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')
	filepath = 'weights/weights.{epoch:02d}-{loss:.2f}.hdf5'
	check_point = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
		save_best_only=False, save_weights_only=False, mode='auto', period=20)

	input_file_path = cfg.TRAIN_SAMPLE_LIST
	val_file_path = cfg.VAL_SAMPLE_LIST
	img_base_dir = cfg.IMAGE_BASE_PATH
	# img_base_dir = '/data/users/jianayang/Thoracic_Disease/images'

	data_entry_path = cfg.DATA_ENTRY_PATH
	bbox_path = cfg.BBOX_PATH
	d = gen(data_entry_path,bbox_path)
	#----------------------------------------------------------------------------------------------


	# print('Generating train data...')
	# img,p,eta,bbox = gen_all_inputs(d,input_file_path,img_base_dir,False)
	# y = np.random.rand(img.shape[0])
	# print('Generating val data...')
	# img_val,p_val,eta_val,bbox_val = gen_all_inputs(d,val_file_path,img_base_dir,False)
	# y_val = np.random.rand(img.shape[0])
	# print('Start training...')
	# hist = model.fit([img,p,eta,bbox],y,epochs=epoch_num,batch_size=batch_size,validation_data=([img_val,p_val,eta_val,bbox_val],y_val),
	# 																callbacks=[check_point])

	# draw_train_curve(hist,'vgg19',prefix='auto')

	# step = 5000
	


	#-----------------------------------------------------------------------------
	train_steps = cfg.TRAIN_GENERATOR_STEP
	val_steps = cfg.VAL_GENERATOR_STEP
	# train_steps = 3
	# val_steps = 1

	log_dir = cfg.TENSORBOARD_LOG_PATH
	writer = tf.summary.FileWriter(log_dir)
	img_vis,eta_vis,p_vis,bbox_vis = gen_all_inputs(d,val_file_path,img_base_dir,num=1)
	data = [img_vis,eta_vis,p_vis,bbox_vis]
	# y_vis = np.ones(img_vis.shape[0])

	record_pred = LogPredG(logdir=log_dir,data=data)

	hist = model.fit_generator(gen_batch_inputs(d,input_file_path,img_base_dir,batch_size,False),
		steps_per_epoch=train_steps+1,
		validation_data=gen_batch_inputs(d,val_file_path,img_base_dir,batch_size,False),
		validation_steps=val_steps+1,
		epochs=epoch_num,
		verbose=1,
		# callbacks=[change_lr,check_point,early_stop,TensorBoard(log_dir=log_dir),record_pred])
		callbacks=[change_lr,check_point,early_stop,record_pred])

	print(hist.history.keys())

	# import cPickle as pickle 
	import pickle
	suffix = int(time.time()*10%100000)
	history_backup = os.path.join(cfg.HISTORY_BACKUP,'/history.%d.pkl'%suffix)
	with open(history_backup,'wb') as f:
		pickle.dump(hist.history,f)
	#------------------------------------------------------------------------------



