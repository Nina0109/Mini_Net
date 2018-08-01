import keras
import keras.backend as K
import tensorflow as tf

from keras.models import Model
from keras.layers import MaxPooling2D,Conv2D,Input,BatchNormalization
from keras.layers import Activation
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.vgg16 import VGG16
from keras import regularizers


from thoracicnet.thoracicloss import lossLayer
import numpy as np

import thoracicnet.config as cfg

def thoracic_prob(input_img):
	# input_img = Input(tensor=input_tensor, shape=input_shape)
	x = VGG16(include_top=False, weights='imagenet',input_shape=(512,512,3))(input_img)
	x = MaxPooling2D(pool_size=(2, 2),strides=(1,1),name='ps')(x)
	# x = Conv2D(512,(3,3),activation=None,name='conv1',kernel_regularizer=regularizers.l2(0.01))(x)
	x = Conv2D(512,(3,3),activation=None,name='conv1')(x)
	x = BatchNormalization(name='bn1')(x)
	x = Activation('relu',name='relu1')(x)
	class_num = cfg.CLASS_NUM
	# prob = Conv2D(class_num,(1,1),activation='sigmoid',name='conv2',kernel_regularizer=regularizers.l2(0.01))(x)
	prob = Conv2D(class_num,(1,1),activation='sigmoid',name='conv2')(x)
	# a normalization left
	return prob

def thoracic_model(batch_size):
	img_input = Input(shape=(512,512,3))
	eta_input = Input(shape=(cfg.CLASS_NUM,))
	p_input = Input(shape=(cfg.CLASS_NUM,))
	bbox_input = Input(shape=(cfg.CLASS_NUM,4))


	y_pred = thoracic_prob(img_input)
	th_loss = lossLayer(lambda_bbox=0.5,bs=batch_size,name='thoracic_loss')([y_pred,eta_input,p_input,bbox_input])
	thoracic_model = Model(inputs=[img_input,eta_input,p_input,bbox_input], outputs=th_loss, name='thoracic')
	return thoracic_model


def thoracic_model_deploy(batch_size):
	img_input = Input(shape=(512,512,3))

	y_pred = thoracic_prob(img_input)

	thoracic_model = Model(inputs=img_input, outputs=y_pred, name='thoracic_deploy')
	return thoracic_model


def _thoracic_loss(y_pred,eta,p,bbox,lambda_bbox=0.5):
	# y_predis a PxP tensor, input img is a 512x512 tensor
	# bbox=(xa,wa,ya,ha)
	lambda_bbox = 0.5
	# p2 = K.ones_like(y_pred) - y_pred
	p2 = 1 - y_pred
	# p_y_x = 1-tf.reduce_prod(p2)
	p_y_x = 1-K.prod(p2)
	

	P = K.cast(K.shape(y_pred)[1],'float')
	x = bbox[0][0]*P/float(512)
	y = bbox[0][1]*P/float(512)
	w = bbox[0][2]*P/float(512)
	h = bbox[0][3]*P/float(512)

	x = K.cast(x,'int32')
	y = K.cast(y,'int32')
	w = K.cast(w,'int32')
	h = K.cast(h,'int32')

	in_box = y_pred[:,y:y+h,x:x+w]

	p1 = K.prod(in_box)
	p3 = K.prod(1-in_box)
	p2 = K.prod(y_pred)
	p_y_x_bbox = p3*p2/p1

	loss = -eta*K.log(p_y_x_bbox)-(1-eta)*p*K.log(p_y_x)-(1-eta)*(1-p)*K.log(1-p_y_x)
	print(loss)
	return loss


if __name__ == '__main__':
	# y_pred = 0.1*K.ones((1,5,5))
	img = K.constant(np.random.random((1,512,512,3)))
	eta = K.constant([[1.0]])
	p = K.constant([[1.0]])
	bbox = K.constant([[10.0,15.0,500.0,500.0]])
	# loss = _thoracic_loss(y_pred,eta,p,bbox,lambda_bbox=0.5)

	model = thoracic_model()
	model.compile(optimizer='rmsprop', loss=None)

	loss = model([img,eta,p,bbox])

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	
	print(sess.run(loss))





