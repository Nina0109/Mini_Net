import keras
from keras.models import Model
from keras.layers import MaxPooling2D,Conv2D,Input,BatchNormalization
# from keras.layers import Activations
import keras
import keras.backend as K
from keras.engine.topology import Layer

import tensorflow as tf

class lossLayer(Layer):
    '''
    Input shape: a list of 4 tensor
    y_pred:(14,P,P)
    eta:(14,1)
    p:(14,1)
    bbox:(1,4)
    '''

    def __init__(self, lambda_bbox, bs, **kwargs):
        self.lambda_bbox = lambda_bbox
        self.bs = bs
        super(lossLayer, self).__init__(**kwargs)

    # def build(self, input_shape):
    #     # Create a trainable weight variable for this layer.
    #     self.kernel = self.add_weight(name='kernel', 
    #                                   shape=(input_shape[1], self.output_dim),
    #                                   initializer='uniform',
    #                                   trainable=True)
    #     super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
      # print(x[0][0].shape)
      # print(x[1][0].shape)
      # print(x[2][0].shape)
      # print(x[3][0].shape)
      loss = 0
      # self.thoracic_loss(x[0][:,:,:,0],x[1][:,0],x[2][:,0],x[3][:,0],lambda_bbox=self.lambda_bbox)

      for i in range(self.bs):
        for j in range(14):
          loss += self.thoracic_loss(x[0][i,:,:,j],x[1][i,j],x[2][i,j],x[3][i,j],lambda_bbox=self.lambda_bbox)
      loss = loss/self.bs
      return loss

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)

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

      y_pred = K.minimum(1.0,y_pred)
      
      y_pred_tmp = 1 - y_pred
      y_pred_tmp = K.minimum(1.0,y_pred_tmp)

      y_pred = y_pred*0.02+0.98   # preform normalization
      y_pred_tmp = y_pred_tmp*0.02+0.98  # preform normalization

      p_y_x = 1-K.prod(y_pred_tmp) 
      

      P = K.cast(K.shape(y_pred)[1],'float')

      x = bbox[0]*P
      y = bbox[1]*P
      w = bbox[2]*P
      h = bbox[3]*P

      x = K.cast(x,'int32')
      y = K.cast(y,'int32')
      w = K.cast(w,'int32')
      h = K.cast(h,'int32')

      # in_box = y_pred[:,y:y+h,x:x+w]
      in_box = y_pred[y:y+h,x:x+w]

      p1 = K.prod(in_box)
      # p3 = K.prod(1.0-in_box)
      p3 = K.prod(y_pred_tmp[y:y+h,x:x+w])
      # p2 = K.prod(1-y_pred)
      p2 = K.prod(y_pred_tmp)
      p_y_x_bbox = p1*p2/p3 + 1e-10

      loss = -eta*K.log(p_y_x_bbox)-(1.0-eta)*p*K.log(p_y_x)-(1.0-eta)*(1.0-p)*K.log(1.0-p_y_x)
      return loss


      # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html
      # p1 = K.prod(in_box)
      # p3 = K.prod(1.0-in_box)
      # p2 = K.prod(y_pred)
      # p_y_x_bbox = p3*p2/p1

if __name__ == '__main__':
  e = 1e-20
  K.set_epsilon(e)
  # y_pred = 0.1*K.ones((14,5,5))
  # eta = K.constant([[1.0]])
  # p = K.constant([[1.0]])
  # bbox = K.constant([[10.0,15.0,500.0,500.0]])
  # import cPickle as pickle
  import pickle
  import numpy as np
  # with open('input.pkl','rb') as f:
  #   x = pickle.load(f)
  # img,p,eta,bbox = x
  p = np.array([[1,0,0,0,0,0,0,0,0,0,0,0,0,0]])
  eta = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
  bbox = np.array(
  # [[[ 0.07522886,  0.18282728,  0.02901359,  0.0264749 ],
  [[[ 0.5,  0.1,  0.5,  0.1 ],
  [-1.,         -1.,         -1.,         -1.        ],
  [-1.,         -1.,         -1.,         -1.        ],
  [-1.,         -1.,         -1.,         -1.        ],
  [-1.,         -1.,         -1.,         -1.        ],
  [-1.,         -1.,         -1.,         -1.        ],
  [-1.,         -1.,         -1.,         -1.        ],
  [-1.,         -1.,         -1.,         -1.        ],
  [-1.,         -1.,         -1.,         -1.        ],
  [-1.,         -1.,         -1.,         -1.        ],
  [-1.,         -1.,         -1.,         -1.        ],
  [-1.,         -1.,         -1.,         -1.        ],
  [-1.,         -1.,         -1.,         -1.        ],
  [-1.,         -1.,         -1.,         -1.        ]]]) 
  # y_pred = np.load('testout.npy')
  y_pred = np.ones((1,11,11,14))*0.5

  inputs = [K.constant(y_pred),K.constant(eta),K.constant(p),K.constant(bbox)]
  out = lossLayer(lambda_bbox=0.5,bs=1,name='thoracic_loss')(inputs)

  lambda_bbox = 0.5

  y_pred = K.constant(y_pred)[0,:,:,1]
  bbox = K.constant(bbox)[0,1]
  eta = K.constant(eta)[0,1]
  p = K.constant(p)[0,1]


  print(y_pred.shape)
  print(bbox.shape)
  print(eta.shape)
  print(p.shape)

  y_pred = K.minimum(1.0,y_pred)
  

  y_pred_tmp = 1 - y_pred
  y_pred_tmp = K.minimum(1.0,y_pred_tmp)
  y_pred_tmp = y_pred_tmp*0.02 + 0.98
  y_pred = y_pred*0.02 + 0.98   # preform normalization

  p_y_x = 1-(K.prod(y_pred_tmp)+K.epsilon())

  P = K.cast(K.shape(y_pred)[1],'float')

  x = bbox[0]*P
  y = bbox[1]*P
  w = bbox[2]*P
  h = bbox[3]*P

  x = K.cast(x,'int32')
  y = K.cast(y,'int32')
  w = K.cast(w,'int32')
  h = K.cast(h,'int32')

  in_box = y_pred[y:y+h,x:x+w]
  print('in_box:',in_box.shape)

  p1 = K.prod(in_box) + K.epsilon()
  p3 = K.prod(1.0-in_box) + K.epsilon()
  p2 = K.prod(1-y_pred) + K.epsilon()
  p_y_x_bbox = p1*p2/p3 + K.epsilon()

  loss1 = -eta*K.log(p_y_x_bbox)
  loss2 = -(1.0-eta)*p*K.log(p_y_x)
  loss3 = -(1.0-eta)*(1.0-p)*K.log(1.0-p_y_x)

  # loss = -eta*K.log(p_y_x_bbox)-(1.0-eta)*p*K.log(p_y_x)-(1.0-eta)*(1.0-p)*K.log(1.0-p_y_x)

 
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  # print('P: ')
  # print(sess.run(P))

  print('x: ')
  print(sess.run(x))

  print('y: ')
  print(sess.run(y))

  print('w: ')
  print(sess.run(w))

  print('h: ')
  print(sess.run(h))

  print(sess.run(bbox))
  print(sess.run(eta))
  print(sess.run(p))
  print(sess.run(y_pred))

  print('in_box:',sess.run(in_box))
  print(sess.run(in_box).shape)

  print('1-in_box:',sess.run(1-in_box))
  print(sess.run(1-in_box).shape)

  print('P1:',sess.run(p1))
  print('P3:',sess.run(p3))
  print('P2:',sess.run(p2))
  print('p_y_x_bbox:',sess.run(p_y_x_bbox))

  print('y_pred:',sess.run(y_pred))
  print('y_pred_tmp:',sess.run(y_pred_tmp))

  print('p_y_x:',sess.run(p_y_x))
  print('1-p_y_x:',sess.run(1-p_y_x))

  # print()
  print('loss1: ',sess.run(loss1))
  print('loss2: ',sess.run(loss2))
  print('loss3: ',sess.run(loss3))
  print('out: ',sess.run(out))

