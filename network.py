import tensorflow as tf
import numpy as np
import sys

import tensorflow as tf
import numpy as np
import sys


class CNN: # cnn--> lstm --> cnnatt
	def __init__(self,wordVec,word_len):
	# def __init__(self,word_len,dict_size):
		self.class_num = 10
		
		self.window = 3
		self.windows=[1,2,3,4]

		self.word_len = word_len

		self.input_dim = 50
		self.hidden_dim = 100

		self.word = tf.placeholder(dtype=tf.int32,shape=[None,self.word_len],name='word')
		
		self.label = tf.placeholder(dtype=tf.int32,shape=[None,self.class_num],name='label')
		self.keep_prob = tf.placeholder(dtype=tf.float32,name='keep_prob')
		self.word_embedding = tf.get_variable(name='word_embedding', initializer=wordVec)
		



		


		# self.char_w = tf.get_variable(name='char_w',shape=[self.window,self.input_dim,1,self.hidden_dim])
		# self.char_b = tf.get_variable(name='char_b',shape=[self.hidden_dim])
		
		# with tf.variable_scope('RNN', initializer=tf.orthogonal_initializer()):
		# with tf.variable_scope('RNN'):# initializer=tf.orthogonal_initializer()):

		# 	layer1_forward = self.LSTM()
		# 	layer1_backward = self.LSTM()


		
		# self.word_all1= self.embed_l1(layer1_forward,layer1_backward,self.word,self.word_embedding,self.word_len,scope='WORD_l1')
		# input_forward = tf.nn.embedding_lookup(embedding,inputs)
		self.word_all1 = tf.nn.embedding_lookup(self.word_embedding,self.word)
		all_reps = []
		for w in self.windows:

			word_w = tf.get_variable(name='word_w_w'+str(w),shape=[w,self.input_dim,1,self.hidden_dim])
			word_b = tf.get_variable(name='word_b_w'+str(w),shape=[self.hidden_dim])

			rep1 = self.cnn(self.word_all1,word_w,word_b,self.word_len,w)
	
			all_reps.append(rep1)

		self.final_rep = tf.concat(axis=1,values=all_reps)

		self.dense_w = tf.get_variable(name='dense_w',shape=[4*self.hidden_dim,self.class_num])
		self.dense_b = tf.get_variable(name='dense_b',shape=[self.class_num])
		out = tf.reshape(tf.add(tf.matmul(self.final_rep,self.dense_w),self.dense_b),[-1,self.class_num])   

		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out,labels=self.label))

		self.prob = tf.nn.softmax(out,1)
		self.prediction = tf.argmax(self.prob,1)
		self.accuracy = tf.cast(tf.equal(self.prediction,tf.argmax(self.label,1)),tf.float32)



		
		

		


	
	
		
	def cnn(self,inputs,w,b,lens,window):

		# input_forward = tf.nn.embedding_lookup(embedding,inputs)
		input_forward = tf.expand_dims(inputs,-1)
	
		conv = tf.nn.conv2d(
			input_forward,
			w,
			strides=[1,1,1,1],
			padding='VALID', # padding='VALID',
			name='conv'
			)
		h = tf.nn.bias_add(conv,b)
		pooled = tf.nn.max_pool(
			h,
			ksize=[1,lens-window+1,1,1],
			strides=[1,1,1,1],
			padding='VALID',
			name='pool'
			)
		h = tf.reshape(pooled,[-1,self.hidden_dim])
		h = tf.nn.tanh(h)

		return h
	

	def embed_l1(self,cell_forward,cell_backward,inputs,embedding,lens,scope='WORD',reuse=False,raw=True):

		

		if raw:
			inputs_forward = tf.nn.embedding_lookup(embedding,inputs)
			
		else:
			inputs_forward = inputs
			

		
	
		with tf.variable_scope(scope+'_LSTM_FORWARD'):
			if reuse:
				tf.get_variable_scope().reuse_variables()

			outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_forward,cell_backward,inputs_forward,dtype=tf.float32)


		
		output_h = tf.concat(axis=2,values=outputs)



		return output_h

	def LSTM(self,layers=1):
		lstms = []

		for num in range(layers):

			lstm = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim, forget_bias=1.0)
			lstm = tf.contrib.rnn.DropoutWrapper(lstm,output_keep_prob=self.keep_prob)
			lstms.append(lstm)

		lstms = tf.contrib.rnn.MultiRNNCell(lstms)
		return lstms
class LSTM_CNN: # cnn--> lstm --> cnnatt
	def __init__(self,wordVec,word_len):
	# def __init__(self,word_len,dict_size):
		self.class_num = 10
		
		self.window = 3
		self.windows=[1,2,3,4]

		self.word_len = word_len

		self.input_dim = 50
		self.hidden_dim = 100

		self.word = tf.placeholder(dtype=tf.int32,shape=[None,self.word_len],name='word')
		
		self.label = tf.placeholder(dtype=tf.int32,shape=[None,self.class_num],name='label')
		self.keep_prob = tf.placeholder(dtype=tf.float32,name='keep_prob')
		self.word_embedding = tf.get_variable(name='word_embedding', initializer=wordVec)
		



		


		# self.char_w = tf.get_variable(name='char_w',shape=[self.window,self.input_dim,1,self.hidden_dim])
		# self.char_b = tf.get_variable(name='char_b',shape=[self.hidden_dim])
		
		# with tf.variable_scope('RNN', initializer=tf.orthogonal_initializer()):
		with tf.variable_scope('RNN'):# initializer=tf.orthogonal_initializer()):

			layer1_forward = self.LSTM()
			layer1_backward = self.LSTM()


		
		self.word_all1= self.embed_l1(layer1_forward,layer1_backward,self.word,self.word_embedding,self.word_len,scope='WORD_l1')

		all_reps = []
		for w in self.windows:

			word_w = tf.get_variable(name='word_w_w'+str(w),shape=[w,2*self.hidden_dim,1,self.hidden_dim])
			word_b = tf.get_variable(name='word_b_w'+str(w),shape=[self.hidden_dim])

			rep1 = self.cnn(self.word_all1,word_w,word_b,self.word_len,w)
	
			all_reps.append(rep1)

		self.final_rep = tf.concat(axis=1,values=all_reps)

		self.dense_w = tf.get_variable(name='dense_w',shape=[4*self.hidden_dim,self.class_num])
		self.dense_b = tf.get_variable(name='dense_b',shape=[self.class_num])
		out = tf.reshape(tf.add(tf.matmul(self.final_rep,self.dense_w),self.dense_b),[-1,self.class_num])   

		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out,labels=self.label))

		self.prob = tf.nn.softmax(out,1)
		self.prediction = tf.argmax(self.prob,1)
		self.accuracy = tf.cast(tf.equal(self.prediction,tf.argmax(self.label,1)),tf.float32)



		
		

		


	
	
		
	def cnn(self,inputs,w,b,lens,window):

		# input_forward = tf.nn.embedding_lookup(embedding,inputs)
		input_forward = tf.expand_dims(inputs,-1)
	
		conv = tf.nn.conv2d(
			input_forward,
			w,
			strides=[1,1,1,1],
			padding='VALID', # padding='VALID',
			name='conv'
			)
		h = tf.nn.bias_add(conv,b)
		pooled = tf.nn.max_pool(
			h,
			ksize=[1,lens-window+1,1,1],
			strides=[1,1,1,1],
			padding='VALID',
			name='pool'
			)
		h = tf.reshape(pooled,[-1,self.hidden_dim])
		h = tf.nn.tanh(h)

		return h
	

	def embed_l1(self,cell_forward,cell_backward,inputs,embedding,lens,scope='WORD',reuse=False,raw=True):

		

		if raw:
			inputs_forward = tf.nn.embedding_lookup(embedding,inputs)
			
		else:
			inputs_forward = inputs
			

		
	
		with tf.variable_scope(scope+'_LSTM_FORWARD'):
			if reuse:
				tf.get_variable_scope().reuse_variables()

			outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_forward,cell_backward,inputs_forward,dtype=tf.float32)


		
		output_h = tf.concat(axis=2,values=outputs)



		return output_h

	def LSTM(self,layers=1):
		lstms = []

		for num in range(layers):

			lstm = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim, forget_bias=1.0)
			lstm = tf.contrib.rnn.DropoutWrapper(lstm,output_keep_prob=self.keep_prob)
			lstms.append(lstm)

		lstms = tf.contrib.rnn.MultiRNNCell(lstms)
		return lstms
