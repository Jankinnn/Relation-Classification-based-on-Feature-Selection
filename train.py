import tensorflow as tf
import numpy as np
import time
import datetime
import os
from init import *
import network
import pandas as pd
import random


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('summary_dir','../summary/','path to store summary')
tf.app.flags.DEFINE_string('gpu','','gpu id')
tf.app.flags.DEFINE_string('v','1','model version')
tf.app.flags.DEFINE_string('batch','10','batch size')
tf.app.flags.DEFINE_string('epochs','22','train epochs')
tf.app.flags.DEFINE_string('dim','200','dimension')
tf.app.flags.DEFINE_string('target','../model','save model dir')
tf.app.flags.DEFINE_string('source','../data','data dir to load')
tf.app.flags.DEFINE_string('train','1','is train')
tf.app.flags.DEFINE_string('id','14000','model id')
# tf.app.flags.DEFINE_string('word','25','word len')
tf.app.flags.DEFINE_string('word','30','word len')

tf.app.flags.DEFINE_string('filter','0','if filter')
tf.app.flags.DEFINE_string('thre','300','filter thre')
tf.app.flags.DEFINE_string('drop','0.5','drop')
tf.app.flags.DEFINE_string('lr','0.001','drop')
def loadData(prefix,file):
	fr = open(os.path.join(prefix,file),'r')
	train_list = []
	while True:
		line = fr.readline()
		if not line:
			break
		train_list.append(map(int,line.strip().split()))
	fr.close()
	return train_list


def main(_):
	
	hidden_dim = int(FLAGS.dim)
	batch = int(FLAGS.batch)

	is_train = int(FLAGS.train)
	source = FLAGS.source
	target = FLAGS.target
	total_epochs = int(FLAGS.epochs)
	word_len = int(FLAGS.word)

	drop = float(FLAGS.drop)

	filters = int(FLAGS.filter)
	if filters == 0:
		filters = False
	else:
		filters = True
	filters_thre = int(FLAGS.thre)


	wordMap,wordVec = getEmbed(filters=filters,thre=filters_thre)

	
	class_types = getTypes()

	train_set = getQuestion(wordMap,class_types,source,'corpus_train.txt',word_size=word_len,filters=filters,thre=filters_thre)
	
	dev_set = getQuestion(wordMap,class_types,source,'corpus_dev.txt',word_size=word_len,filters=filters,thre=filters_thre)



	

	gpu_options = tf.GPUOptions(visible_device_list=FLAGS.gpu,allow_growth=True)

	with tf.Graph().as_default():
		sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True))
		with sess.as_default():

			
			initializer = tf.contrib.layers.xavier_initializer()
			# initializer = tf.orthogonal_initializer()
			with tf.variable_scope('model',initializer=initializer):
				
				m = network.CNN(wordVec,word_len)
				


			global_step = tf.Variable(0,name='global_step',trainable=False)
			optimizer = tf.train.AdamOptimizer(float(FLAGS.lr))
			# optimizer = tf.train.RMSPropOptimizer(float(FLAGS.lr))

			# lr = tf.train.exponential_decay(float(FLAGS.lr),global_step=global_step,decay_steps=500,decay_rate=0.98)
			# optimizer = tf.train.AdamOptimizer(lr)

			# optimizer = tf.train.MomentumOptimizer(lr,momentum=0.9)

			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):
				train_op = optimizer.minimize(m.loss,global_step=global_step)
			
			sess.run(tf.global_variables_initializer())

			saver = tf.train.Saver(max_to_keep=None)
				# merged_summary = tf.summary.merge_all()
				# summary_writer = tf.summary.FileWriter(FLAGS.summary_dir,sess.graph)
			
			def train_step(word,label):
				feed_dict = {}
				feed_dict[m.word] = word
				feed_dict[m.label] = label
				feed_dict[m.keep_prob] = drop


				_,step,loss,accuracy  = sess.run([train_op, global_step, m.loss, m.accuracy],feed_dict)
				time_str = datetime.datetime.now().isoformat()
				accuracy = np.reshape(np.array(accuracy),(-1))
				accuracy = np.mean(accuracy)
				# summary_writer.add_summary(summary,step)

				if step % 10 == 0:
					tempstr = "{}: step {}, softmax_loss {:g}, acc {:g}".format(time_str, step, np.mean(loss), accuracy)
					print tempstr

			def dev_step(word,label):
				feed_dict = {}

				feed_dict[m.word] = word

				feed_dict[m.label] = label

				feed_dict[m.keep_prob] = 1.0
				loss, accuracy,prediction = sess.run([m.loss,m.accuracy,m.prediction],feed_dict)

				return loss,accuracy,prediction

			
			def getData(lst,dataset):
				word = []
				label = []
				

				for k in lst:
					w,l = dataset[k]
					word.append(w)
					label.append(l)

				
				word = np.array(word)
				label = np.array(label)

				return word,label

			def evaluate(total_labels,total_pred):
				a = np.sum(total_labels)
				b = np.sum(total_pred)
				c = 0
				for i,j in zip(total_labels,total_pred):
					if i==1 and j==1:
						c+=1
				if b<=0:
					precision=0.0
				else:
					precision = float(c)/float(b)
				recall = float(c)/float(a)
				f1 = 2*precision*recall/(precision+recall)
				return precision,recall,f1


			
			max_accuracy = 0.0
			
			for one_epoch in range(total_epochs):
				
				print('turn: ' + str(one_epoch))
				temp_order = range(len(train_set))
				np.random.shuffle(temp_order)
				for i in range(int(len(temp_order)/float(batch))):
					
					temp_input = temp_order[i*batch:(i+1)*batch]


					word,label = getData(temp_input,train_set)
					train_step(word,label)

					current_step = tf.train.global_step(sess,global_step)
					if (current_step%50==0):
						accuracy = []
						losses=[]
						total_pred=[]
						total_labels = []
						dev_order = range(len(dev_set))
						for i in range(int(len(dev_order)/float(batch))):
							temp_input = dev_order[i*batch:(i+1)*batch]
							word,label = getData(temp_input,dev_set)
							loss,accs,prediction = dev_step(word,label)
					#if current_step == 50:
							for acc in accs:
								accuracy.append(acc)
							
			

						accuracy = np.reshape(np.array(accuracy),(-1))
						accuracy = np.mean(accuracy)
						
						print('dev...')

						
						if accuracy > max_accuracy:
							max_accuracy = accuracy
							
							# if losses < min_loss:
							# 	min_loss = losses
							print('accuracy:  ' + str(accuracy))
							# print('precision: ' + str(precision))
							# print('recall:    ' + str(recall))
							# print('f1:        ' + str(f1)) 
							# print('loss: ' + str(losses))

							# if accuracy < 91 and min_loss>0.2:
							# # if min_loss>0.2:
							# 	continue

							print 'saving model'
							# path = saver.save(sess,target +'/CNN_model.'+FLAGS.v,global_step=current_step)
							path = saver.save(sess,target +'/laiye_model.'+FLAGS.v,global_step=0)
							tempstr = 'have saved model to '+path
							print tempstr
			

		# 	path = target +'/laiye_model.'+FLAGS.v+'-'+'0'
		# 	# path = target +'/CNN_model.'+FLAGS.v+'-'+FLAGS.id	
		# 	print 'load model:',path
		# 	saver = tf.train.Saver()
		# 	saver.restore(sess,path)
		# 	print 'end load model'

		# 	total_prob = []
		# 	total_pred = []
		# 	temp_order = range(len(test_set))
		# 	for i in range(0,len(test_set),batch):
		# 		temp_input = range(i,min(len(test_set),i+batch))
		# 		q1_word,q1_word_mask,q2_word,q2_word_mask,q1_word_mask01,q2_word_mask01 = getDataT(temp_input,test_set)
		# 		prob,prediction = test_step(q1_word,q1_word_mask,q2_word,q2_word_mask,q1_word_mask01,q2_word_mask01)
		# #if current_step == 50:
		# 		prob = np.reshape(prob,(-1,2))
		# 		for p in prob:
		# 			total_prob.append(p)
		# 		for p in prediction:
		# 			total_pred.append(p)
		# 	total_prob = np.reshape(np.array(total_prob),(-1,2))
		# 	total_pred = np.reshape(np.array(total_pred),(-1))
			
		# 	make_submission(total_pred)







	                        


	        # else:




if __name__ == '__main__':
	tf.app.run()
