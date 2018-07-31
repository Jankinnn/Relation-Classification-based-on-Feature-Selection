#coding=utf-8
import os
import numpy as np
import pandas as pd
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


def getEmbed(prefix='../data/',file='travel_word_50d_word2vec.word2vec',filters=False,thre=300):
# def getEmbed(prefix='../data/',file='sgns.zhihu.word.ccks'):
	if filters:
		f = open(os.path.join(prefix,'chi_square_word_%d.txt' % thre),'r')
		filter_words = {}
		while True:
			word = f.readline()
			if not word:
				break
			filter_words[word.strip()] = 1
		f.close()

	fr = open(os.path.join(prefix,file),'r')
	wordVec = []
	wordMap = {}

	fr.readline()
	while True:
		line = fr.readline()
		if not line:
			break
		w = line.strip().split()[0]
		if filters and w not in filter_words:
			continue
		content = map(float,line.strip().split()[1:])
		

		wordMap[w] = len(wordVec)
		wordVec.append(content)

	wordMap['UNK'] = len(wordMap)
	wordMap['BLANK'] = len(wordMap)
	wordVec.append([0.0]*50)
	wordVec.append([0.0]*50)
	wordVec = np.reshape(np.array(wordVec,dtype=np.float32),(-1,50))

	return wordMap,wordVec

def getTypes(prefix='../data/',file='class_types.txt'):
	fr = open(os.path.join(prefix,file),'r')
	class_types = {}
	num = 0
	while True:
		line = fr.readline()
		if not line:
			break
		class_types[line.strip()] = num
		num+=1

	fr.close()
	return class_types



def getQuestion(wordMap,class_types,prefix='../data/',file='corpus_train.txt',word_size=30,filters=False,thre=300): #word_size 41 , char_size 60

	if filters:
		f = open(os.path.join(prefix,'chi_square_word_%d.txt' % thre),'r')
		filter_words = {}
		while True:
			word = f.readline()
			if not word:
				break
			filter_words[word.strip()] = 1
		f.close()

	fr = open(os.path.join(prefix,file),'r')
	
	train_set = []

	while True:
		line = fr.readline()
		if not line:
			break
		content = line.strip().split()
		label = [0]*len(class_types)
		label[class_types[content[0]]] = 1

		word = []
		for w in content[1:]:
			if len(word)>=word_size:
				break
			if filters and w not in filter_words:
				continue
			if w in wordMap:
				w = wordMap[w]
			else:
				w = wordMap['UNK']
			word.append(w)
		while len(word)<word_size:
			word.append(wordMap['BLANK'])


		train_set.append((word,label))

	fr.close()
	return train_set
