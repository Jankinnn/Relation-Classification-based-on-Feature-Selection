#coding=utf-8
import sys
import numpy as np
reload(sys)
sys.setdefaultencoding('utf-8')

fr = open('../data/corpus_txt.txt','r')

total_data = {}
while True:
	line = fr.readline()
	if not line:
		break
	class_type = line.strip().split()[0]
	if class_type in total_data:
		dataset = total_data[class_type]
	else:
		dataset = []
	dataset.append(line.strip())
	total_data[class_type] = dataset
fr.close()

ftrain = open('../data/corpus_train.txt','w')
fdev = open('../data/corpus_dev.txt','w')

for class_type in total_data.keys():
	sizes = len(total_data[class_type])
	order = range(sizes)
	np.random.shuffle(order)
	for i in order[:int(sizes*0.8)]:
		ftrain.write(total_data[class_type][i] + '\n')
	for i in order[int(sizes*0.8):]:
		fdev.write(total_data[class_type][i] + '\n')
ftrain.close()
fdev.close()

