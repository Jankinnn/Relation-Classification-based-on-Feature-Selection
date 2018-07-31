#coding=utf-8
import sys
import numpy as np
reload(sys)
sys.setdefaultencoding('utf-8')

thre = int(sys.argv[1])
fr = open('../data/corpus_train.txt','r')
class_types = {}
class_types_num = 0

all_text = []
while True:
	line = fr.readline()
	if not line:
		break
	content = line.strip().split()
	all_text.append(content)
	if content[0] not in class_types:
		class_types[content[0]] = class_types_num
		class_types_num+=1
fr.close()

word_text_map = {}

text_num_of_each_class = [0]*class_types_num

for text in all_text:
	duplicate = {}
	text_num_of_each_class[class_types[text[0]]] +=1
	for word in text[1:]:
		if word in duplicate:
			continue
		duplicate[word] = 1
		if word in word_text_map:
			text_map = word_text_map[word]
		else:
			text_map = [0]*class_types_num
		text_map[class_types[text[0]]] += 1
		word_text_map[word] = text_map

total_text_num = len(all_text)

word_text_list = []
word_list = []
for k in word_text_map.keys():
	word_list.append(k)
	word_text_list.append(word_text_map[k])

word_text_list = np.array(word_text_list,dtype=np.float32)
text_num_of_each_class = np.array(text_num_of_each_class,dtype=np.float32)

word_total_frequence = np.reshape(np.sum(word_text_list,1),(-1,1))
word_in_other_class = word_total_frequence - word_text_list


other_word_in_class = text_num_of_each_class - word_text_list

not_the_word_and_class = total_text_num - word_text_list - word_in_other_class - other_word_in_class

word_num = len(word_list)
print('word num:\t%d' % word_num)

chi_square_value = np.divide(np.square(np.multiply(word_text_list,not_the_word_and_class) - np.multiply(word_in_other_class,other_word_in_class)) , np.multiply(word_text_list+word_in_other_class, other_word_in_class+not_the_word_and_class))

chosed_word = {}
for cla in range(class_types_num):
	cur_chi = chi_square_value[:,cla]
	order = np.argsort(-cur_chi)
	for i in order[:thre]:
		if i in chosed_word:
			continue
		chosed_word[i] = 1

fw = open('../data/chi_square_word_%d.txt' % thre,'w')
for k in chosed_word:
	fw.write(word_list[k]+'\n')
fw.close()










