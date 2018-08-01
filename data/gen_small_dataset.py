import sys
sys.path.insert(0,'E:/Preparation_For_Summer/project/mini_net')


from random import shuffle
import thoracicnet.config as cfg

disease2id = cfg.disease2id

def find_without_bbox_item():
	f1 = open('train_test.txt','w')

	with_box = []

	with open('BBox_List_2017.csv','r') as f:
		for line in f:
			line = line.strip().split(',')
			with_box.append(line[0])



	with open('Data_Entry_2017.csv','r') as f:
		for line in f:
			if(line.strip().split(',')[0]) not in with_box:
				f1.write(line)

	f1.close()


def count_disease():
	count  = {}
	with open('Data_Entry_2017.csv','r') as f:
		for line in f:
			line = line.strip().split(',')
			ids = line[1].split('|')
			for item in ids:
				idx = disease2id[item]
				count[idx] = count.get(idx,0) + 1

	for k,v in count.items():
		print('disease%d : %d'%(k,v))



def select_small_set():
	tmp = [ [] for _ in range(15)]
	with open('Data_Entry_2017.csv','r') as f:
		for line in f:
			ids = line.strip().split(',')[1].split('|')
			if len(ids)!=1:
				continue
			idx = disease2id[ids[0]]
			tmp[idx].append(line)

	select_train = []
	select_val = []
	select_test = []

	for i in range(1,2):
		select_train += tmp[i][:200]
		select_val += tmp[i][-50:]
		select_test += tmp[i][-100:-50]

	shuffle(select_train)
	shuffle(select_val)
	shuffle(select_test)

	with open('train_test.txt','w') as f:
		for item in select_train:
			item = item.strip().split(',')[0]+'\n'
			f.write(item)

	with open('val_test.txt','w') as f:
		for item in select_val:
			item = item.strip().split(',')[0]+'\n'
			f.write(item)

	with open('test_test.txt','w') as f:
		for item in select_test:
			item = item.strip().split(',')[0]+'\n'
			f.write(item)	



if __name__ == '__main__':
	select_small_set()


