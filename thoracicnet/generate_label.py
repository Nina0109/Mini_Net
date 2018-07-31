import csv

disease2id = {'No Finding':0,'Atelectasis':1,'Cardiomegaly':2,'Effusion':3,'Infiltrate':4,
'Mass':5,'Nodule':6,'Pneumonia':7,'Pneumothorax':8,'Consolidation':9,'Edema':10,'Emphysema':11,'Fibrosis':12,'Pleural_Thickening':13,'Hernia':14,'Infiltration':4}
disease_type = 14

def gen(data_entry,bbox_list):
	d = {}
	with open(data_entry,'r') as f:
		lines = csv.reader(f)

		for line in lines:
			# print line
			# line = line.strip().split(',')
			img_path = line[0]
			d[img_path] = {}

			disease_label = line[1].split('|')
			disease_id = [0 for _ in range(disease_type)]
			for dis in disease_label:
				if dis is not 'No Finding':
					disease_id[disease2id[dis]-1] = 1
			d[img_path]['dl'] = disease_id

			w = line[7]
			h = line[8]
			d[img_path]['w'] = float(w)
			d[img_path]['h'] = float(h)
			tmp = [0 for _ in range(disease_type)]
			d[img_path]['bbox'] = [tmp,[[-1,-1,-1,-1] for _ in range(disease_type)]]

	with open(bbox_list) as f:
		lines = csv.reader(f)

		for line in lines:

			# line = line.strip().split(',')
			img_path = line[0]
			disease_label = line[1]
			xa = line[2]
			wa = line[3]
			ya = line[4]
			ha = line[5]
			w = d[img_path]['w']
			h = d[img_path]['h']

			xa = float(xa)*1.0/w
			wa = float(wa)*1.0/w
			ya = float(ya)*1.0/h
			ha = float(ha)*1.0/h

			# print(xa,wa,ya,ha)			

			# tmp = [0 for _ in range(disease_type)]
			# tmp[disease2id[disease_label]-1] = 1

			d[img_path]['bbox'][0][disease2id[disease_label]-1] = 1

			d[img_path]['bbox'][1][disease2id[disease_label]-1] = [xa,wa,ya,ha]

	return d









