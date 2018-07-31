import numpy as np
import config as cfg

if __name__ == '__main__':
	ftrain = open(cfg.TRAIN_SAMPLE_LIST,'w')
	fval = open(cfg.VAL_SAMPLE_LIST,'w')
	with open(cfg.TRAIN_VAL_SAMPLE_LIST) as f:
		for line in f:
			r = np.random.random()
			if r < 0.3:
				fval.write(line)
			else:
				ftrain.write(line)
	ftrain.close()
	fval.close()