import os

####################################################################
# Training Parameters
####################################################################

BATCH_SIZE = 4
EPOCH_NUM = 1000

TRAIN_GENERATOR_STEP = 100 #15142
VAL_GENERATOR_STEP = 20 #6486


####################################################################
# Data file path
####################################################################
# PROJECT_ROOT = os.path.abspath('./../')
PROJECT_ROOT = os.path.abspath('./')

# The dir where you put all the .txt and .csv labels
DATA_PATH = os.path.join(PROJECT_ROOT,'data')    

TRAIN_SAMPLE_LIST = os.path.join(DATA_PATH,'train_test.txt')
VAL_SAMPLE_LIST = os.path.join(DATA_PATH,'train_test.txt')
TRAIN_VAL_SAMPLE_LIST = os.path.join(DATA_PATH,'train_val_list.txt')

DATA_ENTRY_PATH = os.path.join(DATA_PATH,'Data_Entry_2017.csv')
BBOX_PATH = os.path.join(DATA_PATH,'BBox_List_2017.csv')

# The path to save tensorboard record
TENSORBOARD_LOG_PATH = os.path.join(PROJECT_ROOT,'tmp')

# The pass to backup loss history
HISTORY_BACKUP = os.path.join(PROJECT_ROOT,'historybackup')

# The dir where we put all the scripts for network debuging.
DEBUG_PATH = os.path.join(PROJECT_ROOT,'debug')

# Where we put all .jpg images
IMAGE_BASE_PATH = '/workspace/data'


disease2id = {'No Finding':0,'Atelectasis':1,'Cardiomegaly':2,
	'Effusion':3,'Infiltrate':4,'Mass':5,'Nodule':6,'Pneumonia':7,
	'Pneumothorax':8,'Consolidation':9,'Edema':10,'Emphysema':11,
	'Fibrosis':12,'Pleural_Thickening':13,'Hernia':14,'Infiltration':4}


CLASS_NUM = 2