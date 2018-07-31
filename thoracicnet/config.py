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