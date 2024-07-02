import os

HOME_PATH = os.path.expanduser("~")
DATA_PATH = f"{HOME_PATH}/DR_DME_DATA"

COMBINED_DATA_PATH = f"{DATA_PATH}/combined"
COMBINED_IMG_PATH = f"{COMBINED_DATA_PATH}/imgs"
COMBINED_LABEL_PATH = f"{COMBINED_DATA_PATH}/labels.csv"
DR_LABEL = "DR-GRADE"
DME_LABEL = "DME-GRADE"
ORIG_DME_LABEL = "ORIG-DME-GRADE"
DATASET_LABEL = "DATASET"
IMG_LABEL = "IMG"
ID_LABEL = "ID"

TEST_DATA_PATH = f"{DATA_PATH}/test"
TEST_OUT_DATA_PATH = f"{DATA_PATH}/test_out"
TEST_IMG = f"IM004343.JPG"
AUGMENTATION_TEST_OUT_DATA_PATH = f"{DATA_PATH}/aug_test_out"

MESSIDOR_2_PATH = f"{DATA_PATH}/MESSIDOR-2"
MESSIDOR_IMG_PATH = f"{MESSIDOR_2_PATH}/IMAGES"
MESSIDOR_LABEL_PATH = f"{MESSIDOR_2_PATH}/messidor_data.csv"

IDRID_PATH = f"{DATA_PATH}/IDRiD"
IDRID_LABEL_PATH_1 = f"{IDRID_PATH}/Groundtruths/IDRiD_Disease Grading_Testing_Labels.csv"
IDRID_LABEL_PATH_2 = f"{IDRID_PATH}/Groundtruths/IDRiD_Disease Grading_Training_Labels.csv"
IDRID_IMG_PATH_1 = f"{IDRID_PATH}/OriginalImages/Test"
IDRID_IMG_PATH_2 = f"{IDRID_PATH}/OriginalImages/Train"

IMG_CROPPER_THRESHOLD = 15

AUGMENTATION_NAME_SEPARATOR = "|"
AUGMENTATION_OUT_PATH = f"{DATA_PATH}/aug_out"
AUGMENTATION_PLAIN_OUT_PATH = f"{DATA_PATH}/aug_out_plain"

TEST_SPLIT_VALUE = 0.15
TRAIN_DATASET_LABEL = "train"
TEST_DATASET_LABEL = "test"
PLAIN_DATASET_LABEL = "plain"
LEARN_OUT_PATH = f"{DATA_PATH}/learn_out"
LAST_EPISODE_DONE_FILE = "last_episode.txt"
LAST_EPOCH_DONE_FILE = "last_epoch.txt"
MODEL_CHECKPOINT_OUT_PATH = "model.torch"
BATCH_SIZE = 32
EPOCHS = 2
LOG_EVERY_BATCHES_AMOUNT = 100

CONFUSION_MATRIX_ORIGINAL_VALUE_LABEL = "orig"
CONFUSION_MATRIX_PREDICTION_VALUE_LABEL = "pred"
CONFUSION_MATRIX_COUNT_LABEL = "count"

TRAIN_ACC_KEY = "trainAcc"
TRAIN_LOSS_KEY = "trainLoss"
TEST_ACC_KEY = "testAcc"
TEST_LOSS_KEY = "testLoss"
PLAIN_ACC_KEY = "plainAcc"
PLAIN_LOSS_KEY = "plainLoss"
EPOCH_KEY = "epoch"
BATCH_SIZE_KEY = "batchSize"
CRITERION_KEY = "criterion"
OPTIMIZER_NAME_KEY = "optimizerName"
OPTIMIZER_LR_KEY = "optimizerLr"
OPTIMIZER_MOMENTUM_KEY = "optimizerMomentum"
BASE_MODEL_NAME_KEY = "baseModelName"
SCHEDULER_NAME_KEY = "schedulerName"
SCHEDULER_STEP_SIZE_KEY = "schedulerStepSize"
SCHEDULER_GAMMA_KEY = "schedulerGamma"
T_DELTA_KEY = "tDelta"
N_SAMPLES_TRAIN_KEY = "nSamplesTrain"
N_SAMPLES_TEST_KEY = "nSamplesTest"
N_SAMPLES_PLAIN_KEY = "nSamplesPlain"
N_CORRECT_TRAIN_KEY = "nCorrectTrain"
N_CORRECT_TEST_KEY = "nCorrectTest"
N_CORRECT_PLAIN_KEY = "nCorrectPlain"
CONFUSION_MATRIX_TRAIN_KEY = "confMatTrain"
CONFUSION_MATRIX_TEST_KEY = "confMatTest"
CONFUSION_MATRIX_PLAIN_KEY = "confMatPlain"
