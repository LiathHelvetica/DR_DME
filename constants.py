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