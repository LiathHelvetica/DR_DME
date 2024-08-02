import copy
import os
import random
from datetime import datetime
import json
import statistics as stat
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from pandas import read_csv, DataFrame
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.model_selection import train_test_split, KFold

from itertools import product

from torchvision.io import ImageReadMode

from constants import AUGMENTATION_OUT_PATH, COMBINED_LABEL_PATH, DME_LABEL, ID_LABEL, TEST_SPLIT_VALUE, TRAIN_DATASET_LABEL, TEST_DATASET_LABEL, \
	LEARN_OUT_PATH, LAST_EPISODE_DONE_FILE, BATCH_SIZE, EPOCHS, LOG_EVERY_BATCHES_AMOUNT, MODEL_CHECKPOINT_OUT_PATH, LAST_EPOCH_DONE_FILE, \
	CONFUSION_MATRIX_ORIGINAL_VALUE_LABEL, CONFUSION_MATRIX_PREDICTION_VALUE_LABEL, CONFUSION_MATRIX_COUNT_LABEL, TRAIN_ACC_KEY, TRAIN_LOSS_KEY, TEST_ACC_KEY, \
	TEST_LOSS_KEY, EPOCH_KEY, BATCH_SIZE_KEY, CRITERION_KEY, OPTIMIZER_NAME_KEY, OPTIMIZER_LR_KEY, OPTIMIZER_MOMENTUM_KEY, BASE_MODEL_NAME_KEY, SCHEDULER_NAME_KEY, \
	SCHEDULER_STEP_SIZE_KEY, SCHEDULER_GAMMA_KEY, T_DELTA_KEY, N_SAMPLES_TRAIN_KEY, N_SAMPLES_TEST_KEY, N_CORRECT_TRAIN_KEY, N_CORRECT_TEST_KEY, \
	CONFUSION_MATRIX_TRAIN_KEY, CONFUSION_MATRIX_TEST_KEY, AUGMENTATION_PLAIN_OUT_PATH, PLAIN_DATASET_LABEL, PLAIN_ACC_KEY, PLAIN_LOSS_KEY, N_SAMPLES_PLAIN_KEY, \
	N_CORRECT_PLAIN_KEY, CONFUSION_MATRIX_PLAIN_KEY, FOLDS, VAL_ACC_KEY, VAL_LOSS_KEY, TOTAL_EPOCHS_KEY, \
	TOTAL_FOLDS_KEY, N_SAMPLES_TRAIN_AND_VAL_KEY, TRAIN_ACC_MEAN_KEY, TRAIN_ACC_STD_DEV_KEY, TRAIN_LOSS_MEAN_KEY, TRAIN_LOSS_STD_DEV_KEY, VAL_ACC_MEAN_KEY, \
	VAL_ACC_STD_DEV_KEY, VAL_LOSS_MEAN_KEY, VAL_LOSS_STD_DEV_KEY, TEST_FAILED_IDS_KEY, PLAIN_FAILED_IDS_KEY, LAST_FOLD_DONE_FILE, FOLD_KEY, RANDOM_STATE_FILE
from fundus_dataset import FundusImageDataset
from img_util import get_id_from_f_name
from no_op_scheduler import NoOpScheduler
from util import try_or_else, dict_update_or_default


def default_item_conf_matrix(orig: float, pred: float) -> dict:
	return {
		CONFUSION_MATRIX_ORIGINAL_VALUE_LABEL: orig,
		CONFUSION_MATRIX_PREDICTION_VALUE_LABEL: pred,
		CONFUSION_MATRIX_COUNT_LABEL: 1
	}


def update_confusion_matrix_entry(d: dict) -> dict:
	d[CONFUSION_MATRIX_COUNT_LABEL] = d[CONFUSION_MATRIX_COUNT_LABEL] + 1
	return d


def update_confusion_matrix(d: dict[(float, float), dict], orig_t: Tensor, pred_t: Tensor) -> dict[(float, float), dict]:
	for orig_t_item, pred_t_item in zip(orig_t, pred_t):
		orig = orig_t_item.item()
		pred = pred_t_item.item()
		dict_update_or_default(
			d,
			(orig, pred),
			default_item_conf_matrix(orig, pred),
			update_confusion_matrix_entry
		)
	return d


def get_model_data(
	train_acc,
	train_loss,
	loss_test,
	acc_test,
	failed_ids_test,
	loss_plain,
	acc_plain,
	failed_ids_plain,
	epoch,
	epochs,
	batch_size,
	fold,
	folds,
	criterion,
	optimizer,
	m_name,
	scheduler,
	t_delta,
	test_val_size,
	test_size,
	plain_size,
	test_conf_matrix,
	plain_conf_matrix
) -> dict:
	return {
		TRAIN_ACC_KEY: train_acc,
		TRAIN_LOSS_KEY: train_loss,
		TEST_ACC_KEY: acc_test,
		TEST_LOSS_KEY: loss_test,
		TEST_FAILED_IDS_KEY: list(failed_ids_test),
		PLAIN_ACC_KEY: acc_plain,
		PLAIN_LOSS_KEY: loss_plain,
		PLAIN_FAILED_IDS_KEY: list(failed_ids_plain),
		EPOCH_KEY: epoch,
		TOTAL_EPOCHS_KEY: epochs,
		BATCH_SIZE_KEY: batch_size,
		FOLD_KEY: fold,
		TOTAL_FOLDS_KEY: folds,
		CRITERION_KEY: type(criterion).__name__,
		OPTIMIZER_NAME_KEY: type(optimizer).__name__,
		OPTIMIZER_LR_KEY: optimizer.defaults["lr"],
		OPTIMIZER_MOMENTUM_KEY: try_or_else(lambda: optimizer.defaults["momentum"], None),
		BASE_MODEL_NAME_KEY: m_name,
		SCHEDULER_NAME_KEY: type(scheduler).__name__,
		SCHEDULER_STEP_SIZE_KEY: try_or_else(lambda: scheduler.step_size, None),
		SCHEDULER_GAMMA_KEY: try_or_else(lambda: scheduler.gamma, None),
		T_DELTA_KEY: str(t_delta),
		N_SAMPLES_TRAIN_AND_VAL_KEY: test_val_size,
		N_SAMPLES_TEST_KEY: test_size,
		N_SAMPLES_PLAIN_KEY: plain_size,
		CONFUSION_MATRIX_TEST_KEY: list(test_conf_matrix.values()),
		CONFUSION_MATRIX_PLAIN_KEY: list(plain_conf_matrix.values())
	}


def regnet(f_model_create, device, n_outputs, x, y, m_name):
	def op():
		model = f_model_create()
		model.fc = nn.Linear(model.fc.in_features, n_outputs)
		return model, x, y, m_name
	return op


def resnext(f_model_create, device, n_outputs, x, y, m_name):
	def op():
		model = f_model_create()
		model.fc = nn.Linear(model.fc.in_features, n_outputs)
		return model, x, y, m_name
	return op


def model_last_layer_fc(f_model_create, device, n_outputs, x, y, m_name):
	def op():
		model = f_model_create()
		model.fc = nn.Linear(model.fc.in_features, n_outputs)
		return model, x, y, m_name
	return op


def inception(f_model_create, device, n_outputs, x, y, m_name):
	def op():
		model = f_model_create()
		model.fc = nn.Linear(model.fc.in_features, n_outputs)
		model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, n_outputs)
		return model, x, y, m_name
	return op


def convnext(f_model_create, device, n_outputs, x, y, m_name):
	def op():
		model = f_model_create()
		fc = model.classifier[-1]
		model.classifier[-1] = nn.Linear(fc.in_features, n_outputs)
		return model, x, y, m_name
	return op


def efficientnet(f_model_create, device, n_outputs, x, y, m_name):
	def op():
		model = f_model_create()
		fc = model.classifier[-1]
		model.classifier[-1] = nn.Linear(fc.in_features, n_outputs)
		return model, x, y, m_name
	return op


def model_last_layer_sequential_classifier(f_model_create, device, n_outputs, x, y, m_name):
	def op():
		model = f_model_create()
		fc = model.classifier[-1]
		model.classifier[-1] = nn.Linear(fc.in_features, n_outputs)
		return model, x, y, m_name
	return op


def model_last_layer_sequential_heads(f_model_create, device, n_outputs, x, y, m_name):
	def op():
		model = f_model_create()
		fc = model.heads[-1]
		model.heads[-1] = nn.Linear(fc.in_features, n_outputs)
		return model, x, y, m_name
	return op


def model_last_layer_classifier(f_model_create, device, n_outputs, x, y, m_name):
	def op():
		model = f_model_create()
		model.classifier = nn.Linear(model.classifier.in_features, n_outputs)
		return model, x, y, m_name
	return op


def swin(f_model_create, device, n_outputs, x, y, m_name):
	def op():
		model = f_model_create()
		model.head = nn.Linear(model.head.in_features, n_outputs)
		return model, x, y, m_name
	return op


def model_last_layer_head(f_model_create, device, n_outputs, x, y, m_name):
	def op():
		model = f_model_create()
		model.head = nn.Linear(model.head.in_features, n_outputs)
		return model, x, y, m_name
	return op


IN_PATH_PREFIX = AUGMENTATION_OUT_PATH
IN_PLAIN_PATH_PREFIX = AUGMENTATION_PLAIN_OUT_PATH
LABELS_PATH = COMBINED_LABEL_PATH


def get_split_ids(label_df: DataFrame) -> (list[str], list[str]):
	dme_ok = label_df[label_df[DME_LABEL] == 0][ID_LABEL].to_list()
	dme_bad = label_df[label_df[DME_LABEL] == 1][ID_LABEL].to_list()
	dme_ok_id_train, dme_ok_id_test = train_test_split(dme_ok, test_size=TEST_SPLIT_VALUE)
	dme_bad_id_train, dme_bad_id_test = train_test_split(dme_bad, test_size=TEST_SPLIT_VALUE)
	dme_id_train = set(dme_ok_id_train + dme_bad_id_train)
	dme_id_test = set(dme_ok_id_test + dme_bad_id_test)

	return dme_id_train, dme_id_test


def get_ok_and_bad_ids(label_df: DataFrame) -> (list[str], list[str]):
	dme_ok = label_df[label_df[DME_LABEL] == 0][ID_LABEL].to_list()
	dme_bad = label_df[label_df[DME_LABEL] == 1][ID_LABEL].to_list()
	return dme_ok, dme_bad



def main() -> None:
	label_df = read_csv(LABELS_PATH)
	sizes = [260, 272, 246, 238, 288, 256, 342, 230, 236, 224, 232] # 480, 600, 528, 456, 320, 384
	device = "cuda"

	model_initializers = [
		inception(lambda: models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True), device, 1, 342, 342, "inception_v3"),
		regnet(lambda: models.regnet_y_400mf(weights=models.RegNet_Y_400MF_Weights.IMAGENET1K_V1), device, 1, 224, 224, "regnet_y_400mf_v1"),
		regnet(lambda: models.regnet_y_400mf(weights=models.RegNet_Y_400MF_Weights.IMAGENET1K_V2), device, 1, 232, 232, "regnet_y_400mf_v2"),
		regnet(lambda: models.regnet_y_800mf(weights=models.RegNet_Y_800MF_Weights.IMAGENET1K_V1), device, 1, 224, 224, "regnet_y_800mf_v1"),
		regnet(lambda: models.regnet_y_800mf(weights=models.RegNet_Y_800MF_Weights.IMAGENET1K_V2), device, 1, 232, 232, "regnet_y_800mf_v2"),
		regnet(lambda: models.regnet_y_1_6gf(weights=models.RegNet_Y_1_6GF_Weights.IMAGENET1K_V1), device, 1, 224, 224, "regnet_y_1_6gf_v1"),
		regnet(lambda: models.regnet_y_1_6gf(weights=models.RegNet_Y_1_6GF_Weights.IMAGENET1K_V2), device, 1, 232, 232, "regnet_y_1_6gf_v2"),
		regnet(lambda: models.regnet_y_3_2gf(weights=models.RegNet_Y_3_2GF_Weights.IMAGENET1K_V1), device, 1, 224, 224, "regnet_y_3_2gf_v1"),
		regnet(lambda: models.regnet_y_3_2gf(weights=models.RegNet_Y_3_2GF_Weights.IMAGENET1K_V2), device, 1, 232, 232, "regnet_y_3_2gf_v2"),
		regnet(lambda: models.regnet_y_8gf(weights=models.RegNet_Y_8GF_Weights.IMAGENET1K_V1), device, 1, 224, 224, "regnet_y_8gf_v1"),
		regnet(lambda: models.regnet_y_8gf(weights=models.RegNet_Y_8GF_Weights.IMAGENET1K_V2), device, 1, 232, 232, "regnet_y_8gf_v2"),
		regnet(lambda: models.regnet_x_400mf(weights=models.RegNet_X_400MF_Weights.IMAGENET1K_V1), device, 1, 224, 224, "regnet_x_400mf_v1"),
		regnet(lambda: models.regnet_x_400mf(weights=models.RegNet_X_400MF_Weights.IMAGENET1K_V2), device, 1, 232, 232, "regnet_x_400mf_v2"),
		regnet(lambda: models.regnet_x_800mf(weights=models.RegNet_X_800MF_Weights.IMAGENET1K_V1), device, 1, 224, 224, "regnet_x_800mf_v1"),
		regnet(lambda: models.regnet_x_800mf(weights=models.RegNet_X_800MF_Weights.IMAGENET1K_V2), device, 1, 232, 232, "regnet_x_800mf_v2"),
		regnet(lambda: models.regnet_x_1_6gf(weights=models.RegNet_X_1_6GF_Weights.IMAGENET1K_V1), device, 1, 224, 224, "regnet_x_1_6gf_v1"),
		regnet(lambda: models.regnet_x_1_6gf(weights=models.RegNet_X_1_6GF_Weights.IMAGENET1K_V2), device, 1, 232, 232, "regnet_x_1_6gf_v2"),
		regnet(lambda: models.regnet_x_3_2gf(weights=models.RegNet_X_3_2GF_Weights.IMAGENET1K_V1), device, 1, 224, 224, "regnet_x_3_2gf_v1"),
		regnet(lambda: models.regnet_x_3_2gf(weights=models.RegNet_X_3_2GF_Weights.IMAGENET1K_V2), device, 1, 232, 232, "regnet_x_3_2gf_v2"),
		regnet(lambda: models.regnet_x_8gf(weights=models.RegNet_X_8GF_Weights.IMAGENET1K_V1), device, 1, 224, 224, "regnet_x_8gf_v1"),
		regnet(lambda: models.regnet_x_8gf(weights=models.RegNet_X_8GF_Weights.IMAGENET1K_V2), device, 1, 232, 232, "regnet_x_8gf_v2"),
		resnext(lambda: models.resnext101_64x4d(weights=models.ResNeXt101_64X4D_Weights.IMAGENET1K_V1), device, 1, 232, 232, "resnext101_64x4d_v1"),
		####
		convnext(lambda: models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1), device, 1, 236, 236, "convnext_tiny"),
		convnext(lambda: models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1), device, 1, 230, 230, "convnext_small"),
		convnext(lambda: models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1), device, 1, 232, 232, "convnext_base"),
		efficientnet(lambda: models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1), device, 1, 256, 256, "efficientnet_b0"),
		efficientnet(lambda: models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1), device, 1, 256, 256, "efficientnet_b1"),
		efficientnet(lambda: models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1), device, 1, 288, 288, "efficientnet_b2"),
		####
		####
		####
		swin(lambda: models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1), device, 1, 238, 238, "swin_b"),
		swin(lambda: models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1), device, 1, 232, 232, "swin_t"),
		swin(lambda: models.swin_s(weights=models.Swin_S_Weights.IMAGENET1K_V1), device, 1, 246, 246, "swin_s"),
		swin(lambda: models.swin_v2_b(weights=models.Swin_V2_B_Weights.IMAGENET1K_V1), device, 1, 272, 272, "swin_v2_b"),
		swin(lambda: models.swin_v2_t(weights=models.Swin_V2_T_Weights.IMAGENET1K_V1), device, 1, 260, 260, "swin_v2_t"),
		swin(lambda: models.swin_v2_s(weights=models.Swin_V2_S_Weights.IMAGENET1K_V1), device, 1, 260, 260, "swin_v2_s"),
		####
		# model_last_layer_fc(lambda: models.regnet_x_16gf(weights=models.RegNet_X_16GF_Weights.IMAGENET1K_V1), device, 1, 224, 224, "regnet_x_16gf_v1"),
		# model_last_layer_fc(lambda: models.regnet_x_16gf(weights=models.RegNet_X_16GF_Weights.IMAGENET1K_V2), device, 1, 232, 232, "regnet_x_16gf_v2"),
		# model_last_layer_fc(lambda: models.regnet_x_32gf(weights=models.RegNet_X_32GF_Weights.IMAGENET1K_V1), device, 1, 224, 224, "regnet_x_32gf_v1"),
		# model_last_layer_fc(lambda: models.regnet_x_32gf(weights=models.RegNet_X_32GF_Weights.IMAGENET1K_V2), device, 1, 232, 232, "regnet_x_32gf_v2"),
		# model_last_layer_fc(lambda: models.regnet_y_16gf(weights=models.RegNet_Y_16GF_Weights.IMAGENET1K_V1), device, 1, 224, 224, "regnet_y_16gf_v1"),
		# model_last_layer_fc(lambda: models.regnet_y_16gf(weights=models.RegNet_Y_16GF_Weights.IMAGENET1K_V2), device, 1, 232, 232, "regnet_y_16gf_v2"),
		# model_last_layer_fc(lambda: models.regnet_y_16gf(weights=models.RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_E2E_V1), device, 1, 384, 384, "regnet_y_16gf_e2e"),
		# model_last_layer_fc(lambda: models.regnet_y_16gf(weights=models.RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_LINEAR_V1), device, 1, 224, 224, "regnet_y_16gf_linear"),
		# model_last_layer_fc(lambda: models.regnet_y_32gf(weights=models.RegNet_Y_32GF_Weights.IMAGENET1K_V1), device, 1, 224, 224, "regnet_y_32gf_v1"),
		# model_last_layer_fc(lambda: models.regnet_y_32gf(weights=models.RegNet_Y_32GF_Weights.IMAGENET1K_V2), device, 1, 232, 232, "regnet_y_32gf_v2"),
		# model_last_layer_fc(lambda: models.regnet_y_32gf(weights=models.RegNet_Y_32GF_Weights.IMAGENET1K_SWAG_LINEAR_V1), device, 1, 224, 224, "regnet_y_32gf"),
		# model_last_layer_fc(lambda: models.regnet_y_32gf(weights=models.RegNet_Y_32GF_Weights.IMAGENET1K_SWAG_E2E_V1), device, 1, 384, 384, "regnet_y_32gf"),
		# model_last_layer_fc(lambda: models.regnet_y_128gf(weights=models.RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_E2E_V1), device, 1, 384, 384, "regnet_y_128gf"),
		# model_last_layer_fc(lambda: models.regnet_y_128gf(weights=models.RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_LINEAR_V1), device, 1, 224, 224, "regnet_y_128gf"),
		# model_last_layer_sequential_classifier(lambda: models.convnext_large(weights=models.ConvNeXt_Large_Weights.IMAGENET1K_V1), device, 1, 232, 232, "convnext_large"),
		# model_last_layer_sequential_classifier(lambda: models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1), device, 1, 320, 320, "efficientnet_b3"),
		# model_last_layer_sequential_classifier(lambda: models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1), device, 1, 384, 384, "efficientnet_b4"),
		# model_last_layer_sequential_classifier(lambda: models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1), device, 1, 384, 384, "efficientnet_v2_s"),
		# model_last_layer_sequential_classifier(lambda: models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1), device, 1, 480, 480, "efficientnet_v2_m"),
		# model_last_layer_sequential_classifier(lambda: models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.IMAGENET1K_V1), device, 1, 480, 480, "efficientnet_v2_l"),
		# model_last_layer_sequential_classifier(lambda: models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.IMAGENET1K_V1), device, 1, 456, 456, "efficientnet_b5"),
		# model_last_layer_sequential_classifier(lambda: models.efficientnet_b6(weights=models.EfficientNet_B6_Weights.IMAGENET1K_V1), device, 1, 528, 528, "efficientnet_b6"),
		# model_last_layer_sequential_classifier(lambda: models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1), device, 1, 600, 600, "efficientnet_b7"),
	]
	optimizers = [
		lambda params: optim.SGD(params, lr=0.001, momentum=0.9),
		# lambda params: optim.SGD(params, lr=0.01, momentum=0.9),
		lambda params: optim.Adam(params, lr=0.001),
		# lambda params: optim.Adam(params, lr=0.01),
		lambda params: optim.Adagrad(params, lr=0.01),
		# lambda params: optim.Adagrad(params, lr=0.1),
		lambda params: optim.RMSprop(params, lr=0.001)
		# lambda params: optim.RMSprop(params, lr=0.1),
		# lambda params: optim.RMSprop(params, lr=0.01, momentum=0.1),
		##### lambda params: optim.Adadelta(params)
	]
	schedulers = [
		lambda opt, n_epochs: NoOpScheduler(),
		# lambda opt, n_epochs: lr_scheduler.StepLR(optimizer=opt, step_size=int(n_epochs / 6), gamma=0.9),
		# lambda opt, n_epochs: lr_scheduler.StepLR(optimizer=opt, step_size=int(n_epochs / 3), gamma=0.1),
		# lambda opt, n_epochs: lr_scheduler.StepLR(optimizer=opt, step_size=int(n_epochs / 4), gamma=0.1),
		# lambda opt, n_epochs: lr_scheduler.StepLR(optimizer=opt, step_size=int(n_epochs / 4), gamma=0.5),
		# lambda opt, n_epochs: lr_scheduler.StepLR(optimizer=opt, step_size=int(n_epochs / 6), gamma=0.1),
		# lambda opt, n_epochs: ReduceLROnPlateau(optimizer=opt)
	]

	criterion = nn.BCEWithLogitsLoss()

	model_read_from_file = False
	last_episode = -1
	with open(LAST_EPISODE_DONE_FILE, "r") as f_le:
		last_episode = int(f_le.read())

	episodes = list(product(model_initializers, optimizers, schedulers))[last_episode + 1:]

	learn_out_file = f"{LEARN_OUT_PATH}/{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.txt"
	with open(learn_out_file, "w") as f_out:

		for model_f, optim_f, schedul_f in episodes:

			last_fold = -1
			with open(LAST_FOLD_DONE_FILE, "r") as f_lf:
				last_fold = int(f_lf.read())
				last_fold = -1 if last_fold >= FOLDS - 1 else last_fold

			random_state = -1
			with open(RANDOM_STATE_FILE, "r") as ran_f:
				random_state = int(ran_f.read())

			_, x_size, y_size, _ = model_f()

			in_path = f"{IN_PATH_PREFIX}_{x_size}"
			img_list = os.listdir(in_path)
			plain_in_path = f"{IN_PLAIN_PATH_PREFIX}_{x_size}"
			plain_img_list = os.listdir(plain_in_path)
			dme_ok_ids, dme_bad_ids = get_ok_and_bad_ids(label_df)
			folder = KFold(n_splits=FOLDS, shuffle=True, random_state=random_state)
			dme_ok_folds = list(folder.split(dme_ok_ids))[last_fold + 1:]
			dme_bad_folds = list(folder.split(dme_bad_ids))[last_fold + 1:]

			for fold, ((dme_ok_train_indices, dme_ok_test_indices), (dme_bad_train_indices, dme_bad_test_indices)) in enumerate(zip(dme_ok_folds, dme_bad_folds)):
				train_ids = set([dme_ok_ids[i] for i in dme_ok_train_indices] + [dme_bad_ids[i] for i in dme_bad_train_indices])
				test_ids = set([dme_ok_ids[i] for i in dme_ok_test_indices] + [dme_bad_ids[i] for i in dme_bad_test_indices])
				train_imgs = list(filter(lambda img: get_id_from_f_name(img) in train_ids, img_list))
				test_imgs = list(filter(lambda img: get_id_from_f_name(img) in test_ids, img_list))
				plain_imgs = list(filter(lambda img: get_id_from_f_name(img) in test_ids, plain_img_list))
				train_ds = FundusImageDataset(
					in_path,
					train_imgs,
					label_df,
					DME_LABEL,
					is_grayscale=True
				)
				train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
				test_ds = FundusImageDataset(
					in_path,
					test_imgs,
					label_df,
					DME_LABEL,
					is_grayscale=True
				)
				test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)
				plain_ds = FundusImageDataset(
					plain_in_path,
					plain_imgs,
					label_df,
					DME_LABEL,
					is_grayscale=True
				)
				plain_dl = DataLoader(plain_ds, batch_size=BATCH_SIZE, shuffle=True)

				start = datetime.now()
				model, x_size, y_size, m_name = model_f()
				print(f"Using: {m_name}")
				model = model.to(device)

				optimizer = optim_f(model.parameters()) # b2 in previous version likely didn't work because I didn't update these, just the model ref
				scheduler = schedul_f(optimizer, EPOCHS)

				for epoch in list(range(EPOCHS)):

					model.train()
					running_corrects_train = 0
					running_loss_train = 0.0
					acc_train = -1.0
					loss_train = -1.0
					n_batches = len(train_dl)
					train_conf_matrix: dict[(float, float), dict] = dict()
					i_batch = 1
					print(f"Train - {n_batches} batches, fold {fold + 1}")
					for inputs, labels, _ in train_dl:
						inputs = inputs.to(device)
						labels = labels.float().to(device)
						optimizer.zero_grad(set_to_none=True) # TODO: technically this doesn't matter but grad behaviour _might_ be different, try with False
						if m_name == "inception_v3":
							outputs, aux_outputs = model(inputs)
							outputs = torch.squeeze(outputs)
							aux_outputs = torch.squeeze(aux_outputs)
							loss = criterion(outputs, labels) + criterion(aux_outputs, labels) * 0.3
						else:
							outputs = torch.squeeze(model(inputs)) # squeezing is ok in binary classification
							loss = criterion(outputs, labels)
						preds = torch.sigmoid(outputs).round()
						update_confusion_matrix(train_conf_matrix, labels.data, preds)
						running_corrects_train += torch.sum(preds == labels.data).item()
						running_loss_train += loss.item() * inputs.size(0)
						loss.backward()
						optimizer.step()
						scheduler.step()
						if i_batch % LOG_EVERY_BATCHES_AMOUNT == 0:
							print(f"{str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))} > {i_batch} / {n_batches}, {fold + 1} / {FOLDS}")
						i_batch = i_batch + 1
					acc_train = running_corrects_train / len(train_ds)
					loss_train = running_loss_train / len(train_ds)

					with torch.no_grad():
						model.eval()
						running_corrects_test = 0
						running_loss_test = 0.0
						acc_test = -1.0
						loss_test = -1.0
						n_batches = len(test_dl)
						test_conf_matrix: dict[(float, float), dict] = dict()
						failed_test_ids = set()
						i_batch = 1
						print(f"Test - {n_batches} batches, fold {fold + 1}")
						for inputs, labels, ids in test_dl:
							inputs = inputs.to(device)
							labels = labels.float().to(device)
							# optimizer.zero_grad()
							outputs = torch.squeeze(model(inputs)) # squeezing is ok in binary classification
							loss = criterion(outputs, labels)
							preds = torch.sigmoid(outputs).round()
							update_confusion_matrix(test_conf_matrix, labels.data, preds)
							correct_indices = preds == labels.data
							incorrect_ids = [id for id, pred in zip(list(ids), correct_indices.tolist()) if not pred]
							failed_test_ids.update(incorrect_ids)
							running_corrects_test += torch.sum(correct_indices).item()
							running_loss_test += loss.item() * inputs.size(0)
							if i_batch % LOG_EVERY_BATCHES_AMOUNT == 0:
								print(f"{str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))} > {i_batch} / {n_batches}, {fold + 1} / {FOLDS}")
							i_batch = i_batch + 1
						loss_test = running_loss_test / len(test_ds)
						acc_test = running_corrects_test / len(test_ds)

					with torch.no_grad():
						model.eval()
						running_corrects_plain = 0
						running_loss_plain = 0.0
						acc_plain = -1.0
						loss_plain = -1.0
						n_batches = len(plain_dl)
						plain_conf_matrix: dict[(float, float), dict] = dict()
						failed_plain_ids = set()
						i_batch = 1
						print(f"Plain - {n_batches} batches, fold {fold + 1}")
						for inputs, labels, ids in plain_dl:
							inputs = inputs.to(device)
							labels = labels.float().to(device)
							# optimizer.zero_grad()
							outputs = torch.squeeze(model(inputs)) # squeezing is ok in binary classification
							loss = criterion(outputs, labels)
							preds = torch.sigmoid(outputs).round()
							update_confusion_matrix(plain_conf_matrix, labels.data, preds)
							correct_indices = preds == labels.data
							incorrect_ids = [id for id, pred in zip(list(ids), correct_indices.tolist()) if not pred]
							failed_plain_ids.update(incorrect_ids)
							running_corrects_plain += torch.sum(correct_indices).item()
							running_loss_plain += loss.item() * inputs.size(0)
							if i_batch % LOG_EVERY_BATCHES_AMOUNT == 0:
								print(f"{str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))} > {i_batch} / {n_batches}, {fold + 1} / {FOLDS}")
							i_batch = i_batch + 1
						loss_plain = running_loss_plain / len(plain_ds)
						acc_plain = running_corrects_plain / len(plain_ds)

					stop = datetime.now()

					model_data = get_model_data(
						acc_train,
						loss_train,
						loss_test,
						acc_test,
						failed_test_ids,
						loss_plain,
						acc_plain,
						failed_plain_ids,
						epoch + 1,
						EPOCHS,
						BATCH_SIZE,
						fold + 1,
						FOLDS,
						criterion,
						optimizer,
						m_name,
						scheduler,
						stop - start,
						len(train_ds),
						len(test_ds),
						len(plain_ds),
						test_conf_matrix,
						plain_conf_matrix
					)
					out_data = json.dumps(model_data)
					print(out_data)
					f_out.write(out_data)
					f_out.write("\n")
					f_out.flush()

				with open(LAST_FOLD_DONE_FILE, "w") as f_lf:
					last_fold += 1
					f_lf.write(str(last_fold))

			with open(LAST_EPISODE_DONE_FILE, "w") as f_le:
				last_episode += 1
				f_le.write(str(last_episode))

			with open(RANDOM_STATE_FILE, "w") as ran_f:
				ran_f.write(str(random.randint(0, 9999999)))


if __name__ == "__main__":
	main()
