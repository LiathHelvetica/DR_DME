import os
from datetime import datetime
import json

import torch
import torch.nn as nn
import torch.optim as optim
from pandas import read_csv, DataFrame
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.model_selection import train_test_split

from itertools import product

from constants import AUGMENTATION_OUT_PATH, COMBINED_LABEL_PATH, DME_LABEL, ID_LABEL, TEST_SPLIT_VALUE, TRAIN_DATASET_LABEL, TEST_DATASET_LABEL, LEARN_OUT_PATH, LAST_EPISODE_DONE_FILE, BATCH_SIZE, EPOCHS, LOG_EVERY_BATCHES_AMOUNT, MODEL_CHECKPOINT_OUT_PATH, LAST_EPOCH_DONE_FILE, CONFUSION_MATRIX_ORIGINAL_VALUE_LABEL, CONFUSION_MATRIX_PREDICTION_VALUE_LABEL, CONFUSION_MATRIX_COUNT_LABEL, TRAIN_ACC_KEY, TRAIN_LOSS_KEY, TEST_ACC_KEY, TEST_LOSS_KEY, EPOCH_KEY, BATCH_SIZE_KEY, CRITERION_KEY, OPTIMIZER_NAME_KEY, OPTIMIZER_LR_KEY, OPTIMIZER_MOMENTUM_KEY, BASE_MODEL_NAME_KEY, SCHEDULER_NAME_KEY, SCHEDULER_STEP_SIZE_KEY, SCHEDULER_GAMMA_KEY, T_DELTA_KEY, N_SAMPLES_TRAIN_KEY, N_SAMPLES_TEST_KEY, N_CORRECT_TRAIN_KEY, N_CORRECT_TEST_KEY, CONFUSION_MATRIX_TRAIN_KEY, CONFUSION_MATRIX_TEST_KEY, AUGMENTATION_PLAIN_OUT_PATH, PLAIN_DATASET_LABEL, PLAIN_ACC_KEY, PLAIN_LOSS_KEY, N_SAMPLES_PLAIN_KEY, N_CORRECT_PLAIN_KEY, CONFUSION_MATRIX_PLAIN_KEY
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
	epoch_acc_train,
	epoch_loss_train,
	epoch_acc_val,
	epoch_loss_val,
	epoch_acc_plain,
	epoch_loss_plain,
	epoch,
	batch_size,
	criterion,
	optimizer,
	m_name,
	scheduler,
	t_delta,
	len_train_ds,
	len_test_ds,
	len_plain_ds,
	running_corrects_train,
	running_corrects_val,
	running_corrects_plain,
	train_conf_matrix,
	val_conf_matrix,
	plain_conf_matrix
) -> dict:
	return {
		TRAIN_ACC_KEY: epoch_acc_train.item(),
		TRAIN_LOSS_KEY: epoch_loss_train,
		TEST_ACC_KEY: epoch_acc_val.item(),
		TEST_LOSS_KEY: epoch_loss_val,
		PLAIN_ACC_KEY: epoch_acc_plain.item(),
		PLAIN_LOSS_KEY: epoch_loss_plain,
		EPOCH_KEY: epoch,
		BATCH_SIZE_KEY: batch_size,
		CRITERION_KEY: type(criterion).__name__,
		OPTIMIZER_NAME_KEY: type(optimizer).__name__,
		OPTIMIZER_LR_KEY: optimizer.defaults["lr"],
		OPTIMIZER_MOMENTUM_KEY: try_or_else(lambda: optimizer.defaults["momentum"], "-"),
		BASE_MODEL_NAME_KEY: m_name,
		SCHEDULER_NAME_KEY: type(scheduler).__name__,
		SCHEDULER_STEP_SIZE_KEY: try_or_else(lambda: scheduler.step_size, "-"),
		SCHEDULER_GAMMA_KEY: try_or_else(lambda: scheduler.gamma, "-"),
		T_DELTA_KEY: str(t_delta),
		N_SAMPLES_TRAIN_KEY: len_train_ds,
		N_SAMPLES_TEST_KEY: len_test_ds,
		N_SAMPLES_PLAIN_KEY: len_plain_ds,
		N_CORRECT_TRAIN_KEY: running_corrects_train.item(),
		N_CORRECT_TEST_KEY: running_corrects_val.item(),
		N_CORRECT_PLAIN_KEY: running_corrects_plain.item(),
		CONFUSION_MATRIX_TRAIN_KEY: list(train_conf_matrix.values()),
		CONFUSION_MATRIX_TEST_KEY: list(val_conf_matrix.values()),
		CONFUSION_MATRIX_PLAIN_KEY: list(plain_conf_matrix.values())
	}


def model_last_layer_fc(f_model_create, device, n_outputs, x, y, m_name):
	def op():
		model = f_model_create()
		model.fc = nn.Linear(model.fc.in_features, n_outputs)
		model.to(device)
		return model, x, y, m_name
	return op


def inception(f_model_create, device, n_outputs, x, y, m_name):
	def op():
		model = f_model_create()
		model.fc = nn.Linear(model.fc.in_features, n_outputs)
		model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, n_outputs)
		model.to(device)
		return model, x, y, m_name
	return op


def model_last_layer_sequential_classifier(f_model_create, device, n_outputs, x, y, m_name):
	def op():
		model = f_model_create()
		fc = model.classifier[-1]
		model.classifier[-1] = nn.Linear(fc.in_features, n_outputs)
		model.to(device)
		return model, x, y, m_name
	return op


def model_last_layer_sequential_heads(f_model_create, device, n_outputs, x, y, m_name):
	def op():
		model = f_model_create()
		fc = model.heads[-1]
		model.heads[-1] = nn.Linear(fc.in_features, n_outputs)
		model.to(device)
		return model, x, y, m_name
	return op


def model_last_layer_classifier(f_model_create, device, n_outputs, x, y, m_name):
	def op():
		model = f_model_create()
		model.classifier = nn.Linear(model.classifier.in_features, n_outputs)
		model.to(device)
		return model, x, y, m_name
	return op


def model_last_layer_head(f_model_create, device, n_outputs, x, y, m_name):
	def op():
		model = f_model_create()
		model.head = nn.Linear(model.head.in_features, n_outputs)
		model.to(device)
		return model, x, y, m_name
	return op


IN_PATH_PREFIX = AUGMENTATION_OUT_PATH
IN_PLAIN_PATH_PREFIX = AUGMENTATION_PLAIN_OUT_PATH
LABELS_PATH = COMBINED_LABEL_PATH


def get_dataset_for_size(in_size: int, label_df: DataFrame) -> (FundusImageDataset, FundusImageDataset, FundusImageDataset):
	in_path = f"{IN_PATH_PREFIX}_{in_size}"
	img_list = os.listdir(in_path)
	dme_ok = label_df[label_df[DME_LABEL] == 0][ID_LABEL].to_list()
	dme_bad = label_df[label_df[DME_LABEL] == 1][ID_LABEL].to_list()
	dme_ok_id_train, dme_ok_id_test = train_test_split(dme_ok, test_size=TEST_SPLIT_VALUE)
	dme_bad_id_train, dme_bad_id_test = train_test_split(dme_bad, test_size=TEST_SPLIT_VALUE)
	dme_id_train = set(dme_ok_id_train + dme_bad_id_train)
	dme_id_test = set(dme_ok_id_test + dme_bad_id_test)
	train_imgs = list(filter(lambda img: get_id_from_f_name(img) in dme_id_train, img_list))
	test_imgs = list(filter(lambda img: get_id_from_f_name(img) in dme_id_test, img_list))

	plain_in_path = f"{IN_PLAIN_PATH_PREFIX}_{in_size}"
	img_list = os.listdir(plain_in_path)
	plain_imgs = list(filter(lambda img: get_id_from_f_name(img) in dme_id_test, img_list))

	train_ds = FundusImageDataset(
		in_path,
		train_imgs,
		label_df,
		DME_LABEL
	)

	test_ds = FundusImageDataset(
		in_path,
		test_imgs,
		label_df,
		DME_LABEL
	)

	plain_ds = FundusImageDataset(
		plain_in_path,
		plain_imgs,
		label_df,
		DME_LABEL
	)

	return train_ds, test_ds, plain_ds


def get_datasets_dict(in_size: int, label_df: DataFrame) -> dict[str, FundusImageDataset]:
	train_ds, test_ds, plain_ds = get_dataset_for_size(in_size, label_df)
	return {TRAIN_DATASET_LABEL: train_ds, TEST_DATASET_LABEL: test_ds, PLAIN_DATASET_LABEL: plain_ds}


def get_dataloader(in_size: int, label_df: DataFrame) -> dict[str, DataLoader]:
	train_ds, test_ds, plain_ds = get_dataset_for_size(in_size, label_df)
	return {
		TRAIN_DATASET_LABEL: DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
		TEST_DATASET_LABEL: DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False),
		PLAIN_DATASET_LABEL: DataLoader(plain_ds, batch_size=BATCH_SIZE, shuffle=False)
	}


def main() -> None:
	label_df = read_csv(LABELS_PATH)
	sizes = [260, 272, 246, 238, 518, 480, 600, 528, 456, 320, 288, 256, 342, 230, 236, 384, 224, 232]
	dataloaders = {size: get_dataloader(size, label_df) for size in sizes}
	device = "cuda"
	classes = torch.tensor([0, 1])

	model_initializers = [
		model_last_layer_fc(lambda: models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1), device, 1, 224, 224, "resnet50"),
		model_last_layer_fc(lambda: models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2), device, 1, 232, 232, "resnet50"),
		model_last_layer_fc(lambda: models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1), device, 1, 224, 224, "resnet18"),
		model_last_layer_fc(lambda: models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1), device, 1, 224, 224,"resnet34"),
		model_last_layer_fc(lambda: models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1), device, 1, 224,224, "resnet101"),
		model_last_layer_fc(lambda: models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2), device, 1, 232, 232, "resnet101"),
		model_last_layer_fc(lambda: models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1), device, 1, 224, 224, "resnet152"),
		model_last_layer_fc(lambda: models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2), device, 1, 232, 232, "resnet152"),
		model_last_layer_fc(lambda: models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1), device, 1, 224, 224, "googlenet"),
		inception(lambda: models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True), device, 1, 342, 342, "inception_v3"),
		model_last_layer_fc(lambda: models.regnet_y_400mf(weights=models.RegNet_Y_400MF_Weights.IMAGENET1K_V1), device, 1, 224, 224, "regnet_y_400mf"),
		model_last_layer_fc(lambda: models.regnet_y_400mf(weights=models.RegNet_Y_400MF_Weights.IMAGENET1K_V2), device, 1, 232, 232, "regnet_y_400mf"),
		model_last_layer_fc(lambda: models.regnet_y_800mf(weights=models.RegNet_Y_800MF_Weights.IMAGENET1K_V1), device, 1, 224, 224, "regnet_y_800mf"),
		model_last_layer_fc(lambda: models.regnet_y_800mf(weights=models.RegNet_Y_800MF_Weights.IMAGENET1K_V2), device, 1, 232, 232, "regnet_y_800mf"),
		model_last_layer_fc(lambda: models.regnet_y_1_6gf(weights=models.RegNet_Y_1_6GF_Weights.IMAGENET1K_V1), device, 1, 224, 224, "regnet_y_1_6gf"),
		model_last_layer_fc(lambda: models.regnet_y_1_6gf(weights=models.RegNet_Y_1_6GF_Weights.IMAGENET1K_V2), device, 1, 232, 232, "regnet_y_1_6gf"),
		model_last_layer_fc(lambda: models.regnet_y_3_2gf(weights=models.RegNet_Y_3_2GF_Weights.IMAGENET1K_V1), device, 1, 224, 224, "regnet_y_3_2gf"),
		model_last_layer_fc(lambda: models.regnet_y_3_2gf(weights=models.RegNet_Y_3_2GF_Weights.IMAGENET1K_V2), device, 1, 232, 232, "regnet_y_3_2gf"),
		model_last_layer_fc(lambda: models.regnet_y_8gf(weights=models.RegNet_Y_8GF_Weights.IMAGENET1K_V1), device, 1, 224, 224, "regnet_y_8gf"),
		model_last_layer_fc(lambda: models.regnet_y_8gf(weights=models.RegNet_Y_8GF_Weights.IMAGENET1K_V2), device, 1, 232, 232, "regnet_y_8gf"),
		model_last_layer_fc(lambda: models.regnet_y_16gf(weights=models.RegNet_Y_16GF_Weights.IMAGENET1K_V1), device, 1, 224, 224, "regnet_y_16gf"),
		model_last_layer_fc(lambda: models.regnet_y_16gf(weights=models.RegNet_Y_16GF_Weights.IMAGENET1K_V2), device, 1, 232, 232, "regnet_y_16gf"),
		model_last_layer_fc(lambda: models.regnet_y_16gf(weights=models.RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_E2E_V1), device, 1, 384, 384, "regnet_y_16gf"),
		model_last_layer_fc(lambda: models.regnet_y_16gf(weights=models.RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_LINEAR_V1), device, 1, 224, 224, "regnet_y_16gf"),
		model_last_layer_fc(lambda: models.regnet_y_32gf(weights=models.RegNet_Y_32GF_Weights.IMAGENET1K_V1), device, 1, 224, 224, "regnet_y_32gf"),
		model_last_layer_fc(lambda: models.regnet_y_32gf(weights=models.RegNet_Y_32GF_Weights.IMAGENET1K_V2), device, 1, 232, 232, "regnet_y_32gf"),
		model_last_layer_fc(lambda: models.regnet_y_32gf(weights=models.RegNet_Y_32GF_Weights.IMAGENET1K_SWAG_LINEAR_V1), device, 1, 224, 224, "regnet_y_32gf"),
		model_last_layer_fc(lambda: models.regnet_y_32gf(weights=models.RegNet_Y_32GF_Weights.IMAGENET1K_SWAG_E2E_V1), device, 1, 384, 384, "regnet_y_32gf"),
		model_last_layer_fc(lambda: models.regnet_y_128gf(weights=models.RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_E2E_V1), device, 1, 384, 384, "regnet_y_128gf"),
		model_last_layer_fc(lambda: models.regnet_y_128gf(weights=models.RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_LINEAR_V1), device, 1, 224, 224, "regnet_y_128gf"),
		model_last_layer_fc(lambda: models.regnet_x_400mf(weights=models.RegNet_X_400MF_Weights.IMAGENET1K_V1), device, 1, 224, 224, "regnet_x_400mf"),
		model_last_layer_fc(lambda: models.regnet_x_400mf(weights=models.RegNet_X_400MF_Weights.IMAGENET1K_V2), device, 1, 232, 232, "regnet_x_400mf"),
		model_last_layer_fc(lambda: models.regnet_x_800mf(weights=models.RegNet_X_800MF_Weights.IMAGENET1K_V1), device, 1, 224, 224, "regnet_x_800mf"),
		model_last_layer_fc(lambda: models.regnet_x_800mf(weights=models.RegNet_X_800MF_Weights.IMAGENET1K_V2), device, 1, 232, 232, "regnet_x_800mf"),
		model_last_layer_fc(lambda: models.regnet_x_1_6gf(weights=models.RegNet_X_1_6GF_Weights.IMAGENET1K_V1), device, 1, 224, 224, "regnet_x_1_6gf"),
		model_last_layer_fc(lambda: models.regnet_x_1_6gf(weights=models.RegNet_X_1_6GF_Weights.IMAGENET1K_V2), device, 1, 232, 232, "regnet_x_1_6gf"),
		model_last_layer_fc(lambda: models.regnet_x_3_2gf(weights=models.RegNet_X_3_2GF_Weights.IMAGENET1K_V1), device, 1, 224, 224, "regnet_x_3_2gf"),
		model_last_layer_fc(lambda: models.regnet_x_3_2gf(weights=models.RegNet_X_3_2GF_Weights.IMAGENET1K_V2), device, 1, 232, 232, "regnet_x_3_2gf"),
		model_last_layer_fc(lambda: models.regnet_x_8gf(weights=models.RegNet_X_8GF_Weights.IMAGENET1K_V1), device, 1, 224, 224, "regnet_x_8gf"),
		model_last_layer_fc(lambda: models.regnet_x_8gf(weights=models.RegNet_X_8GF_Weights.IMAGENET1K_V2), device, 1, 232, 232, "regnet_x_8gf"),
		model_last_layer_fc(lambda: models.regnet_x_16gf(weights=models.RegNet_X_16GF_Weights.IMAGENET1K_V1), device, 1, 224, 224, "regnet_x_16gf"),
		model_last_layer_fc(lambda: models.regnet_x_16gf(weights=models.RegNet_X_16GF_Weights.IMAGENET1K_V2), device, 1, 232, 232, "regnet_x_16gf"),
		model_last_layer_fc(lambda: models.regnet_x_32gf(weights=models.RegNet_X_32GF_Weights.IMAGENET1K_V1), device, 1, 224, 224, "regnet_x_32gf"),
		model_last_layer_fc(lambda: models.regnet_x_32gf(weights=models.RegNet_X_32GF_Weights.IMAGENET1K_V2), device, 1, 232, 232, "regnet_x_32gf"),
		model_last_layer_fc(lambda: models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1), device, 1, 224, 224, "resnext50_32x4d"),
		model_last_layer_fc(lambda: models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2), device, 1, 232, 232, "resnext50_32x4d"),
		model_last_layer_fc(lambda: models.resnext101_32x8d(weights=models.ResNeXt101_32X8D_Weights.IMAGENET1K_V1), device, 1, 224, 224, "resnext101_32x8d"),
		model_last_layer_fc(lambda: models.resnext101_32x8d(weights=models.ResNeXt101_32X8D_Weights.IMAGENET1K_V2), device, 1, 232, 232, "resnext101_32x8d"),
		model_last_layer_fc(lambda: models.resnext101_64x4d(weights=models.ResNeXt101_64X4D_Weights.IMAGENET1K_V1), device, 1, 232, 232, "resnext101_64x4d"),
		model_last_layer_fc(lambda: models.shufflenet_v2_x0_5(weights=models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1), device, 1, 224, 224, "shufflenet_v2_x0_5"),
		model_last_layer_fc(lambda: models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1), device, 1, 224, 224, "shufflenet_v2_x1_0"),
		model_last_layer_fc(lambda: models.shufflenet_v2_x1_5(weights=models.ShuffleNet_V2_X1_5_Weights.IMAGENET1K_V1), device, 1, 232, 232, "shufflenet_v2_x1_5"),
		model_last_layer_fc(lambda: models.shufflenet_v2_x2_0(weights=models.ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1), device, 1, 232, 232, "shufflenet_v2_x2_0"),
		model_last_layer_fc(lambda: models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1), device, 1, 224, 224, "wide_resnet50_2"),
		model_last_layer_fc(lambda: models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V2), device, 1, 232, 232, "wide_resnet50_2"),
		model_last_layer_fc(lambda: models.wide_resnet101_2(weights=models.Wide_ResNet101_2_Weights.IMAGENET1K_V1), device, 1, 224, 224, "wide_resnet101_2"),
		model_last_layer_fc(lambda: models.wide_resnet101_2(weights=models.Wide_ResNet101_2_Weights.IMAGENET1K_V2), device, 1, 232, 232, "wide_resnet101_2"),
		####
		model_last_layer_sequential_classifier(lambda: models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1), device, 1, 224, 224, "alexnet"),
		model_last_layer_sequential_classifier(lambda: models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1), device, 1, 236, 236, "convnext_tiny"),
		model_last_layer_sequential_classifier(lambda: models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1), device, 1, 230, 230, "convnext_small"),
		model_last_layer_sequential_classifier(lambda: models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1), device, 1, 232, 232, "convnext_base"),
		model_last_layer_sequential_classifier(lambda: models.convnext_large(weights=models.ConvNeXt_Large_Weights.IMAGENET1K_V1), device, 1, 232, 232, "convnext_large"),
		model_last_layer_sequential_classifier(lambda: models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1), device, 1, 256, 256, "efficientnet_b0"),
		model_last_layer_sequential_classifier(lambda: models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1), device, 1, 256, 256, "efficientnet_b1"),
		model_last_layer_sequential_classifier(lambda: models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1), device, 1, 288, 288, "efficientnet_b2"),
		model_last_layer_sequential_classifier(lambda: models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1), device, 1, 320, 320, "efficientnet_b3"),
		model_last_layer_sequential_classifier(lambda: models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1), device, 1, 384, 384, "efficientnet_b4"),
		model_last_layer_sequential_classifier(lambda: models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.IMAGENET1K_V1), device, 1, 456, 456, "efficientnet_b5"),
		model_last_layer_sequential_classifier(lambda: models.efficientnet_b6(weights=models.EfficientNet_B6_Weights.IMAGENET1K_V1), device, 1, 528, 528, "efficientnet_b6"),
		model_last_layer_sequential_classifier(lambda: models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1), device, 1, 600, 600, "efficientnet_b7"),
		model_last_layer_sequential_classifier(lambda: models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1), device, 1, 384, 384, "efficientnet_v2_s"),
		model_last_layer_sequential_classifier(lambda: models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1), device, 1, 480, 480, "efficientnet_v2_m"),
		model_last_layer_sequential_classifier(lambda: models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.IMAGENET1K_V1), device, 1, 480, 480, "efficientnet_v2_l"),
		model_last_layer_sequential_classifier(lambda: models.maxvit_t(weights=models.MaxVit_T_Weights.IMAGENET1K_V1), device, 1, 224, 224, "maxvit_t"),
		model_last_layer_sequential_classifier(lambda: models.mnasnet0_5(weights=models.MNASNet0_5_Weights.IMAGENET1K_V1), device, 1, 224, 224, "mnasnet0_5"),
		model_last_layer_sequential_classifier(lambda: models.mnasnet0_75(weights=models.MNASNet0_75_Weights.IMAGENET1K_V1), device, 1, 232, 232, "mnasnet0_75"),
		model_last_layer_sequential_classifier(lambda: models.mnasnet1_0(weights=models.MNASNet1_0_Weights.IMAGENET1K_V1), device, 1, 224, 224, "mnasnet1_0"),
		model_last_layer_sequential_classifier(lambda: models.mnasnet1_3(weights=models.MNASNet1_3_Weights.IMAGENET1K_V1), device, 1, 232, 232, "mnasnet1_3"),
		model_last_layer_sequential_classifier(lambda: models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1), device, 1, 224, 224, "mobilenet_v2"),
		model_last_layer_sequential_classifier(lambda: models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2), device, 1, 232, 232, "mobilenet_v2"),
		model_last_layer_sequential_classifier(lambda: models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1), device, 1, 224, 224, "mobilenet_v3_small"),
		model_last_layer_sequential_classifier(lambda: models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1), device, 1, 224, 224, "mobilenet_v3_large"),
		model_last_layer_sequential_classifier(lambda: models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2), device, 1, 232, 232, "mobilenet_v3_large"),
		model_last_layer_sequential_classifier(lambda: models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1), device, 1, 224, 224, "vgg11"),
		model_last_layer_sequential_classifier(lambda: models.vgg13(weights=models.VGG13_Weights.IMAGENET1K_V1), device, 1, 224, 224, "vgg13"),
		model_last_layer_sequential_classifier(lambda: models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1), device, 1, 224, 224, "vgg16"),
		model_last_layer_sequential_classifier(lambda: models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1), device, 1, 224, 224, "vgg19"),
		model_last_layer_sequential_classifier(lambda: models.vgg11_bn(weights=models.VGG11_BN_Weights.IMAGENET1K_V1), device, 1, 224, 224, "vgg11_bn"),
		model_last_layer_sequential_classifier(lambda: models.vgg13_bn(weights=models.VGG13_BN_Weights.IMAGENET1K_V1), device, 1, 224, 224, "vgg13_bn"),
		model_last_layer_sequential_classifier(lambda: models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1), device, 1, 224, 224, "vgg16_bn"),
		model_last_layer_sequential_classifier(lambda: models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1), device, 1, 224, 224, "vgg19_bn"),
		####
		model_last_layer_sequential_heads(lambda: models.vit_h_14(weights=models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1), device, 1, 518, 518, "vit_h_14"),
		model_last_layer_sequential_heads(lambda: models.vit_h_14(weights=models.ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1), device, 1, 224, 224, "vit_h_14"),
		model_last_layer_sequential_heads(lambda: models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1), device, 1, 224, 224, "vit_b_16"),
		model_last_layer_sequential_heads(lambda: models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1), device, 1, 224, 224, "vit_b_16"),
		model_last_layer_sequential_heads(lambda: models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1), device, 1, 384, 384, "vit_b_16"),
		model_last_layer_sequential_heads(lambda: models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_V1), device, 1, 242, 242, "vit_l_16"),
		model_last_layer_sequential_heads(lambda: models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1), device, 1, 512, 512, "vit_l_16"),
		model_last_layer_sequential_heads(lambda: models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_SWAG_LINEAR_V1), device, 1, 224, 224, "vit_l_16"),
		model_last_layer_sequential_heads(lambda: models.vit_b_32(weights=models.ViT_B_32_Weights.IMAGENET1K_V1), device, 1, 224, 224, "vit_b_32"),
		model_last_layer_sequential_heads(lambda: models.vit_l_32(weights=models.ViT_L_32_Weights.IMAGENET1K_V1), device, 1, 224, 224, "vit_l_32"),
		####
		model_last_layer_classifier(lambda: models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1), device, 1, 224, 224, "densenet121"),
		model_last_layer_classifier(lambda: models.densenet161(weights=models.DenseNet161_Weights.IMAGENET1K_V1), device, 1, 224, 224, "densenet161"),
		model_last_layer_classifier(lambda: models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1), device, 1, 224, 224, "densenet169"),
		model_last_layer_classifier(lambda: models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1), device, 1, 224, 224, "densenet201"),
		####
		model_last_layer_head(lambda: models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1), device, 1, 238, 238, "swin_b"),
		model_last_layer_head(lambda: models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1), device, 1, 232, 232, "swin_t"),
		model_last_layer_head(lambda: models.swin_s(weights=models.Swin_S_Weights.IMAGENET1K_V1), device, 1, 246, 246, "swin_s"),
		model_last_layer_head(lambda: models.swin_v2_b(weights=models.Swin_V2_B_Weights.IMAGENET1K_V1), device, 1, 272, 272, "swin_v2_b"),
		model_last_layer_head(lambda: models.swin_v2_t(weights=models.Swin_V2_T_Weights.IMAGENET1K_V1), device, 1, 260, 260, "swin_v2_t"),
		model_last_layer_head(lambda: models.swin_v2_s(weights=models.Swin_V2_S_Weights.IMAGENET1K_V1), device, 1, 260, 260, "swin_v2_s")
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

	model_read_from_file = False
	last_episode = -1
	with open(LAST_EPISODE_DONE_FILE, "r") as f_le:
		last_episode = int(f_le.read())

	episodes = list(product(model_initializers, optimizers, schedulers))[last_episode + 1:]

	learn_out_file = f"{LEARN_OUT_PATH}/{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.txt"
	with open(learn_out_file, "w") as f_out:

		for model_f, optim_f, schedul_f in episodes:
			start = datetime.now()
			model, x_size, y_size, m_name = model_f()
			model = model.to(device)

			dataloader_dict = dataloaders[x_size]
			train_dataloader = dataloader_dict[TRAIN_DATASET_LABEL]
			test_dataloader = dataloader_dict[TEST_DATASET_LABEL]
			plain_dataloader = dataloader_dict[PLAIN_DATASET_LABEL]
			train_ds = train_dataloader.dataset
			test_ds = test_dataloader.dataset
			plain_ds = plain_dataloader.dataset

			criterion = nn.BCEWithLogitsLoss()

			optimizer = optim_f(model.parameters())
			scheduler = schedul_f(optimizer, EPOCHS)

			last_epoch = -1
			with open(LAST_EPOCH_DONE_FILE, "r") as f_lep:
				last_epoch = int(f_lep.read())
				last_epoch = -1 if last_epoch >= EPOCHS - 1 else last_epoch
				if last_epoch != -1 and not model_read_from_file:
					model.load_state_dict(torch.load(MODEL_CHECKPOINT_OUT_PATH))
					model_read_from_file = True

			for epoch in list(range(EPOCHS))[last_epoch + 1:]:
				model.train()
				running_corrects_train = 0
				running_loss_train = 0
				epoch_acc_train = -1.0
				epoch_loss_train = -1.0
				n_batches = len(train_dataloader)
				train_conf_matrix: dict[(float, float), dict] = dict()
				i_batch = 1
				print(f"Train - {n_batches} batches")
				for inputs, labels in train_dataloader:
					inputs = inputs.to(device)
					labels = labels.float().to(device)
					optimizer.zero_grad(set_to_none=True)
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
					running_corrects_train += torch.sum(preds == labels.data)
					running_loss_train += loss.item() * inputs.size(0)
					loss.backward()
					optimizer.step()
					scheduler.step()
					if i_batch % LOG_EVERY_BATCHES_AMOUNT == 0:
						print(f"{str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))} > {i_batch} / {n_batches}")
					i_batch = i_batch + 1
				epoch_acc_train = running_corrects_train / len(train_ds)
				epoch_loss_train = running_loss_train / len(train_ds)

				with torch.no_grad():
					model.eval()
					running_loss_val = 0.0
					running_corrects_val = 0
					epoch_acc_val = -1.0
					epoch_loss_val = -1.0
					n_batches = len(test_dataloader)
					val_conf_matrix: dict[(float, float), dict] = dict()
					i_batch = 1
					print(f"Validate - {n_batches} batches")
					for inputs, labels in test_dataloader:
						inputs = inputs.to(device)
						labels = labels.float().to(device)
						# optimizer.zero_grad()
						if m_name == "inception_v3":
							outputs, _ = model(inputs)
							outputs = torch.squeeze(outputs)
							loss = criterion(outputs, labels)
						else:
							outputs = torch.squeeze(model(inputs)) # squeezing is ok in binary classification
							loss = criterion(outputs, labels)
						preds = torch.sigmoid(outputs).round()
						update_confusion_matrix(val_conf_matrix, labels.data, preds)
						running_corrects_val += torch.sum(preds == labels.data)
						running_loss_val += loss.item() * inputs.size(0)
						if i_batch % LOG_EVERY_BATCHES_AMOUNT == 0:
							print(f"{str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))} > {i_batch} / {n_batches}")
						i_batch = i_batch + 1
					epoch_loss_val = running_loss_val / len(test_ds)
					epoch_acc_val = running_corrects_val / len(test_ds)

				with torch.no_grad():
					model.eval()
					running_loss_plain = 0.0
					running_corrects_plain = 0
					epoch_acc_plain = -1.0
					epoch_loss_plain = -1.0
					n_batches = len(plain_dataloader)
					plain_conf_matrix: dict[(float, float), dict] = dict()
					i_batch = 1
					print(f"Plain - {n_batches} batches")
					for inputs, labels in plain_dataloader:
						inputs = inputs.to(device)
						labels = labels.float().to(device)
						# optimizer.zero_grad()
						if m_name == "inception_v3":
							outputs, _ = model(inputs)
							outputs = torch.squeeze(outputs)
							loss = criterion(outputs, labels)
						else:
							outputs = torch.squeeze(model(inputs)) # squeezing is ok in binary classification
							loss = criterion(outputs, labels)
						preds = torch.sigmoid(outputs).round()
						update_confusion_matrix(plain_conf_matrix, labels.data, preds)
						running_corrects_plain += torch.sum(preds == labels.data)
						running_loss_plain += loss.item() * inputs.size(0)
						if i_batch % LOG_EVERY_BATCHES_AMOUNT == 0:
							print(f"{str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))} > {i_batch} / {n_batches}")
						i_batch = i_batch + 1
					epoch_loss_plain = running_loss_plain / len(plain_ds)
					epoch_acc_plain = running_corrects_plain / len(plain_ds)

				torch.save(model.state_dict(), MODEL_CHECKPOINT_OUT_PATH)
				stop = datetime.now()

				model_data = get_model_data(
					epoch_acc_train,
					epoch_loss_train,
					epoch_acc_val,
					epoch_loss_val,
					epoch_acc_plain,
					epoch_loss_plain,
					epoch + 1,
					BATCH_SIZE,
					criterion,
					optimizer,
					m_name,
					scheduler,
					stop - start,
					len(train_ds),
					len(test_ds),
					len(plain_ds),
					running_corrects_train,
					running_corrects_val,
					running_corrects_plain,
					train_conf_matrix,
					val_conf_matrix,
					plain_conf_matrix
				)
				out_data = json.dumps(model_data)
				print(out_data)
				f_out.write(out_data)
				f_out.write("\n")
				f_out.flush()

				with open(LAST_EPOCH_DONE_FILE, "w") as f_lep:
					f_lep.write(str(epoch))

			with open(LAST_EPISODE_DONE_FILE, "w") as f_le:
				last_episode += 1
				f_le.write(str(last_episode))


if __name__ == "__main__":
	main()
