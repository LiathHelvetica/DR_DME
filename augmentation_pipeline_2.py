import cv2
from numpy import ndarray

from augmentation_util import get_rotation_transform, get_random_translation_transform, crop_circle_from_img, get_extract_channel_transform
from augmentation_util.fundus_image import FundusImage
from augmentation_util.fundus_transformation import FundusTransformation
from random import shuffle, sample

import albumentations as al
import itertools as it
import pandas as pd

from constants import COMBINED_LABEL_PATH, IMG_LABEL, COMBINED_IMG_PATH, DME_LABEL, AUGMENTATION_OUT_PATH


def main(out_size: int, out_path: str = AUGMENTATION_OUT_PATH) -> None:
	OUT_SIZE = out_size
	IN_PATH = COMBINED_LABEL_PATH
	OUT_PATH = f"{out_path}_{out_size}"
	HEALTHY_TARGET_PER_IMG = 32
	UNHEALTHY_TARGET_PER_IMG = 128

	safe_transforms = [
		FundusTransformation()
		.compose(al.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0), name="clahe")
		.compose(crop_circle_from_img)
	]

	# * 3
	tmp = []
	for tr in safe_transforms:
		tmp.append(tr.compose(al.VerticalFlip(p=1), name="xflip"))
		tmp.append(tr.compose(al.HorizontalFlip(p=1), name="yflip"))
	safe_transforms = safe_transforms + tmp

	# * 8
	rotations = map(
		lambda deg: FundusTransformation(get_rotation_transform(deg), name=f"rot{deg}"),
		[45, 90, 135, 180, 225, 260, 305]
	)

	tmp = []
	for rot, tr in it.product(rotations, safe_transforms):
		tmp.append(tr.compose(rot))
	safe_transforms = safe_transforms + tmp

	# * 25
	translate_unzoomed_transforms = []
	translations = filter(lambda v: v is not None, map(
		lambda tpl: get_random_translation_transform(tpl[0], tpl[1]),
		it.product([(-0.09, -0.057), (-0.057, -0.025), (-0.025, 0.025), (0.025, 0.057), (0.057, 0.09)], repeat=2)
	))

	tmp = []
	for trans, tr in it.product(translations, safe_transforms):
		tmp.append(tr.compose(trans))
	translate_unzoomed_transforms = translate_unzoomed_transforms + tmp

	# * 2
	zoomed_out_transforms = []
	tmp = []
	for tr in safe_transforms:
		tmp.append(tr.compose(FundusTransformation(
			al.Affine(scale=(0.94, 0.90), cval=0, p=1.0),
			name="zoomout"
		)))
	zoomed_out_transforms = zoomed_out_transforms + tmp

	# * 49
	zoomout_translations = filter(lambda v: v is not None, map(
		lambda tpl: get_random_translation_transform(tpl[0], tpl[1]),
		it.product([(-0.123, -0.09), (-0.09, -0.057), (-0.057, -0.025), (-0.025, 0.025), (0.025, 0.057), (0.057, 0.09), (0.09, 0.123)], repeat=2)
	))

	tmp = []
	for trans, tr in it.product(zoomout_translations, zoomed_out_transforms):
		tmp.append(tr.compose(trans))
	zoomed_out_transforms = zoomed_out_transforms + tmp


	# * 9
	zoomed_in_transforms = []
	zoomin_translations = filter(lambda v: v is not None, map(
		lambda tpl: get_random_translation_transform(tpl[0], tpl[1]),
		it.product([(-0.057, -0.025), (-0.025, 0.025), (0.025, 0.057)], repeat=2)
	))

	tmp = []
	for trans, tr in it.product(zoomin_translations, safe_transforms):
		tmp.append(tr.compose(trans))
	zoomed_in_transforms = zoomed_in_transforms + tmp

	# * 2
	for i, tr in enumerate(zoomed_in_transforms):
		zoomed_in_transforms[i] = tr.compose(FundusTransformation(
			al.Affine(scale=(1.1, 1.04), cval=0, p=1.0),
			name="zoomin"
		))

	###

	# compose only with selected transforms
	to_square_and_resize_transform: FundusTransformation = FundusTransformation(
		get_extract_channel_transform(1),
		name="exGreen"
	).compose(FundusTransformation(
		al.Resize(OUT_SIZE, OUT_SIZE, p=1.0),
		name=f"res{OUT_SIZE}"
	))

	label_df = pd.read_csv(IN_PATH)
	for id, row in label_df.iterrows():
		f_name = row[IMG_LABEL]
		dme_score = row[DME_LABEL]
		target = HEALTHY_TARGET_PER_IMG if dme_score == 0 else UNHEALTHY_TARGET_PER_IMG
		data: ndarray = cv2.imread(f"{COMBINED_IMG_PATH}/{f_name}")
		data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
		img = FundusImage(data, f_name)
		unsafe_target = target - len(safe_transforms)
		tr_unzoom_transforms = sample(translate_unzoomed_transforms, unsafe_target - 2 * (unsafe_target // 3))
		tr_zoomin_transforms = sample(zoomed_in_transforms, unsafe_target // 3)
		tr_zoomout_transforms = sample(zoomed_out_transforms, unsafe_target // 3)
		transforms = safe_transforms + tr_unzoom_transforms + tr_zoomin_transforms + tr_zoomout_transforms
		for tr in sample(transforms, target):
			tr = tr.compose(to_square_and_resize_transform, name=None)
			aug_img = tr.apply(img)
			cv2.imwrite(f"{OUT_PATH}/{aug_img.file_name}", cv2.cvtColor(aug_img.image, cv2.COLOR_RGB2BGR))
