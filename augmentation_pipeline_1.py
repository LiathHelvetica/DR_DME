from copy import copy

import cv2
from numpy import ndarray

from augmentation_util import get_rotation_transform
from augmentation_util.fundus_image import FundusImage
from augmentation_util.fundus_transformation import FundusTransformation

import albumentations as al
import itertools as it
import pandas as pd

from constants import COMBINED_LABEL_PATH, IMG_LABEL, COMBINED_IMG_PATH, AUGMENTATION_TEST_OUT_DATA_PATH

OUT_SIZE = 224

# TODO: apply mask and look


def main() -> None:
	transforms = [FundusTransformation().compose(al.CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=1.0), name="clahe")]

	# * 3
	"""
	tmp = []
	for tr in transforms:
		tmp.append(tr.compose(al.VerticalFlip(p=1), name="xflip"))
		tmp.append(tr.compose(al.HorizontalFlip(p=1), name="yflip"))
	transforms = transforms + tmp
	"""

	# * 8
	"""
	rotations = map(
		lambda deg: FundusTransformation(get_rotation_transform(deg), name=f"rot{deg}"),
		[45, 90, 135, 180, 225, 260, 305]
	)

	tmp = []
	for rot, tr in it.product(rotations, transforms):
		tmp.append(tr.compose(rot))
	transforms = transforms + tmp
	"""

	# * ???
	def get_tranlation_transform(x, y) -> FundusTransformation | None:
		return None if x == 0.0 and y == 0.0 else FundusTransformation(
			al.Affine(
				translate_percent=(x, y), cval=0, p=1.0
			), name=f"trx{int(x * 1000)}y{int(y * 1000)}"
		)

	translate_unzoomed_transforms = copy(transforms)
	translations = filter(lambda v: v is not None, map(
		lambda tpl: get_tranlation_transform(tpl[0], tpl[1]),
		it.product([-0.045, -0.025, 0.0, 0.025, 0.045], repeat=2)
	))

	tmp = []
	for trans, tr in it.product(translations, translate_unzoomed_transforms):
		tmp.append(tr.compose(trans))
	transforms = transforms + tmp # TODO: remove

	###

	# transforms = list(transforms)

	# compose only with selected transforms
	"""
	to_square_and_resize_transform: FundusTransformation = FundusTransformation(
		al.Resize(OUT_SIZE, OUT_SIZE, p=1.0),
		name=f"res{OUT_SIZE}"
	)
	"""

	# tr = tr.compose(to_square_and_resize_transform, name=None)

	label_df = pd.read_csv(COMBINED_LABEL_PATH)
	for id, row in label_df.iterrows():
		f_name = row[IMG_LABEL]
		data: ndarray = cv2.imread(f"{COMBINED_IMG_PATH}/{f_name}")
		data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
		img = FundusImage(data, f_name)
		for tr in transforms:
			# tr = tr.compose(to_square_and_resize_transform, name=f"res{OUT_SIZE}")
			aug_img = tr.apply(img)
			cv2.imwrite(f"{AUGMENTATION_TEST_OUT_DATA_PATH}/{aug_img.file_name}", cv2.cvtColor(aug_img.image, cv2.COLOR_RGB2BGR))
			print("----")
		exit()




# 1 healthy DME -> ???
if __name__ == "__main__":
	main()