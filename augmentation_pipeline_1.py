from augmentation_util import get_rotation_transform
from augmentation_util.fundus_transformation import FundusTransformation

import albumentations as al
import itertools as it


def main() -> None:
	transforms = [FundusTransformation().compose(al.CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=1.0), name="clahe")]

	# * 3
	tmp = []
	for tr in transforms:
		tmp.append(tr.compose(al.VerticalFlip(p=1), name="xflip"))
		tmp.append(tr.compose(al.HorizontalFlip(p=1), name="yflip"))
	transforms = transforms + tmp

	# * 8
	rotations = map(
		lambda deg: FundusTransformation(get_rotation_transform(deg), name=f"rot{deg}"),
		[45, 90, 135, 180, 225, 260, 305]
	)

	tmp = []
	for rot, tr in it.product(rotations, transforms):
		tmp.append(tr.compose(rot))
	transforms = transforms + tmp




# 1 healthy DME -> ???
if __name__ == "__main__":
	main()