import os

import cv2
from numpy import ndarray

from constants import AUGMENTATION_OUT_PATH

PATH = f"{AUGMENTATION_OUT_PATH}_342"


def main():
	mean = 0.0
	std = 0.0
	all_imgs_gen = os.listdir(PATH)
	for f_name in all_imgs_gen:
		data: ndarray = cv2.imread(f"{PATH}/{f_name}", cv2.IMREAD_GRAYSCALE)
		data = data / 255.0
		mean += data.mean()
		std += data.std()

	mean /= len(all_imgs_gen)
	std /= len(all_imgs_gen)

	print(f"Mean: {mean.item()}, Std: {std.item()}")


# 232: Mean: 0.2398845442635873, Std: 0.15427074712467462
# 224: Mean: 0.23989109103293937, Std: 0.15425883483689162
# 236: Mean: 0.2398626524088128, Std: 0.15426536771543234
if __name__ == "__main__":
	main()
