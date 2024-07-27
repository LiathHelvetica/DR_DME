import cv2
import albumentations as al
import numpy as np
from nptyping import NDArray, Shape
from numpy import ndarray

from augmentation_util.fundus_transformation import FundusTransformation


def crop(fi: ndarray) -> ndarray:
	x, y, _ = fi.shape
	min_dim = min(x, y)
	t = al.CenterCrop(min_dim, min_dim)
	return t(image=fi)["image"]


def identity_transform(fi: ndarray) -> ndarray:
	return fi


""""
def to_square(fi: ndarray) -> ndarray:
	size = max(fi.shape[0], fi.shape[1])
	out = tf.image.resize_with_pad(fi, size, size).numpy()
	return out
"""


def get_rotation_transform(deg: int):
	def out(fi: ndarray) -> ndarray:
		(h, w) = fi.shape[:2]
		(cX, cY) = (w // 2, h // 2)
		M = cv2.getRotationMatrix2D((cX, cY), deg, 1.0)
		return cv2.warpAffine(fi, M, (w, h))

	return out


def get_tranlation_transform(x, y) -> FundusTransformation:
	return FundusTransformation(
		al.Affine(
			translate_percent={"x": x, "y": y}, cval=0, p=1.0
		), name=f"trx{int(x * 1000)}y{int(y * 1000)}"
	)


def get_random_translation_transform(x, y) -> FundusTransformation:
	return FundusTransformation(
		al.Affine(
			translate_percent={"x": x, "y": y}, cval=0, p=1.0
		), name=f"trx{int(x[0] * 1000)}x{int(x[1] * 1000)}y{int(y[0] * 1000)}y{int(y[1] * 1000)}"
	)


def crop_circle_from_img(img: NDArray[Shape["*, *, 3"], np.uint8]) -> NDArray[Shape["*, *, 3"], np.uint8]:
	(x, y, _) = img.shape
	(x_c, y_c) = (x // 2, y // 2)
	mask = np.zeros(img.shape, dtype=np.uint8)
	cv2.circle(mask, (x_c, y_c), max(x_c, y_c), (999, 999, 999), -1)
	return cv2.min(img, mask)


def get_extract_channel_transform(channel: int):
	def out(img: NDArray[Shape["*, *, 3"], np.uint8]) -> NDArray[Shape["*, *, 1"], np.uint8]:
		return img[:, :, channel]
	return out