import cv2
import albumentations as al
from numpy import ndarray
import tensorflow as tf


def crop(fi: ndarray) -> ndarray:
	x, y, _ = fi.shape
	min_dim = min(x, y)
	t = al.CenterCrop(min_dim, min_dim)
	return t(image=fi)["image"]


def identity_transform(fi: ndarray) -> ndarray:
	return fi


def to_square(fi: ndarray) -> ndarray:
	size = max(fi.shape[0], fi.shape[1])
	out = tf.image.resize_with_pad(fi, size, size).numpy()
	return out


def get_rotation_transform(deg: int):
	def out(fi: ndarray) -> ndarray:
		(h, w) = fi.shape[:2]
		(cX, cY) = (w // 2, h // 2)
		M = cv2.getRotationMatrix2D((cX, cY), deg, 1.0)
		return cv2.warpAffine(fi, M, (w, h))
	return out
