from typing import Self

from albumentations import BasicTransform
from keras import Layer

from augmentation_util import identity_transform
from augmentation_util.fundus_image import FundusImage
from constants import AUGMENTATION_NAME_SEPARATOR


# t - Layer (tensorflow) | BasicTransform (album) | f: ndarray -> ndarray | FundusTransformation
class FundusTransformation:

	def __init__(self, transform_f = identity_transform, name: str = ""):
		self.name = name
		if isinstance(transform_f, Layer):
			def out(ndarr):
				o = transform_f(ndarr).numpy()
				return o
			self.f = out
		elif isinstance(transform_f, BasicTransform):
			def out(ndarr):
				o = transform_f(image=ndarr)["image"]
				return o
			self.f = out
		elif callable(transform_f):
			def out(ndarr):
				o = transform_f(ndarr)
				return o
			self.f = out
		elif isinstance(transform_f, FundusTransformation):
			def out(ndarr):
				o = transform_f.f(ndarr)
				return o
			self.f = out
		else:
			raise Exception("Provided unsupported transform")

	def apply(self, fim: FundusImage) -> FundusImage:
		nd_out = self.f(fim.image)
		return FundusImage(nd_out, f"{self.name}{AUGMENTATION_NAME_SEPARATOR}{fim.file_name}")

	def compose(self, t, name=None) -> Self:
		if isinstance(t, FundusTransformation):
			assert name is None
		out_name = f"{name}{AUGMENTATION_NAME_SEPARATOR}{self.name}" if name is not None else self.name
		if isinstance(t, Layer):
			def out(ndarr):
				o = t(self.f(ndarr)).numpy()
				return o
			out_f = out
		elif isinstance(t, BasicTransform):
			def out(ndarr):
				o = t(image=self.f(ndarr))["image"]
				return o
			out_f = out
		elif callable(t):
			def out(ndarr):
				o = t(self.f(ndarr))
				return o
			out_f = out
		elif isinstance(t, FundusTransformation):
			def out(ndarr):
				o = t.f(self.f(ndarr))
				return o
			out_f = out
			out_name = f"{t.name}{AUGMENTATION_NAME_SEPARATOR}{self.name}"
		else:
			raise Exception("Provided unsupported transform")
		return FundusTransformation(out_f, out_name)
