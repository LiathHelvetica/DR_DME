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
		self.f = transform_f

	def apply(self, fim: FundusImage) -> FundusImage:
		nd_out = self.f(fim.image)
		return FundusImage(nd_out, f"{self.name}{AUGMENTATION_NAME_SEPARATOR}{fim.file_name}")

	def compose(self, t, name) -> Self:
		out_name = f"{name}{AUGMENTATION_NAME_SEPARATOR}{self.name}" if name is not None else self.name
		if isinstance(t, Layer):
			out_f = lambda ndarr: t(self.f(ndarr)).numpy()
		elif isinstance(t, BasicTransform):
			out_f = lambda ndarr: t(image=self.f(ndarr))["image"]
		elif callable(t):
			out_f = lambda ndarr: t(self.f(ndarr))
		elif isinstance(t, FundusTransformation):
			out_f = lambda ndarr: t.f(self.f(ndarr))
			out_name = f"{t.name}{AUGMENTATION_NAME_SEPARATOR}{self.name}"
		else:
			raise Exception("Provided unsupported transform")
		return FundusTransformation(out_f, out_name)
