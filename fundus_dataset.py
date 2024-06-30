from pandas import read_csv, DataFrame
from torch.utils.data import Dataset
from torchvision.io import read_image
from torch import Tensor
import torchvision.transforms.functional as tft

from constants import ID_LABEL
from img_util import get_id_from_f_name
from util import dict_update_or_default


class FundusImageDataset(Dataset):

	def __init__(
		self,
		img_path: str,
		img_list: list[str],
		label_df: DataFrame,
		score_column: str
	):
		self.img_path: str = img_path
		self.img_list: list[str] = img_list
		self.label_df = label_df
		self.score_column: str = score_column
		self.unique_score_count: int = self.label_df[score_column].nunique()

	def count_class_representation(self) -> dict[str, int]:
		acc: dict[str, int] = dict()
		for f_name in self.img_list:
			id = get_id_from_f_name(f_name)
			v = self.get_label_by_id(id)
			dict_update_or_default(acc, v, 1, lambda v: v + 1)
		return acc

	def get_label_by_id(self, id: str) -> int:
		return self.label_df[self.label_df[ID_LABEL] == id][self.score_column].values[0]

	def get_labels(self) -> list[int]:
		return list(range(self.unique_score_count))

	def get_all_int_labels(self) -> list[int]:
		def iter(f_name):
			id = get_id_from_f_name(f_name)
			return self.get_label_by_id(id)
		return list(map(lambda f_name: iter(f_name), self.img_list))

	def __len__(self) -> int:
		return len(self.img_list)

	def __getitem__(self, index) -> (Tensor, int):
		f_name = self.img_list[index]
		id = get_id_from_f_name(f_name)
		data = read_image(f"{self.img_path}/{f_name}")
		data = data / 255.0
		tft.normalize(data, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True)
		return data, self.get_label_by_id(id)
