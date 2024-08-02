import cv2
from numpy import ndarray

from augmentation_util import crop_circle_from_img, get_extract_channel_transform
from augmentation_util.fundus_image import FundusImage
from augmentation_util.fundus_transformation import FundusTransformation

import albumentations as al
import pandas as pd

from constants import COMBINED_LABEL_PATH, IMG_LABEL, COMBINED_IMG_PATH, AUGMENTATION_PLAIN_OUT_PATH


def main(out_size: int, out_path: str = AUGMENTATION_PLAIN_OUT_PATH) -> None:
	OUT_SIZE = out_size
	IN_PATH = COMBINED_LABEL_PATH
	OUT_PATH = f"{out_path}_{out_size}"

	transform = (FundusTransformation()
		.compose(al.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0), name="clahe")
		.compose(crop_circle_from_img)
		.compose(al.ToGray(p=1.0), name="gscale")
		.compose(al.Resize(OUT_SIZE, OUT_SIZE, p=1.0), name=""))

	label_df = pd.read_csv(IN_PATH)
	for id, row in label_df.iterrows():
		f_name = row[IMG_LABEL]
		data: ndarray = cv2.imread(f"{COMBINED_IMG_PATH}/{f_name}")
		data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
		img = FundusImage(data, f_name)
		aug_img = transform.apply(img)
		cv2.imwrite(f"{OUT_PATH}/{aug_img.file_name}", aug_img.image)


if __name__ == "__main__":
	main(224)