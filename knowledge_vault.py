from pandas import read_csv

from constants import COMBINED_LABEL_PATH, DR_LABEL, DME_LABEL


def main() -> None:
	combined_dataset_df = read_csv(COMBINED_LABEL_PATH)
	print(f"Number of images: {len(combined_dataset_df)}")
	print(f"Number of DR healthy: {(combined_dataset_df[DR_LABEL] == 0).sum()}")
	print(f"Number of DR disease: {(combined_dataset_df[DR_LABEL] != 0).sum()}")
	print(f"Number of DR 1: {(combined_dataset_df[DR_LABEL] == 1).sum()}")
	print(f"Number of DR 2: {(combined_dataset_df[DR_LABEL] == 2).sum()}")
	print(f"Number of DR 3: {(combined_dataset_df[DR_LABEL] == 3).sum()}")
	print(f"Number of DR 4: {(combined_dataset_df[DR_LABEL] == 4).sum()}")
	print(f"Number of DME healthy: {(combined_dataset_df[DME_LABEL] == 0).sum()}")
	print(f"Number of DME disease: {(combined_dataset_df[DME_LABEL] != 0).sum()}")


if __name__ == "__main__":
	main()