import cv2
import pandas as pd

from constants import MESSIDOR_LABEL_PATH, MESSIDOR_IMG_PATH, COMBINED_IMG_PATH, DR_LABEL, DME_LABEL, DATASET_LABEL, IMG_LABEL, ID_LABEL, COMBINED_LABEL_PATH, IDRID_LABEL_PATH_1, IDRID_IMG_PATH_1, IDRID_LABEL_PATH_2, IDRID_IMG_PATH_2, ORIG_DME_LABEL
from img_util import img_to_square


def main() -> None:

  out_df_data = []

  messidor_df = pd.read_csv(MESSIDOR_LABEL_PATH)
  n_messidor = len(messidor_df)
  for i, row in messidor_df.iterrows():
    if row["adjudicated_gradable"] != 0:
      f_name = row["image_id"]
      f_name_split = f_name.split(".")
      extension = "png" if f_name_split[1].lower() == "png" else "JPG"
      f_name = f_name_split[0] + "." + extension
      dr_grade = int(row["adjudicated_dr_grade"])
      orig_dme_grade = int(row["adjudicated_dme"])
      dme_grade = 0 if orig_dme_grade == 0 else 1
      dataset_name = "messidor"
      out_name = f"{dataset_name}:{f_name}"
      out_id = out_name.split(".")[0]
      out_img = img_to_square(f"{MESSIDOR_IMG_PATH}/{f_name}")
      out_df_data.append({
        ID_LABEL: out_id,
        IMG_LABEL: out_name,
        DATASET_LABEL: dataset_name,
        DR_LABEL: dr_grade,
        DME_LABEL: dme_grade,
        ORIG_DME_LABEL: orig_dme_grade
      })
      cv2.imwrite(f"{COMBINED_IMG_PATH}/{out_name}", out_img)
      print(f"> {dataset_name} -".ljust(13) + f"{i} / {n_messidor}".rjust(12))

  idrid_1_df = pd.read_csv(IDRID_LABEL_PATH_1)
  n_idrid_1 = len(idrid_1_df)
  for i, row in idrid_1_df.iterrows():
      f_name = f"{row["Image name"]}.jpg"
      dr_grade = int(row["Retinopathy grade"])
      orig_dme_grade = int(row["Risk of macular edema"])
      dme_grade = 0 if orig_dme_grade == 0 else 1
      dataset_name = "idrid:1"
      out_name = f"{dataset_name}:{f_name}"
      out_id = out_name.split(".")[0]
      out_img = img_to_square(f"{IDRID_IMG_PATH_1}/{f_name}")
      out_df_data.append({
        ID_LABEL: out_id,
        IMG_LABEL: out_name,
        DATASET_LABEL: dataset_name,
        DR_LABEL: dr_grade,
        DME_LABEL: dme_grade,
        ORIG_DME_LABEL: orig_dme_grade
      })
      cv2.imwrite(f"{COMBINED_IMG_PATH}/{out_name}", out_img)
      print(f"> {dataset_name} -".ljust(13) + f"{i} / {n_idrid_1}".rjust(12))

  idrid_2_df = pd.read_csv(IDRID_LABEL_PATH_2)
  n_idrid_2 = len(idrid_2_df)
  for i, row in idrid_2_df.iterrows():
      f_name = f"{row["Image name"]}.jpg"
      dr_grade = int(row["Retinopathy grade"])
      orig_dme_grade = int(row["Risk of macular edema"])
      dme_grade = 0 if orig_dme_grade == 0 else 1
      dataset_name = "idrid:2"
      out_name = f"{dataset_name}:{f_name}"
      out_id = out_name.split(".")[0]
      out_img = img_to_square(f"{IDRID_IMG_PATH_2}/{f_name}")
      out_df_data.append({
        ID_LABEL: out_id,
        IMG_LABEL: out_name,
        DATASET_LABEL: dataset_name,
        DR_LABEL: dr_grade,
        DME_LABEL: dme_grade,
        ORIG_DME_LABEL: orig_dme_grade
      })
      cv2.imwrite(f"{COMBINED_IMG_PATH}/{out_name}", out_img)
      print(f"> {dataset_name} -".ljust(13) + f"{i} / {n_idrid_2}".rjust(12))

  out_df = pd.DataFrame(out_df_data)
  out_df.to_csv(COMBINED_LABEL_PATH)

if __name__ == "__main__":
  main()