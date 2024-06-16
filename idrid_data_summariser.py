from pandas import read_csv

from constants import IDRID_LABEL_PATH_1, IDRID_LABEL_PATH_2

df = read_csv(IDRID_LABEL_PATH_2)
print((df["Retinopathy grade"] > 0).sum())
print((df["Risk of macular edema "] > 0).sum())
