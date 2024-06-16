from pandas import read_csv

from constants import MESSIDOR_LABEL_PATH

df = read_csv(MESSIDOR_LABEL_PATH)
print(df.columns)
print(df)

print((df["adjudicated_dme"] > 0).sum())
print((df["adjudicated_dr_grade"] > 0).sum())