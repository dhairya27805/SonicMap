import pandas as pd

df = pd.read_csv(r"C:\Users\l\Downloads\SonicMap_Test_Dataset_20000_Songs.csv")

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.dtypes)

df.head(100).to_csv("sonicmap_sample.csv", index=False)
