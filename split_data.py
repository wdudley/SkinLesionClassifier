import pandas as pd

df = pd.read_csv('labels_ds3.csv', usecols=["Image", "Class"])
print(df.columns[0])

df0, df1, df2, df3 = [], [], [], []

for i in range(len(df)):
  temp = df.iloc[i,1]
  
  if temp == 0:
    df0.append((df.iloc[i,0], temp))
  if temp == 1:
    df1.append((df.iloc[i,0], temp))
  if temp == 2:
    df2.append((df.iloc[i,0], temp))
  if temp == 3:
    df3.append((df.iloc[i,0], temp))

print(len(df0), len(df1), len(df2), len(df3))