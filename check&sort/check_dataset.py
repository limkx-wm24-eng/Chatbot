import pandas as pd

df = pd.read_csv("faq_dataset.csv")

print(df.shape)
print(df.columns)
print(df['intent'].value_counts())