import pandas as pd
from thesis.connect_sql import connect_sql

df = pd.read_csv("data.csv", sep="\t", header=None, names=["text","label"])
df = df.dropna(subset=['text'])
print(df.head())
conn = connect_sql.Connect()
cursor = conn.cursor()
data_tuples = list(df[['text', 'label']].itertuples(index=False, name=None))
sql = '''
INSERT INTO Tourism.dbo.Training_Set(text, lable) VALUES(?, ?)
'''

cursor.executemany(sql, data_tuples)
cursor.commit()
cursor.close()
conn.close()