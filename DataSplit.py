import pandas as pd

df = pd.read_csv('data\\Collecte.csv', sep=',')

train = df.sample(frac=0.90, random_state=200)
dev = df.drop(train.index)

print(train.shape)
print(dev.shape)

dev.to_csv('data\\Collecte_dev.csv', sep=',', encoding='utf-8', index=False)
train.to_csv('data\\Collecte_train.csv', sep=',', encoding='utf-8', index=False)