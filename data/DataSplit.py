import pandas as pd

df = pd.read_csv('Collecte.csv', sep=',')

train = df.sample(frac=0.90, random_state=200)
dev = df.drop(train.index)

print(train.shape)
print(dev.shape)

dev.to_csv('Collecte_dev.csv', sep=',', encoding='utf-8', index=False)
train.to_csv('Collecte_train.csv', sep=',', encoding='utf-8', index=False)