import os

from pandas import read_csv
from sklearn.linear_model import LogisticRegression
from data_loader import LabelledTextDS
from plotting import print_accuracies

dataset = LabelledTextDS(os.path.join('data', 'test.csv'))
# dataset.df = pd.read_csv(path).sample(frac=1.0, random_state=6490)
# print(dataset.df)
# print(dataset.df['label'])  # 2    pos, 1    neg, 0    pos
# print(dataset.df['label'].unique())  # ['pos' 'neg']
# print(dataset.class_to_id)  # {'pos': 0, 'neg': 1}
# print(dataset.df['label'].unique())  # [0 1]
# print(dataset.df['tokens'])
for a in dataset.df['tokens']:
    print(a)
# print(dataset.df['label'].values)

# train, valid, test = dataset.get_vector_representation()
# print("train[0]:")
# print(train[0])
# print("train[1]:")
# print(train[1])
# print("valid[0]:")
# print(valid[0])
# print("valid[1]:")
# print(valid[1])
# print("test[0]:")
# print(test[0])
# print("test[1]:")
# print(test[1])

model = LogisticRegression()  # You can change the hyper-parameters of the model by passing args here

# model.fit(train[0], train[1])
# train_accuracy = (model.predict(train[0]) == train[1]).astype(float).mean()
# valid_accuracy = (model.predict(valid[0]) == valid[1]).astype(float).mean()
# test_accuracy = (model.predict(test[0]) == test[1]).astype(float).mean()
#
# print_accuracies((train_accuracy, valid_accuracy, test_accuracy))
