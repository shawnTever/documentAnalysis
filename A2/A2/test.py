import os

# from torch.autograd import Variable
from pandas import read_csv
from sklearn.linear_model import LogisticRegression
from torch import nn

from data_loader import LabelledTextDS
from plotting import print_accuracies

# dataset = LabelledTextDS(os.path.join('data', 'test.csv'))
# # dataset.df = pd.read_csv(path).sample(frac=1.0, random_state=6490)
# # print(dataset.df)
# # print(dataset.df['label'])  # 2    pos, 1    neg, 0    pos
# # print(dataset.df['label'].unique())  # ['pos' 'neg']
# # print(dataset.class_to_id)  # {'pos': 0, 'neg': 1}
# # print(dataset.df['label'].unique())  # [0 1]
# # print(dataset.df['tokens'])
# for a in dataset.df['tokens']:
#     print(a)
# # print(dataset.df['label'].values)
#
# # train, valid, test = dataset.get_vector_representation()
# # print("train[0]:")
# # print(train[0])
# # print("train[1]:")
# # print(train[1])
# # print("valid[0]:")
# # print(valid[0])
# # print("valid[1]:")
# # print(valid[1])
# # print("test[0]:")
# # print(test[0])
# # print("test[1]:")
# # print(test[1])
#
# model = LogisticRegression()  # You can change the hyper-parameters of the model by passing args here
#
# # model.fit(train[0], train[1])
# # train_accuracy = (model.predict(train[0]) == train[1]).astype(float).mean()
# # valid_accuracy = (model.predict(valid[0]) == valid[1]).astype(float).mean()
# # test_accuracy = (model.predict(test[0]) == test[1]).astype(float).mean()
# #
# # print_accuracies((train_accuracy, valid_accuracy, test_accuracy))


import os
import torch.optim

from data_loader import LabelledTextDS
from models import FastText
from plotting import *
from training import train_model

num_epochs = 5
num_hidden = 32  # Number of hidden neurons in model

# dev = 'cuda' if torch.cuda.is_available() else 'cpu'  # If you have a GPU installed, use that, otherwise CPU
# dataset = LabelledTextDS(os.path.join('data', 'test.csv'), dev=dev)

# # print(dataset.token_to_id)
# # print(len(dataset.class_to_id))
# # print(dataset.df['ids'])
#
# word_to_ix = dataset.token_to_id
# dataset.embeddings = nn.Embedding(len(dataset.token_to_id) + 2, 5, padding_idx=0)
# # print(type(word_to_ix['found']))
# # print(type(8))
# found_idx = torch.LongTensor([8])
# # print(found_idx)
# # hello_idx = Variable(hello_idx)
# found_embed = dataset.embeddings(found_idx)
# # print(found_embed)

# model = FastText(len(dataset.token_to_id)+2, num_hidden, len(dataset.class_to_id)).to(dev)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
#
# losses, accuracies = train_model(dataset, model, optimizer, num_epochs)
# torch.save(model, os.path.join('saved_models', 'classifier.pth'))
#
# print('')
# print_accuracies(accuracies)
# plot_losses(losses)

a = torch.FloatTensor([[[[1, 2, 1],
                        [2, 4, 1]],
                       [[3, 1, 1],
                        [4, 0, 1]],
                       [[1, 1, 1],
                        [1, 1, 1]]],
                       [[[1, 2, 1],
                         [2, 4, 1]],
                        [[3, 1, 1],
                         [4, 0, 1]],
                        [[1, 1, 1],
                         [1, 1, 1]]]])
print(f'0: {a.max(0)}')
print(f'0: {torch.max(a, 0)}')
# print(f'1: {a.max(1)}')
# print(f'2: {a.max(2)}')
# print(f'3: {a.max(3)}')
