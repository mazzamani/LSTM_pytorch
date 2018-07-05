import argparse

from dataset import IdentityDataset, ModeDataSet
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn

import torch.optim as optim
from multiprocessing import set_start_method

# try:
#     set_start_method('spawn')
# except RuntimeError:
#     pass
from models import LSTMmodel
from utils import adjust_learning_rate, collate_batch

USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LONG = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
LONGTENSOR = torch.LongTensor


def update(mode):
    if mode == 'train':
        model.train()
        load = train_loader

    elif mode == 'val':
        load = val_loader
        model.eval()

    elif mode == 'test':
        load = test_loader
        model.eval()

    #optimizer = adjust_learning_rate(optimizer, epoch)
    total = 0.0
    total_loss = 0.0
    total_acc = 0.0
    for iter, data in enumerate(load):
        inputs, labels, seq_lens = data
        labels = torch.squeeze(labels)

        if USE_CUDA:
            inputs, labels = Variable(inputs.cuda()), labels.cuda()
        else:
            inputs = Variable(inputs)

        if mode == 'train':
            model.zero_grad()

        model.batch_size = len(labels)
        model.hidden = model.init_hidden()
        output = model(inputs, seq_lens)
        loss = loss_function(output, Variable(labels))

        if mode == 'train':
            loss.backward()
            optimizer.step()

        _, predicted = torch.max(output.data, 1)
        total_acc += (predicted == labels).sum()
        total += len(labels)
        total_loss += loss.data[0]
    loss = total_loss / total
    acc= total_acc / total
    return loss, acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')

    #experiment setup
    parser.add_argument('-rate', default=0.1, type=float, help='learning rate')
    parser.add_argument('-batch_size', default=64, type=int, help='minibatch size')
    parser.add_argument('-init_w', default=0.003, type=float, help='')
    parser.add_argument('-epochs', default=1000, type=int, help='train iters each timestep')
    parser.add_argument('-seed', default=-1, type=int, help='')

    #model
    parser.add_argument('-embed_dim', type=int, default=10, help='number of embedding dimension')
    parser.add_argument('-lstm_hidden_dim', type=int, default=20, help='the number of embedding dimension in LSTM hidden layer')
    parser.add_argument('-lstm_num_layers', type=int, default=3, help='the number of LSTM  layers')
    parser.add_argument('-init_weight', action='store_true', help='init w')
    parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
    parser.add_argument('-dropout', type=float, default=0.0, help='the probability for dropout [default: 0.0]')


    #Train Dataset
    parser.add_argument('-cls', type=int, default=5, help='number of output classes in the generated training dataset')
    parser.add_argument('-train_sample_no', type=int, default=2048, help='total number of samples in the generated training dataset')
    parser.add_argument('-train_min_seq_len', type=int, default=5, help='minimum length for each sample in the generated training dataset')
    parser.add_argument('-train_max_seq_len', type=int, default=10, help='maximum length for each sample in the generated training dataset')

    #Val Dataset
    parser.add_argument('-val_sample_no', type=int, default=256, help='total number of samples in the generated validation dataset')
    parser.add_argument('-val_min_seq_len', type=int, default=7, help='minimum length for each sample in the generated validation dataset')
    parser.add_argument('-val_max_seq_len', type=int, default=11, help='maximum length for each sample in the generated validation dataset')

    #Test Dataset
    parser.add_argument('-test_sample_no', type=int, default=256, help='total number of samples in the generated test dataset')
    parser.add_argument('-test_min_seq_len', type=int, default=8, help='minimum length for each sample in the generated test dataset')
    parser.add_argument('-test_max_seq_len', type=int, default=13, help='maximum length for each sample in the generated test dataset')


    #Dataset
    parser.add_argument('--identity', dest='identity', action='store_true', help='to regenerate what is given as in input sequence')
    parser.add_argument('--mode', dest='mode', action='store_true',  help=' calculates the reminder of sum of a variable sequence to the given #class_no')

    parser.set_defaults(visualize=False)
    parser.set_defaults(reduced=False)
    parser.set_defaults(mode=False)

    args = parser.parse_args()

    if args.mode:
        MyDatatSet = ModeDataSet

    else:
        MyDatatSet = IdentityDataset

    train = MyDatatSet(class_no=args.cls, sample_no=args.train_sample_no, min_seq_len=args.train_min_seq_len, max_seq_len=args.train_max_seq_len)
    val = MyDatatSet(class_no=args.cls, sample_no=args.val_sample_no, min_seq_len=args.val_min_seq_len, max_seq_len=args.val_max_seq_len)
    test = MyDatatSet(class_no=args.cls, sample_no=args.test_sample_no, min_seq_len=args.test_min_seq_len, max_seq_len=args.test_max_seq_len)

    model = LSTMmodel(args.cls + 1, args)
    print(model)

    if USE_CUDA:
        model.cuda()

    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=1, collate_fn=collate_batch,
                              drop_last=True)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=1, collate_fn=collate_batch,
                              drop_last=True)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=1, collate_fn=collate_batch,
                              drop_last=True)

    #early_stopping = EarlyStopping(mode='min', patience=10, min_delta= -0.0001)
    optimizer = optim.Adam(model.parameters(), lr=args.rate)
    loss_function = nn.CrossEntropyLoss()

    train_loss_ = []
    test_loss_ = []
    train_acc_ = []
    test_acc_ = []
    val_loss_ = []
    val_acc_ = []
    for epoch in range(args.epochs):

        optimizer = adjust_learning_rate(optimizer, epoch, args.rate)
        loss, acc = update('train')
        train_loss_.append(loss)
        train_acc_.append(acc)

        loss, acc = update('val')
        val_loss_.append(loss)
        val_acc_.append(acc)

        loss, acc = update('test')
        test_loss_.append(loss)
        test_acc_.append(acc)


        print('[Epoch: %3d/%3d] Train Loss: %.4f, Val Loss: %.4f, Test Loss: %.4f,   Train Acc: %.4f, Val Acc: %.4f, Test Acc: %.4f'
              % (epoch, args.epochs, train_loss_[epoch], val_loss_[epoch], test_loss_[epoch], train_acc_[epoch], val_acc_[epoch], test_acc_[epoch]))





