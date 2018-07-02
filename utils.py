import torch
import numpy as np
USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LONG = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
LONGTENSOR = torch.LongTensor


def batch_padding(seq_batch, batch_size, dtype=FLOAT):
    seq_lengths = torch.from_numpy(np.array([i[0].shape[0] for i in seq_batch])).type(dtype)
    seq_tensor = torch.zeros(batch_size, max(seq_lengths)).type(dtype)
    seq_label = torch.zeros(batch_size, 1).type(dtype)
    counter = 0
    # print(seq_batch)
    for i in range(batch_size):
        # print(i)
        seq_batch[i][0]
        seq_tensor[counter, :seq_lengths[i]]
        seq_tensor[counter, :seq_lengths[i]] = seq_batch[i][0]
        seq_label[counter, :seq_lengths[i]] = seq_batch[i][1]
        counter += 1

        # SORT YOUR TENSORS BY LENGTH!
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    seq_label = seq_label[perm_idx]
    return seq_tensor, seq_label, seq_lengths


def adjust_learning_rate(optimizer, epoch, rate):
    lr = rate * (0.1 ** (epoch // 100))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def collate_batch(lst_samples):
    batch_x, batch_y, seq_lengths = batch_padding(lst_samples, len(lst_samples), dtype=LONGTENSOR)
    return batch_x, batch_y, seq_lengths