from torch.utils.data import Dataset
import torch
USE_CUDA = False# for multiprocessing torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LONG = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
import numpy as np

class IdentityDataset(Dataset):
    def __init__(self, class_no, sample_no, min_seq_len, max_seq_len):
        super(IdentityDataset, self).__init__()
        self.x = []
        self.y = []
        self.dataset_length = sample_no
        for i in range(self.dataset_length):
            y = np.random.randint(1, class_no + 1)
            sln = np.random.randint(min_seq_len, max_seq_len+1)
            inseq = np.ones(sln)*y
            label = np.array([y - 1])
            self.x.append(torch.from_numpy(inseq).type(LONG))
            self.y.append(torch.from_numpy(label).type(LONG))

    def __getitem__(self, item):
        return self.x[item], self.y[item]  # return in tensor x,y

    def __len__(self):
        return self.dataset_length


class ModeDataSet(Dataset):
    def __init__(self, class_no, sample_no, min_seq_len, max_seq_len):
        super(ModeDataSet, self).__init__()
        self.x = []
        self.y = []
        self.dataset_length = sample_no
        for i in range(self.dataset_length):
            inseq = np.random.randint(1, class_no, np.random.randint(min_seq_len, max_seq_len + 1))
            (values, counts) = np.unique(inseq, return_counts=True)
            label = np.array([np.argmax(counts) + 1])
            self.x.append(torch.from_numpy(inseq).type(LONG))
            self.y.append(torch.from_numpy(label).type(LONG))

    def __getitem__(self, item):
        return self.x[item], self.y[item]  # return in tensor x,y

    def __len__(self):
        return self.dataset_length


class YourDataSet(Dataset):
    def __init__(self, X, Y):
        super(YourDataSet, self).__init__()
        self.x = []
        self.y = []
        self.dataset_length = len(Y)
        for i in range(self.dataset_length ):
            self.x.append(torch.from_numpy(X[i]).type(LONG))
            self.y.append(torch.from_numpy(Y[i]).type(LONG))

    def __getitem__(self, item):
        return self.x[item], self.y[item]  # return in tensor x,y

    def __len__(self):
        return self.dataset_length