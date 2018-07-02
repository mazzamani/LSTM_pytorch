import torch.nn as nn
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import torch
USE_CUDA = torch.cuda.is_available()

class LSTMmodel(nn.Module):
    def __init__(self, vocab_size, args):
        super(LSTMmodel, self).__init__()
        self.args = args
        self.hidden_dim = args.lstm_hidden_dim
        self.num_layers = args.lstm_num_layers
        V = vocab_size
        self.batch_size = args.batch_size
        self.embed_dim = args.embed_dim
        if args.max_norm is not None:
            print("max_norm = {} ".format(args.max_norm))
            self.embed = nn.Embedding(V, self.embed_dim, max_norm=None, scale_grad_by_freq=False, padding_idx=0)
        else:
            print("max_norm = {} |||||".format(args.max_norm))
            self.embed = nn.Embedding(V, self.embed_dim, scale_grad_by_freq=True, padding_idx=0)

        self.lstm = nn.LSTM(input_size=self.embed_dim, hidden_size=self.hidden_dim, dropout=args.dropout,
                            num_layers=self.num_layers, batch_first=True)


        self.hidden = self.init_hidden()
        self.relu = nn.ReLU()
        self.lin_shallow = nn.Linear(self.hidden_dim, args.cls)
        self.softmax = nn.Softmax()

    def init_hidden(self):
        # the first is the hidden h
        # the second is the cell  c
        if USE_CUDA:  # self.args.cuda is True:
            return (Variable(torch.zeros(1 * self.num_layers, self.batch_size, self.hidden_dim)).cuda(),
                    Variable(torch.zeros(1 * self.num_layers, self.batch_size, self.hidden_dim)).cuda())
        else:
            return (Variable(torch.zeros(1 * self.num_layers, self.batch_size, self.hidden_dim)),
                    Variable(torch.zeros(1 * self.num_layers, self.batch_size, self.hidden_dim)))

    # def forward(self, coord, ins, prev_action, action, seq_lens=None):
    def forward(self, ins, seq_lens=None):
        embed = self.embed(ins)
        packed_input = pack_padded_sequence(embed, seq_lens.numpy(), batch_first=True)

        lstm_out, _ = self.lstm(packed_input, self.hidden)
        "pad_packed_sequence(lstm_out, batch_first=True)[0][::,-1].data.cpu().numpy()"
        pad_seq, _ = pad_packed_sequence(lstm_out, batch_first=True)
        lstm_last = torch.stack([pad_seq[i][seq_lens.cpu().numpy()[i] - 1] for i in range(self.batch_size)])


        output = self.lin_shallow(lstm_last)
        return output
