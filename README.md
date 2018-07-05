# LSTM_pytorch
The goal of this repository is to train LSTM model for a classification purpose on simple datasets which their difficulties/size are scalable. The examples have variable sequence length which using *pack_padded_sequence* and *pad_packed_sequence* is necessary. The code is written based on Pytorch Dataset and Dataloader packages which let you employ parallel workers. 

# Datasets
There are currently two datasets. The first one is a sort of identity function. Given the in input sequence [4,4,4,4,4] and [3,3] the model should be able to learn to classify them as 4 and 3, respectively. You can increase the number of classes (means from 1 to *class_no* can appear in the input sequence), the number of samples, minimum and maximum input sequence length.
Run this example by 

```
python main.py --identity
```
The second example is to find the mode (the most frequent element) in a given sequence. 
```
python main.py --mode
```

# Acknowlegment
Thanks [Egor Lakomkin](https://github.com/EgorLakomkin) and [Chandrakant Bothe](https://github.com/crbothe) for their valuable feedback.
