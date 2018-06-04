import torch
import torch.utils.data as data
import numpy as np


class NumpyDataset(data.Dataset):
    def __init__(self, X, y):
        super(NumpyDataset, self).__init__()
        self.X = X
        self.Y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        X_tensor = torch.from_numpy(self.X[index,:])
        Y_tensor = torch.Tensor([self.Y[index]]) if self.Y is not None else torch.FloatTensor([0])
        return X_tensor, Y_tensor


class AvitoDataset(data.Dataset):
    def __init__(self, X_num, X_cat, X_text, X_word, y):
        super(AvitoDataset, self).__init__()
        self.X_num = X_num
        self.X_cat = X_cat
        self.X_text = X_text
        self.X_word = X_word
        self.Y = y

    def __len__(self):
        return self.X_num.shape[0]

    def __getitem__(self, index):
        X_num_tensor = torch.from_numpy(self.X_num[index, :])
        X_cat_tensor = torch.from_numpy(self.X_cat[index, :])
        X_text_tensor = [torch.from_numpy(np.array(text[index, :].toarray())).float().squeeze() for
                         text in self.X_text]
        X_word_tensor = [torch.from_numpy(np.array(word[index, :])).long() for word in self.X_word]
        Y_tensor = torch.Tensor([self.Y[index]]) if self.Y is not None else torch.FloatTensor([0])
        return X_num_tensor, X_cat_tensor, X_text_tensor, X_word_tensor, Y_tensor