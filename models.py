import torch
import torch.nn as nn
import utils
import numpy as np


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class AvitorEmbedding(nn.Module):
    def __init__(self, token_len, embedding_size):
        super(AvitorEmbedding, self).__init__()
        self.n_embedding_layers = len(token_len)
        self.embedding_layers = [nn.Embedding(int(tl) + 1, embedding_size) for tl in token_len]
        for i, layer in enumerate(self.embedding_layers):
            self.add_module(f"embedding_layer_{i}", layer)

        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        return [self.embedding_layers[i](x[:, i]) for i in range(self.n_embedding_layers)]


class AvitorText(nn.Module):
    def __init__(self, text_inputs, drop_outs):
        super(AvitorText, self).__init__()
        self.text_inputs = text_inputs
        self.drop_outs = drop_outs
        self.fcs = []
        self.out_features = 128
        self.out_txt_features = 0
        for i, (txt_input, dropout) in enumerate(zip(text_inputs, drop_outs)):
            fc = nn.Sequential(
                nn.BatchNorm1d(txt_input),
                nn.Dropout(dropout),
                nn.Linear(txt_input, txt_input // 100),
                nn.Dropout(dropout),
                #nn.BatchNorm1d(txt_input // 100),
            )
            self.out_txt_features += txt_input // 100

            self.add_module(f"Text_layer_{i}", fc)
            self.fcs.append(fc)
        
        self.fc = nn.Sequential(
            nn.BatchNorm1d(self.out_txt_features),
            nn.Linear(self.out_txt_features, self.out_features),
            nn.ReLU(),
            nn.Dropout(self.drop_outs[0])
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, input):
        out = [fc(x) for x, fc in zip(input, self.fcs)]
        out = torch.cat(out, 1)
        out = utils.to_gpu(out)
        return self.fc(out)


class AvitorNum(nn.Module):
    def __init__(self, in_features, out_features = 50, dropout=0.5):
        super(AvitorNum, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.fc = nn.Sequential(
            nn.BatchNorm1d(self.in_features),
            nn.Linear(self.in_features, self.out_features),
            nn.ReLU(),
            nn.BatchNorm1d(self.out_features),
            nn.Dropout(dropout),
        )


        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        return self.fc(input)


class AvitorCat(nn.Module):
    def __init__(self, token_len, embedding_size=4):
        super(AvitorCat, self).__init__()
        self.n_embedding_layers = len(token_len)
        self.embedding_layers = [nn.Embedding(int(tl) + 1, embedding_size) for tl in token_len]
        for i, layer in enumerate(self.embedding_layers):
            self.add_module(f"category_layer_{i}", layer)

        self.out_features = self.n_embedding_layers * embedding_size

        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        return [self.embedding_layers[i](x[:, i]) for i in range(self.n_embedding_layers)]


class TensorRotate(nn.Module):
    # def __init__(self):
    #     super(TensorRotate, self).__init__()

    def forward(self, x):
        return x.permute(0, 2, 1).float()


class FloatTensor(nn.Module):
    def forward(selfs, x):
        return x.float()


# BiRNN Model (Many-to-One)
class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                            bidirectional=True)

    def forward(self, x):
        # Forward propagate RNN
        out, _ = self.gru(x)
        return out.permute(0, 2, 1)


# RNN Model (Many-to-One)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()

        # Forward propagate RNN
        out, _ = self.lstm(x, (h0, c0))

        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])
        return out


class AvitorWord(nn.Module):
    def __init__(self, max_features, token_len, embedding_size, weights):
        super(AvitorWord, self).__init__()
        self.max_features = max_features
        self.embedding_size = embedding_size
        self.token_len = token_len
        self.n_word_layer = len(token_len)

        self.word_layers = []

        for i, tkl in enumerate(token_len):
            embedding = nn.Embedding(self.max_features, self.embedding_size)
            embedding.weight = nn.Parameter(torch.from_numpy(np.array(weights)).double())
            embedding.weight.requires_grad = False

            word_layer = nn.Sequential(
                embedding,
                # FloatTensor(),
                # nn.Dropout(0.5),
                TensorRotate(),
                # BiRNN(self.embedding_size, 32, 2, 16),
                nn.BatchNorm1d(self.embedding_size),
                nn.Conv1d(in_channels=self.embedding_size, out_channels=50, kernel_size=3),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1),
                Flatten(),
                nn.Dropout(0.5)
            )
            self.add_module(f"word_layer_{i}", word_layer)
            self.word_layers.append(word_layer)

        self.out_features = 50

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)


    def forward(self, x):
        # print(self.word_layers[0](x[0]).shape)
        # print((x[0]))
        return [self.word_layers[i](x[i]) for i in range(self.n_word_layer)]


class Avitor(nn.Module):
    def __init__(self, num_model, cat_model, text_model, word_model):
        super(Avitor, self).__init__()
        self.num_model = num_model
        self.cat_model = cat_model
        self.text_model = text_model
        self.word_model = word_model

        self.in_features = self.num_model.out_features + \
                           self.cat_model.out_features + \
                           self.text_model.out_features + \
                           self.word_model.out_features

        self.fc = nn.Sequential(
            nn.BatchNorm1d(self.in_features),
            nn.Linear(in_features=self.in_features, out_features=50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(in_features=50, out_features=1),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, X_num, X_cat, X_text, X_word):
        X_num = utils.to_gpu(X_num)
        X_cat = utils.to_gpu(X_cat)
        X_text = [utils.to_gpu(text) for text in X_text]
        X_word = [utils.to_gpu(word) for word in X_word]

        out_num = self.num_model(X_num)
        out_cat = self.cat_model(X_cat)
        out_txt = self.text_model(X_text)
        out_word = self.word_model(X_word)

        all_features = torch.cat([out_num, *out_cat, out_txt, *out_word], 1)
        all_features = utils.to_gpu(all_features)

        return self.fc(all_features)