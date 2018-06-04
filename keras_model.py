# -*- coding: utf-8 -*-
from keras.models import Model
from keras.layers import Dense, Embedding, Input, concatenate
from keras.layers import *
from attention import Attention
from keras_utils import AttentionWithContext, Capsule
from keras.regularizers import l2


def BidLstmAmp(inp, max_features, embed_size, embedding_matrix):
    x = Embedding(max_features, embed_size, weights=[embedding_matrix],
                  trainable=False)(inp)
    x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(x)
    x = Dropout(0.25)(x)
    x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(x)
    x = Dropout(0.25)(x)
    x1 = GlobalMaxPooling1D()(x)
    x2 = GlobalAveragePooling1D()(x)
    x = concatenate([x1, x2])

    return x


def BidLstmAp(inp, max_features, embed_size, embedding_matrix):
    x = Embedding(max_features, embed_size, weights=[embedding_matrix],
                  trainable=False)(inp)
    x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(x)
    x = Dropout(0.25)(x)
    x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(x)
    x = Dropout(0.25)(x)
    x = GlobalAveragePooling1D()(x)

    return x


def BidLstmMp(inp, max_features, embed_size, embedding_matrix):
    x = Embedding(max_features, embed_size, weights=[embedding_matrix],
                  trainable=False)(inp)
    x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(x)
    x = Dropout(0.25)(x)
    x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(x)
    x = Dropout(0.25)(x)
    x = GlobalMaxPooling1D()(x)

    return x


def BidLstmMpAtn(inp, max_len, max_features, embed_size, embedding_matrix):
    x = Embedding(max_features, embed_size, weights=[embedding_matrix],
                  trainable=False)(inp)
    x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(x)
    x = Dropout(0.25)(x)
    x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(x)
    x = Dropout(0.25)(x)
    x1 = Attention(max_len)(x)
    x2 = GlobalMaxPooling1D()(x)
    x = concatenate([x1, x2])

    return x


def BidGRU(inp, max_len, max_features, embed_size, embedding_matrix):
    x = Embedding(max_features, embed_size, weights=[embedding_matrix],
                  trainable=False)(inp)
    x = Bidirectional(CuDNNGRU(50, return_sequences=True))(x)
    x = Attention(max_len)(x)

    return x


def RNNV2(inputs, max_features, embed_size, embedding_matrix):
    def att_max_avg_pooling(x):
        x_att = AttentionWithContext()(x)
        x_avg = GlobalAvgPool1D()(x)
        x_max = GlobalMaxPool1D()(x)
        return concatenate([x_att, x_avg, x_max])

    emb = Embedding(max_features, embed_size,
                    weights=[embedding_matrix], trainable=False)(inputs)

    l2_penalty = 0.0001
    # model 0
    x0 = SpatialDropout1D(0.25)(emb)
    s0 = Bidirectional(CuDNNGRU(50, return_sequences=True))(x0)
    x0 = att_max_avg_pooling(s0)

    # model 1
    x1 = SpatialDropout1D(0.25)(emb)
    s1 = Bidirectional(CuDNNGRU(50, return_sequences=True))(x1)
    x1 = att_max_avg_pooling(s1)

    # combine sequence output
    x = concatenate([s0, s1])
    x = Bidirectional(CuDNNGRU(50, return_sequences=True))(x)
    x = att_max_avg_pooling(x)

    # combine it all
    x = concatenate([x, x0, x1])
    return x

def CapsuleNet(inputs, max_features, embed_size, embedding_matrix):
    x = Embedding(max_features, embed_size,
                    weights=[embedding_matrix], trainable=False)(inputs)
    x = SpatialDropout1D(0.25)(x)
    x = Bidirectional(CuDNNGRU(50, return_sequences=True))(x)
    x = PReLU()(x)
    x = Capsule(
        num_capsule=5, dim_capsule=8,
        routings=3, share_weights=True)(x)
    x = Flatten()(x)

    return x


def CNN(inputs, max_features, embed_size, embedding_matrix):
    def conv_block(x, n, kernel_size):
        x = Conv1D(n, kernel_size, activation='relu')(x)
        x = Conv1D(25, kernel_size, activation='relu')(x)
        x_att = AttentionWithContext()(x)
        x_avg = GlobalAvgPool1D()(x)
        x_max = GlobalMaxPool1D()(x)
        return concatenate([x_att, x_avg, x_max])

    l2_penalty = 0.0001
    x_words = Embedding(max_features, embed_size,
                        weights=[embedding_matrix], trainable=False)(inputs)

    x = SpatialDropout1D(0.2)(x_words)
    x1 = conv_block(x, 50, 2)
    x2 = conv_block(x, 50, 3)
    x3 = conv_block(x, 50, 4)
    x_words = concatenate([x1, x2, x3])
    # x4 = conv_block(x, 1 * 50, 5)
    # x_words = concatenate([x1, x2, x3, x4])

    return x_words
