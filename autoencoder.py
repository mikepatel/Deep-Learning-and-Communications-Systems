# Michael Patel

# Autoencoder repr. of communications system
#
# Based on research paper: https://arxiv.org/pdf/1702.00832.pdf

# Notes:
#   - send k bits through n channel uses
#   - (n, k) autoencoder
#   - input s -> one-hot encoding -> M-dimensional vector
#   - instead of one-hot encoding, use message indices -> embedding -> vectors
#   - Embedding layer can only be used as 1st layer in model

################################################################################
# IMPORTs
import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, GaussianNoise, Input, Dropout, \
    BatchNormalization, Embedding, Flatten
from tensorflow.keras.activations import relu, softmax, linear
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

################################################################################
# HYPERPARAMETERS and CONSTANTS
M = 4  # messages
k = int(np.log2(M))  # bits

num_channels = 2
R = k / num_channels  # comm rate R (bits per channel)
Eb_No_dB = 7  # from paper
Eb_No = np.power(10, Eb_No_dB / 10)  # convert form dB -> W
beta_variance = 1 / (2*R*Eb_No)

BATCH_SIZE = 16
NUM_EPOCHS = 10

#size_train_data =
dim_embed_out = M  # want same shape (i.e. think one-hot encoding)


################################################################################
# BUILD MODEL
# transmitter
def build_tx():
    m = Sequential()

    #m.add(Input(shape=(M, )))

    m.add(Embedding(
        input_dim=M,
        output_dim=M,
        input_length=1
    ))

    m.add(Flatten())

    m.add(Dense(
        units=M,
        input_shape=(M,),
        activation=relu
    ))

    m.add(Dense(
        units=num_channels,
        activation=linear  # ?????
    ))

    m.add(BatchNormalization())

    m.summary()
    return m


# channel
def build_channel():
    m = Sequential()

    # Noise layer
    m.add(GaussianNoise(
        stddev=np.sqrt(beta_variance)
    ))

    m.summary()
    return m


# receiver
def build_rx():
    m = Sequential()

    m.add(Dense(
        units=M,
        activation=relu
    ))

    m.add(Dense(
        units=M,
        activation=softmax
    ))

    m.summary()
    return m


# autoencoder
autoencoder = Sequential()
autoencoder.add(build_tx())
autoencoder.add(build_channel())
autoencoder.add(build_rx())
autoencoder.compile(
    loss=sparse_categorical_crossentropy,
    optimizer=Adam(),
    metrics=["accuracy"]
)

################################################################################
# TRAIN MODEL

# VISUALIZATION
