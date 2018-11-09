# Michael Patel

# Autoencoder repr. of communications system
#
# Based on research paper: https://arxiv.org/pdf/1702.00832.pdf

# Notes:
#   - send k bits through n channel uses
#   - (n, k) autoencoder
#   - input s -> one-hot encoding -> M-dimensional vector
#   - instead of one-hot encoding, use message indices -> embedding -> vectors

################################################################################
# IMPORTs
import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, GaussianNoise, Input, Dropout, \
    BatchNormalization, Embedding
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

import os
import numpy as np
from datetime import datetime

################################################################################
# HYPERPARAMETERS
M = 4  # messages
k = int(np.log2(M))  # bits

BATCH_SIZE = 64
NUM_EPOCHS = 10

################################################################################
# BUILD MODEL
# transmitter
def build_tx():
    m = Sequential()
    m.add(Dense(
        units=M,
        input_shape=(M,),
        activation=relu
    ))

# channel

# receiver

# autoencoder
autoencoder = Sequential()

################################################################################
# TRAIN MODEL

# VISUALIZATION
