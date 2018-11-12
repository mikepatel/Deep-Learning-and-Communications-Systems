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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, GaussianNoise, Dropout, \
    BatchNormalization, Embedding, Flatten
from tensorflow.keras.activations import relu, softmax, linear
from tensorflow.keras.losses import categorical_crossentropy
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
NUM_EPOCHS = 6
DROPOUT_RATE = 0.4

################################################################################
# create training data
size_train_data = 10000
train_data = []
train_label_idx = np.random.randint(M, size=size_train_data)  # list of indices that will eventually have value=1
for idx in train_label_idx:
    row = np.zeros(M)  # create row of 0s of length M
    row[idx] = 1
    train_data.append(row)

train_data = np.array(train_data)
print(train_data[:5])
#print(train_data.shape)

# create validation set
size_val_data = 2000
val_data = []
val_label_idx = np.random.randint(M, size=size_val_data)
for idx in val_label_idx:
    row = np.zeros(M)
    row[idx] = 1
    val_data.append(row)

val_data = np.array(val_data)

# create test set
size_test_data = 30000
test_data = []
test_label_idx = np.random.randint(M, size=size_test_data)
for idx in test_label_idx:
    row = np.zeros(M)
    row[idx] = 1
    test_data.append(row)

test_data = np.array(test_data)


################################################################################
# BUILD MODEL
def build_model():
    m = Sequential()

    # ========= transmitter =========
    m.add(Embedding(
        input_dim=M,
        output_dim=M,
        input_length=1,
        input_shape=(M, )
    ))

    m.add(Flatten())

    m.add(Dense(
        units=M,
        activation=relu
    ))

    m.add(BatchNormalization())

    m.add(Dense(
        units=num_channels,
        activation=linear  # ?????
    ))

    m.add(BatchNormalization())

    # ========= channel =========
    # Noise layer
    m.add(GaussianNoise(
        stddev=np.sqrt(beta_variance)
    ))

    # ========= receiver =========
    m.add(Dense(
        units=M,
        activation=relu
    ))

    m.add(BatchNormalization())

    m.add(Dense(
        units=M,
        activation=softmax
    ))

    m.summary()
    return m


# autoencoder
autoencoder = build_model()

autoencoder.compile(
    loss=categorical_crossentropy,
    optimizer=Adam(),
    metrics=["accuracy"]
)

################################################################################
# callbacks
dir = os.path.join(os.getcwd(), datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
if not os.path.exists(dir):
    os.makedirs(dir)

history_file = dir + "\checkpoints.h5"
save_callback = ModelCheckpoint(filepath=history_file, verbose=1)
tb_callback = TensorBoard(log_dir=dir)


# TRAIN MODEL
history = autoencoder.fit(
    x=train_data,
    y=train_data,
    epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(val_data, val_data),
    callbacks=[save_callback, tb_callback],
    verbose=1
)

history_dict = history.history
train_accuracy = history_dict["acc"]
train_loss = history_dict["loss"]
valid_accuracy = history_dict["val_acc"]
valid_loss = history_dict["val_loss"]

################################################################################
# VISUALIZATION
start = -10
end = 15
range_SNR_dB = list(np.linspace(start, end, 2*(end-start)+1))
#print(range_SNR)
ber = [None] * len(range_SNR_dB)

for i in range(0, len(range_SNR_dB)):
    # convert dB to W
    snr = np.power(10, range_SNR_dB[i] / 10)

    # noise parameters
    mean_noise = 0
    std_noise = np.sqrt(1 / (2*R*snr))
    noise = std_noise * np.random.randn(size_test_data, M)  # randn => standard normal distribution

    # evaluate model
    predictions = autoencoder.predict(test_data)

    # construct signal = input + noise
    signal = predictions + noise
    signal = np.round(signal)

    errors = np.not_equal(signal, test_data)  # boolean test
    ber[i] = np.mean(errors)

    print("SNR: {}, BER: {:.6f}".format(range_SNR_dB[i], ber[i]))

# Plot
plt.plot(range_SNR_dB, ber, "o")
plt.yscale("log")
plt.ylim(10**(-4), 1)
plt.title("Autoencoder: ")
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.grid()

# save plot fig to file
image_file = dir + "\plot_ber"
plt.savefig(image_file)

#plt.show()


