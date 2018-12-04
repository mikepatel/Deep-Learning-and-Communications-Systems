# Michael Patel

# Autoencoder representation of communications system
# traditional model: tx-channel-rx
#
# Based on research paper: https://arxiv.org/pdf/1702.00832.pdf

# Notes:
#   - send k bits through n channel uses
#   - (n, k) autoencoder
#   - input s -> one-hot encoding -> M-dimensional vector
#   - instead of one-hot encoding, use message indices -> embedding -> vectors
#   - Embedding layer can only be used as 1st layer in model
#   - Rayleigh Fading via Rayleigh Distribution
#   - batch_size > 32 leads to poor accuracy results (BER curve)
#   - Added args parser for rayleigh fading

################################################################################
# IMPORTs
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, GaussianNoise, \
    BatchNormalization, Embedding, Flatten, Lambda
from tensorflow.keras.activations import relu, softmax, linear
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import shutil
import scipy.stats as ss


################################################################################
def delete_dir(d):
    if os.path.exists(d):
        shutil.rmtree(d)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--rayleigh", action="store_true", help="Rayleigh Fading")

    return parser.parse_args()


################################################################################
# Setup
idea_dir = os.path.join(os.getcwd(), ".idea")
delete_dir(idea_dir)

################################################################################
# HYPERPARAMETERS and CONSTANTS
FLAGS = get_args()  # depending on args, change model structure
print(FLAGS.rayleigh)

M = 16  # messages
k = int(np.log2(M))  # bits

num_channels = 2
R = k / num_channels  # comm rate R (bits per channel)
Eb_No_dB = 7  # 7 dB from paper
Eb_No = np.power(10, Eb_No_dB / 10)  # convert form dB -> W
beta_variance = 1 / (2*R*Eb_No)
print("\nBeta variance: ", beta_variance)

size_train_data = 40000
size_val_data = 5000
size_test_data = 10000

BATCH_SIZE = 16
NUM_EPOCHS = 12


################################################################################
def create_data_set(size):
    data = []

    for i in range(size):
        row = np.zeros(M)
        idx = np.random.randint(M)
        row[idx] = 1
        data.append(row)

    data = np.array(data)
    return data


################################################################################
# Rayleigh Fading
def rayleigh(x):
    s = np.sqrt(beta_variance)
    f = ss.rayleigh().pdf(np.linspace(ss.rayleigh.ppf(0.01), ss.rayleigh.ppf(0.99), num_channels))
    x = x * (1-f)
    return x


def rayleigh_shape(input_shape):
    return input_shape


################################################################################
def build_model():
    m = Sequential()

    # ---------------------------------
    # ---------- TRANSMITTER ----------
    # ---------------------------------
    m.add(Embedding(
        input_dim=M,
        output_dim=M,
        input_length=1,
        input_shape=(M,)
    ))

    m.add(Flatten())

    m.add(Dense(
        units=M,
        activation=relu
    ))

    m.add(BatchNormalization())

    m.add(Dense(
        units=M,
        activation=relu
    ))

    m.add(BatchNormalization())

    m.add(Dense(
        units=num_channels,
        activation=linear
    ))

    m.add(BatchNormalization())

    # ---------------------------------
    # ---------- CHANNEL ----------
    # ---------------------------------
    # Rayleigh Distribution
    if FLAGS.rayleigh:
        m.add(Lambda(
            function=rayleigh,
            output_shape=rayleigh_shape
        ))

    # Gaussian noise
    m.add(GaussianNoise(
        stddev=np.sqrt(beta_variance)
    ))

    # ---------------------------------
    # ---------- RECEIVER ----------
    # ---------------------------------
    m.add(Dense(
        units=M,
        activation=relu
    ))

    m.add(BatchNormalization())

    m.add(Dense(
        units=M,
        activation=relu
    ))

    m.add(BatchNormalization())

    m.add(Dense(
        units=M,
        activation=softmax
    ))

    return m


################################################################################
# BUILD MODEL
autoencoder = build_model()
autoencoder.summary()

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

# create training, validation, and test sets
train_data = create_data_set(size_train_data)
val_data = create_data_set(size_val_data)
test_data = create_data_set(size_test_data)

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
start = -15
end = 20
range_SNR_dB = list(np.linspace(start, end, 2*(end-start)+1))
ber = np.zeros(len(range_SNR_dB))

for i in range(0, len(range_SNR_dB)):
    # convert dB to W
    snr = np.power(10, range_SNR_dB[i] / 10)

    # evaluate model
    predictions = autoencoder.predict(test_data)

    # construct signal = (input * fading) + noise
    signal = predictions

    # fading
    if FLAGS.rayleigh:
        scale = np.sqrt(1 / (2 * R * snr))
        print("\nScale: ", scale)
        fading = ss.rayleigh().pdf(np.linspace(ss.rayleigh.ppf(0.01), ss.rayleigh.ppf(0.99), M))
        signal = np.multiply(signal, (1-fading))

    # noise parameters
    mean_noise = 0
    std_noise = np.sqrt(1 / (2 * R * snr))
    noise = mean_noise + std_noise * np.random.randn(size_test_data, M)  # randn => standard normal distribution
    signal = signal + noise

    signal = np.round(signal)

    errors = np.not_equal(signal, test_data)  # boolean test
    ber[i] = np.mean(errors)

    print("SNR: {}, BER: {:.8f}".format(range_SNR_dB[i], ber[i]))

# Plot BER curve
plt.plot(range_SNR_dB, ber, "o")
plt.yscale("log")
plt.ylim(10**(-7), 1)
title = "Autoencoder: Trained at " + str(Eb_No_dB) + " dB_" + str(k) + " bits" + \
        " with Rayleigh fading= " + str(FLAGS.rayleigh)
plt.title(title)
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.grid()

# save plot fig to file
image_file = dir + "\plot_ber_" + str(Eb_No_dB) + "dB_k=" + str(k) + "bits" + \
             " with Rayleigh fading= " + str(FLAGS.rayleigh)
plt.savefig(image_file)

#plt.show()

# save BER results to csv
df = pd.DataFrame([list(range_SNR_dB), list(ber)])
df = df.transpose()
csv_file = dir + "\\ber.csv"
df.to_csv(csv_file, header=None, index=None)
