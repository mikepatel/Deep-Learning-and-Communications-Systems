# Michael Patel
# Fall 2018

# Autoencoder repr. of communications system
#
# Based on research paper: https://arxiv.org/pdf/1702.00832.pdf

# Notes:

################################################################################
# IMPORTs
import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, GaussianNoise, Input, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers Adam

import numpy as np

################################################################################
# HYPERPARAMETERS

# BUILD MODEL
