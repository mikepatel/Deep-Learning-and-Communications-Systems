# IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import os
import pandas as pd

from tensorflow.keras.models import load_model

################################################################################
'''

r = np.random.rayleigh(0.05, (100,))
print(r)


val = ss.rayleigh().pdf(ss.rayleigh.ppf(0.02))
print(val)
'''

################################################################################
#
M = 16  # k=4, so M=2^k=16
R = 2  # rate

################################################################################
# create test data to serve as "ground truth"
test_data = []
size_test_data = 10000
for i in range(size_test_data):
    row = np.zeros(M)
    idx = np.random.randint(M)
    row[idx] = 1
    test_data.append(row)
test_data = np.array(test_data)

################################################################################
# load model
filepath_to_model = os.path.join(os.getcwd(), "checkpoints.h5")
model = load_model(filepath=filepath_to_model)

# get model output
predictions = model.predict(test_data)

################################################################################
# Generate BER data points
start = -15
end = 15
range_SNR_dB = list(np.linspace(start, end, 2*(end-start)+1))
ber = np.zeros(len(range_SNR_dB))

for i in range(0, len(range_SNR_dB)):
    # convert dB to W
    snr = np.power(10, range_SNR_dB[i] / 10)

    signal = predictions

    # noise parameters
    mean_noise = 0
    std_noise = np.sqrt(1 / (2 * R * snr))
    #print("\nVariance: ", std_noise)
    noise = mean_noise + std_noise * np.random.randn(size_test_data, M)  # randn => standard normal distribution

    ################################################################################
    ################################################################################
    # HEY JOHN, HERE'S WHERE I NEED YOUR HELP
    # fading
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rayleigh.html

    #fading = ss.rayleigh().pdf(ss.rayleigh.rvs(size=M))
    fading = ss.rayleigh().pdf(np.linspace(ss.rayleigh.ppf(0.01), ss.rayleigh.ppf(0.99), M))
    fading = fading * 0.631  # correct for skewness
    signal = signal - fading

    ################################################################################
    ################################################################################

    # construct signal
    signal = signal + noise

    signal = np.round(signal)

    # compute errors
    errors = np.not_equal(signal, test_data)  # boolean test
    ber[i] = np.mean(errors)

    print("SNR: {}, BER: {:.8f}".format(range_SNR_dB[i], ber[i]))


################################################################################
# Plot BER curve
'''
plt.plot(range_SNR_dB, ber, "o")
plt.yscale("log")
plt.ylim(10**(-7), 1)
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.grid()
#plt.show()
'''

x = np.linspace(0.01, 0.99, M)
plt.plot(x, fading)
plt.show()

################################################################################
# save BER results to csv
df = pd.DataFrame([list(range_SNR_dB), list(ber)])
df = df.transpose()
csv_file = os.path.join(os.getcwd(), "ber.csv")
df.to_csv(csv_file, header=None, index=None)
