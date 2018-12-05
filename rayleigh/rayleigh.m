% Rayleigh Fading
%
csv_file = "ber.csv";
data = csvread(csv_file);
snr = data(:,1);
ber_auto = data(:,2);

% BPSK
ber_bpsk = berfading(snr, "psk", 16, 1);

% QAM
ber_qam = berfading(snr, "qam", 16, 1);

semilogy(snr, ber_auto, "b", ...
    snr, ber_bpsk, "r", ...
    snr, ber_qam, "g");

legend("16-autoencoder", "16-BPSK", "16-QAM", "Location", "northeast");
grid on
axis([-15 15 10^-7 1])
title("AWGN and Rayleigh Fading");
xlabel("SNR (dB)");
ylabel("BER");