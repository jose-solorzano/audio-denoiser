from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import torch
import torch.fft as fft

if __name__ == '__main__':
    # Define the length of the signals
    N = 256

    # Generate two random signals
    signal1 = torch.randn(N)
    signal2 = torch.randn(N)

    # Calculate the FFT of each signal
    fft_signal1 = torch.absolute(fft.fft(signal1)) ** 2
    fft_signal2 = torch.absolute(fft.fft(signal2)) ** 2

    # Calculate the FFT of the sum of the two signals
    sum_signal = signal1 + signal2
    fft_sum_signal = torch.absolute(fft.fft(sum_signal)) ** 2

    # Calculate the sum of the FFTs of each signal
    fft_sum_individual = fft_signal1 + fft_signal2

    # Convert the FFT results to numpy arrays for plotting
    fft_signal1_np = fft_signal1.numpy()
    fft_signal2_np = fft_signal2.numpy()
    fft_sum_signal_np = fft_sum_signal.numpy()
    fft_sum_individual_np = fft_sum_individual.numpy()

    mse_loss = mean_squared_error(fft_sum_signal_np, fft_sum_individual_np)

    # Calculate the correlation
    correlation, _ = pearsonr(fft_sum_signal_np, fft_sum_individual_np)

    print("MSE Loss:", mse_loss)
    print("Correlation:", correlation)
