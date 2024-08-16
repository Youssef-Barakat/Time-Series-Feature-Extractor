import pandas as pd
import numpy as np
from scipy import signal
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from typing import Tuple
import matplotlib.pyplot as plt
from Utils import global_params
from Functions.Seasonality import get_seasonality

def denoise_time_domain(df: pd.DataFrame) -> Tuple[np.ndarray,float]:
        """
        This function takes a pandas DataFrame containing a time series and denoises it using a time domain approach.
        it also calculates the power of the noise related to the signal.

        Parameters:
            df (pd.DataFrame): A pandas DataFrame containing the time series data.

        Returns:
            Tuple[np.ndarray,float]: A tuple containing the denoised time series and power of the noise related to the signal.
        """
        period,_  = get_seasonality(df)
        res  = seasonal_decompose(df,period=period)
        s2 = max(0,  (np.var(res.resid)/np.var(df.values) ))
        np.sqrt(s2)

        return (res.trend + res.seasonal).fillna(df.values.mean()),s2


def denoise(df: pd.DataFrame) -> Tuple[np.ndarray,float]:
    

    """
    This function takes a pandas DataFrame containing a time series and denoises it using a frequency domain approach.
    it also calculates the power of the noise related to the signal.

    Parameters:
        df (pd.DataFrame): A pandas DataFrame containing the time series data.
        sample_rate (int): The sampling rate of the time series. Defaults to None.
        param (int): The parameter whict is the time period used to determine the cutoff frequency. Defaults to None.

    Returns:
        Tuple[np.ndarray,float]: A tuple containing the denoised time series and power of the noise related to the signal.
    """
    time_period = global_params["noise"]["time_period"]
    plot = global_params["noise"]["plot"]

    signal = df.values.flatten()
    n = len(signal)
    sample_rate = n

    if time_period == None:
        if n > 1000:
            time_period = 4
        else :
            time_period = 4

    if n > 1000:
        time_period *= 4 
        
    cutoff = int((n//2) // time_period) 

    frequencies = np.fft.fftfreq(n, d=1/sample_rate)
    
    fft_values = np.fft.fft(signal)
    fft_noise = np.copy(fft_values)
    fft_total = np.copy(fft_values)
    
    magnitude_spectrum = np.abs(fft_values)
    magnitude_spectrum[n//2:] = 0
    
    fft_values[frequencies>cutoff] = 0
    fft_noise[frequencies<cutoff] = 0

    psd_sig = np.sum(np.abs(fft_total) ** 2 / n)
    psd_noise = np.sum(np.abs(fft_noise) ** 2 / n)

    time_domain_signal = np.fft.ifft(fft_values)
    time_domain_signal = np.real(time_domain_signal)

    if plot:
        plt.figure(figsize=(20, 13))

        plt.subplot(2,1,1)
        plt.plot(df.values.flatten())

        plt.subplot(2,1,2)
        plt.plot(time_domain_signal)

    period,_  = get_seasonality(df)
    res  = seasonal_decompose(df,period=period)
    s2 = max(0,  (np.var(res.resid)/np.var(df.values) ))
    s = np.sqrt(s2)
     


    return time_domain_signal, round(psd_noise/psd_sig,4) , s2