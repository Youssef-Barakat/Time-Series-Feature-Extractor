import pandas as pd
import numpy as np
from scipy import signal
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from typing import Tuple

def get_seasonality(df: pd.DataFrame) -> Tuple[int, float]:

    """
    Identifies the dominant period of seasonality in a time series DataFrame.

    This function takes a pandas DataFrame containing a time series and calculates the dominant period of seasonality.
    It uses the periodogram analysis from the scipy.signal library to find the frequency components of the signal.
    The peak frequency in the periodogram is considered the dominant frequency, which corresponds to the dominant
    seasonality period.

    Parameters:
        df (pd.DataFrame): A pandas DataFrame containing the time series data.

    Returns:
        Tuple[int, float]: A tuple containing the identified dominant period and its corresponding power spectral density (PSD).
            - The dominant period is an integer representing the estimated period of the dominant seasonality component.
            - The PSD is a float value indicating the power of the dominant frequency component in the periodogram.
    """

    values = df.dropna().values.flatten()
    f,s = signal.periodogram(values)
    # if s.max() < 1e4:
    #     raise ValueError('No seasonality')
    res = adfuller(values)
    if res[1] > 0.05:
        period,strength = get_seasonality(df.diff().dropna())
        return period,strength
    peak = s.argmax()

    period = round(1/f[peak])
    s = seasonal_decompose(df,period = period)
    strength = max(0, 1 - (np.var(s.resid)/np.var(s.seasonal + s.resid) ))

    return period,strength