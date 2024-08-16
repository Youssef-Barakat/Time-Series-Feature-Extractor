import pandas as pd
from typing import Optional

def plateau_report(
    length: int, start_index: int, end_index: int
    ) -> dict:

    """
    Returns a dictionary containing information about a detected plateau.
    
    Parameters:
        "length": Plateau length,
        "avg_rolling_std": Average rolling standard deviation,
        "threshold": Threshold at which the plateau was detected,
        "start_index": start index of the plateau,
        "end_index": end index of the plateau

    Returns:
        A dictionary containing information about a detected plateau.
    """

    return {
        "length": length,
        "start_index": start_index,
        "end_index": end_index
    }

def get_plateau(
        data_values: pd.DataFrame, window_size: int = 3, threshold: Optional[float] = None , min_plateau_length: int = 100
        ) -> list:
    """
    Detects plateaus in a given time series.

    Parameters:
        "data_values": A pandas DataFrame containing the values of time series data,
        "window_size": The window size to use for the rolling standard deviation,
        "rstd_threshold": The rolling std threshold at which to detect plateaus,
        "min_plateau_length": The minimum length of a plateau to be detected

    Returns:
        A list of dictionaries containing information about the detected plateaus.
    """

    # Compute the rolling standard deviation
    rolling_std = data_values.rolling(window_size).std()
    
    # If the threshold is not provided, use the lowest non-zero value in the data
    if threshold is None:
        threshold = rolling_std.mean() / window_size

    # Instantiate the list of detected plateaus
    plateaus = []

    idx = 0
    while idx < len(rolling_std):
        std = rolling_std.iloc[idx]

        # If the std is below the threshold, we suspect a plateau
        if std < threshold:
            start_idx = idx - window_size + 1
            plateau_length = window_size - 1

            # Loop until we reach the end of the plateau
            while std <= threshold and idx < len(rolling_std) - 1:
                plateau_length += 1
                idx += 1
                std = rolling_std.iloc[idx]
            
            # If the plateau is long enough, add it to the list.
            if(plateau_length > min_plateau_length):
                plateaus.append(plateau_report(plateau_length, start_idx, idx))

        idx += 1
        
    return plateaus

