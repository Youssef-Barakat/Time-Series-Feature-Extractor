import ruptures as rpt
import numpy as np
import pandas as pd
from typing import Optional 
import matplotlib.pyplot as plt
from scipy import stats
from Utils import global_params

def get_level_shift(df: pd.DataFrame, plot: Optional[bool] = True) -> dict:

    """
    Detects a level shift in a time series DataFrame and calculates the shift scale.

    This function takes a pandas DataFrame containing a time series and detects a single level shift in the data.
    It uses the Kernel Change Point Detection (KernelCPD) algorithm from the ruptures library to find the change point,
    and calculates the shift scale based on the means of the data before and after the change point.

    Parameters:
        df (pd.DataFrame): A pandas DataFrame containing the time series data.

    Returns:
        Tuple[int, float]: A tuple containing the detected change point (index) and the calculated shift scale.
            - The change point is the index where the level shift is detected.
            - The shift scale represents the relative magnitude of the level shift, calculated as the ratio of
            the difference in means after and before the change point to the mean before the change point.
    """

    plot = global_params['level_shift']['plot']

    algo_c = rpt.KernelCPD(kernel="linear", min_size=3).fit(df.values)
    result = algo_c.predict(n_bkps=1)
    change_point = result[0]

    
    shift_scale = (np.mean(df[result[0]:]) - np.mean(df[:result[0]])) / np.mean(df[:result[0]])

    if np.abs(shift_scale) < .43:
        return {'message': 'No Level Shift Detected'} 

    if plot:
        rpt.display(df.values, result, result)
        plt.title('Change Point Detection: Level Shift')
        plt.show()
    
    return {'change_point': change_point, 'scale': round(shift_scale,4)} 
