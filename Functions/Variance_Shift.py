import pandas as pd
import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from scipy import stats
from Utils import global_params


def get_variance_shift(
    df: pd.DataFrame,
    level_shift_index: Optional[int] = None,
    threshold_down: float = 0.8,
    threshold_up: float = 1.2
    ) -> dict:
    """
    Detecting the Variance Shift from a given time series.
    
    Parameters:
        time_series (pd.DataFrame): The input time series as a dataframe.
        level_shift_index (int): The level shift if the data has level shift(if not given, we will calculate it).
        threshold_down: The threshold for the Detection of variance shift for decreasing in variance shift.
        threshold_up: The threshold for the Detection of variance shift for Increasing in variance shift.
        
    Returns:
        dict contains:
            index (int): The index where the shift happened.
            (float): The scale factor of the variance shift.   
    """
    # Charchs of the Data:
    n = len(df)
    array = df.values.flatten()
    plot = global_params['variance_shift']['plot']

    
    # Calculating Level Shift if not given:
    if (level_shift_index == None):
        alg = rpt.KernelCPD(kernel="linear", min_size=3).fit(array)
        level_shift_index = alg.predict(n_bkps=1)[0]
        level_scale_factor = _calc_scale_factor(array, "level_detection", level_shift_index)
        
        if (level_scale_factor==0.43):
            level_shift_index = 0
        
    elif (type(level_shift_index) == str):
        level_shift_index = 0
    
    # Calculate EWMA
    ewma = df.ewm(span=2).std()
    # Using Level shift to find the change in the EWMA:
    algo_c = rpt.KernelCPD(kernel="linear", min_size=3).fit(ewma.dropna().values)
    index = algo_c.predict(n_bkps=1)[0]
    
    # Calculate the scale_factor:
    if (index > level_shift_index):
        var1 = array[level_shift_index:index].var()
        var2 = array[index:].var()
        scale_factor = np.sqrt(var2/var1)
        
    else:
        if (level_shift_index == 0):
            level_shift_index = -1
        level_shift_index = level_shift_index
        var1 = array[0:index].var()
        var2 = array[index:level_shift_index].var()
        if var2 == 0:
            var2 = array[index:].var()
        scale_factor = np.sqrt(var2/var1)

    if (scale_factor>threshold_down and scale_factor<threshold_up):
        return {"message": "No Signifcant Variance Shift detected."} 
    
    else:
        if plot: 
            rpt.display(df.values, [index,len(df)], [index,len(df)])
            plt.title('Change Point Detection: Variance Shift')
            plt.show()

        return {"change_point": index,
                "scale": scale_factor}
    

def _calc_scale_factor(array, func, val_1, val_2=None):
    if func == "level_detection":
        return np.mean(array[0: val_1]) / np.mean(array[val_1:])
    
    elif func == "variance_detection":
        return array[val_1:val_2].std()/array[val_2:].std()
    