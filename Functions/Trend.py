import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats
from typing import Optional
from statsmodels.nonparametric import smoothers_lowess as sm_lowess


def get_trend(
        data: pd.DataFrame, 
        threshold :Optional[float] = 0.1,
        return_smoothed_data :Optional[bool] = 0
        ) -> dict:

        """
        Detecting the Trend from a given time series.
        
        Parameters:
            time_series (pd.DataFrame): The input time series as a dataframe.
            threshold (float): The needed threshold where we accept significant Trend.
            return_smmothed_data (bool): If True, We need to return the model and the data.
            
        Returns:
            If return_smoothed_data is False:
                float: That holds the slope of the fitted line "The strength of the trend".
                int: has three values (1, 0 or -1), 
                    1: The Increasing Trend.
                    0: No Trend.
                    -1: The Decreasing Trend.
                trend_type: First, or exponential Trend (if no significant trend this will not be returned).

            If return_smoothed_data is True:
                LinearRegression: The fitted model.
                np.ndarray: Contain the Smoothed data from Lowess function.
                trend_type: First, or exponential Trend.

            
        """
        # Charchs of the Data:
        array = data.values.flatten()
        n = len(array)
        
        # Smoothing the data:
        index_vector = np.array([i for i in range(n)])
        smoothed_data = sm_lowess.lowess(array, index_vector)

        # Fitting Line:
        lr = LinearRegression()
        lr.fit(index_vector.reshape(-1, 1), array)
        
        # Returning the type of the Trend:
        statistic = stats.anderson(smoothed_data[:,0], 'expon')
        
        if statistic[0] == float("inf"):
            trend_type = "First Order Trend"
        else:
            trend_type = "Exponential Trend"

        if return_smoothed_data == 1:
            return {"model": lr,
                    "data": smoothed_data}

        if (np.abs(np.arctan(lr.coef_[0])) > threshold):
            if (lr.coef_ > 0):
                return {"slope": np.arctan(lr.coef_[0]),
                        "direction": "Up",
                        "trend_type": trend_type}
            else:
                return {"slope": np.arctan(lr.coef_[0]),
                        "direction": "Down",
                        "trend_type": trend_type}
        else:
            return {"message":"No Trend Detected"}