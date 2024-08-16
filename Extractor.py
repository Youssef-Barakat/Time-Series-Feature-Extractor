import pandas as pd
import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from scipy import stats
from Utils import global_params
from Functions.Seasonality import get_seasonality
from Functions.Level_Shift import get_level_shift
from Functions.Trend import get_trend
from Functions.Variance_Shift import get_variance_shift
from Functions.Noise import denoise
from Functions.Plateau import get_plateau

class FeatureExtractor:

    def __init__(self, df):
        self.df = df

    def get_features(self, choice=['all']):

        '''
        This function takes a pandas DataFrame containing a time series and returns a dictionary of features after applying the required 
        feature extractors.

        Parameters: 
            df (pd.DataFrame): A pandas DataFrame containing the time series data.
            choice (list): A list of features to be returned. Defaults to ['all'].
        
        Returns:
            dict: A dictionary containing the features.
        '''

        if 'all' in choice:
            choice = ['variance_shift','seaonality','level_shift','trend','noise','noise','trend','plateau']

        data = self.df
        results = {}
        if 'seaonality' in choice:
            period, strength = get_seasonality(data)
            seasonality_results = {'period':period,'strength':strength}
            results.update({'seasonality':seasonality_results})

        if 'level_shift' in choice:
            res = get_level_shift(data)
            if 'message' in res:
                level_shift_results = res
            else:
                change_point, scale = res.get('change_point'),res.get('scale')
                level_shift_results = {'change_point':change_point,'scale':scale}
            results.update({'level_shift':level_shift_results})
            
        if 'trend' in choice:
            res = get_trend(data)
            if 'message' in res:
                trend_results = res
            else:   
                slope, direction, trend_type = res.get('slope'),res.get('direction'),res.get('trend_type')
                trend_results = {'slope':slope,'direction':direction,'trend_type':trend_type}
            results.update({'trend':trend_results})

        if 'variance_shift' in choice:
            res = get_variance_shift(data)
            if 'message' in res:
                variance_shift_results = res
            else:
                change_point, scale = res.get('change_point'),res.get('scale')
                variance_shift_results = {'change_point':change_point,'scale':scale}
            results.update({'variance_shift':variance_shift_results})

        if 'noise' in choice:
            denoised, strength,var = denoise(data)
            noise_results = {'denoised_data':denoised,'strength':strength,'variance':var}
            results.update({'noise':noise_results})

        if 'plateau' in choice:
            plat = get_plateau(data['value'])
            if len(plat) > 0:
                plat_results = {'start_index': plat[0]['start_index'],'end_index': plat[0]['end_index'], 'length': plat[0]['length']}
                results.update({'plateau':plat_results})
            else:
                plat_results = {'message': 'No Plateaus Detected'}
                results.update({'plateau':plat_results})

        return results