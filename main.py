import pandas as pd
from Extractor import FeatureExtractor
import warnings
warnings.filterwarnings("ignore")
from Utils import print_features


if __name__ == "__main__":

    df_clean = pd.read_csv("data/hourly/H18.csv", index_col=0, parse_dates=True)
    df = pd.read_csv("modified/hourly/level_shift/H18.csv", index_col=0, parse_dates=True)
    features = FeatureExtractor(df)
    features_extracted = features.get_features(choice = ["all"])
    print_features(features_extracted)

