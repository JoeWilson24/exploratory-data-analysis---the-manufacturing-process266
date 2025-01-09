import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import statsmodels.graphics.gofplots as smg
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import normaltest
from scipy import stats

class DataFrameTransform:
    ##def __init__ (self, df):
        ##self.df = df
    
    def impute_nulls_mean(self, df):
        cols = ['Air temperature [K]', 'Process temperature [K]', 'Tool wear [min]']
        for col in cols:
            col_mean = df[col].mean()
            df[col].fillna(col_mean, inplace=True)
        return df
    
    def correct_skew_boxcox(self, df):
        cols = ['Rotational speed [rpm]']
        for col in cols:
            boxcox_column = df[col]
            boxcox_column = stats.boxcox(boxcox_column)
            boxcox_column = pd.Series(boxcox_column[0])
            df[col] = boxcox_column
            return df
        
    def z_scores(self, df):

        cols = ['Torque [Nm]', 'Rotational speed [rpm]']
        for col in cols:
            df[f"{col}_zscore"] = stats.zscore(df[col]) 
        return df
    
    def clean_z_score_outliers(self, df):

        cols = ['Torque [Nm]_zscore', 'Rotational speed [rpm]_zscore']
        for col in cols:
            df = df[(df[col] <= 3) & (df[col] >= -3)]
        return df
    
    def remove_outliers_iqr(self, df):

        cols = ['Air temperature [K]', 'Process temperature [K]', 'Tool wear [min]']
        for col in cols:        
            # Compute Q1 (25th percentile) and Q3 (75th percentile)
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            
            # Calculate IQR
            IQR = Q3 - Q1
            
            # Define outlier thresholds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Remove outliers

            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
        return df