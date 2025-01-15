import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import statsmodels.graphics.gofplots as smg
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import normaltest
from scipy import stats
from sklearn.preprocessing import PowerTransformer

power_transformer = PowerTransformer(method='box-cox')

class DataFrameTransform:
    ##def __init__ (self, df):
        ##self.df = df
    
    def impute_nulls_mean(self, df):
        """Imputes nulls based on the mean value

        Args:
            df : dataframe

        Returns:
           df
        """
        cols = ['Air temperature [K]', 'Process temperature [K]', 'Tool wear [min]']
        for col in cols:
            col_mean = df[col].mean()
            df[col].fillna(col_mean, inplace=True)
        return df
    
    def correct_skew_boxcox(self, df):
        """Corrects skew of data using a Box-Cox transformation
        Args:
            df : dataframe

        Returns:
           df
        """
        cols = ['Rotational speed [rpm]']
        for col in cols:
            df[col] = power_transformer.fit_transform(df[[col]])
            return df
        
    def z_scores(self, df):
        """Generates z-scores

        Args:
            df : dataframe

        Returns:
           df
        """
        cols = ['Torque [Nm]', 'Rotational speed [rpm]']
        for col in cols:
            df[f"{col}_zscore"] = stats.zscore(df[col]) 
        return df
    
    def clean_z_score_outliers(self, df):
        """Cleans the outliers generated from the z-scores

        Args:
            df : dataframe

        Returns:
           df
        """        

        cols = ['Torque [Nm]_zscore', 'Rotational speed [rpm]_zscore']
        for col in cols:
            df = df[(df[col] <= 3) & (df[col] >= -3)]
        return df
    
    def remove_outliers_iqr(self, df):
        """Removes outliers based on calculations of the inter-quartile-range

        Args:
            df : dataframe

        Returns:
           df
        """        

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
    

    def undo_boxcox(self, df):
        """Inverses the Box-Cox transformation

        Args:
            df : dataframe

        Returns:
           df
        """
        cols = ['Rotational speed [rpm]']
        for col in cols:
            # Use inv_boxcox to reverse the transformation
            df[col] = power_transformer.inverse_transform(df[[col]])
            
        return df

    def tool_wear_visualization(self, df):
        """
        Create a visualization displaying the number of tools operating
        at different tool wear values.
        
        Parameters:
        - df (pd.DataFrame): The DataFrame containing the 'Tool wear [min]' column.
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(df['Tool wear [min]'], bins=20, kde=False, color='skyblue')
        plt.title('Distribution of Tool Wear [min]', fontsize=16)
        plt.xlabel('Tool Wear [min]', fontsize=14)
        plt.ylabel('Number of Tools', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()