import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import statsmodels.graphics.gofplots as smg
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import normaltest
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

import statsmodels.api as sm

class Plotter:
    ##def __init__(self, df):
        ##self.df = df 
    
    def box_plot_outlier_check(self, df):
        cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
        for col in cols:
            fig = px.box(df, y=col)
            fig.show()     

    def hist_plotter(self, df):
        cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
        fig, axes = plt.subplots(nrows=len(cols), ncols=1, figsize=(12, 8))  # Create a single subplot for each column

        for i, col in enumerate(cols):
            df[col].hist(bins=40, ax=axes[i])  # Plot histograms on each subplot
            axes[i].set_title(col)  # Set title for each subplot

        plt.tight_layout()  # Adjust spacing between subplots
        plt.show()

    def qq_plotter(self, df):
        cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
        for col in cols:
            fig, ax = plt.subplots()
            smg.qqplot(self.df[col], line='q', ax=ax)  # Create q-q plot with diagonal line
            ax.set_title(f'Q-Q Plot of {col}')
            ax.set_xlabel('Quantiles of Standardized Data')
            ax.set_ylabel('Quantiles of {col}')
            plt.show()
            
    def d_agostino_k2_test(self, df):
        cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
        for col in cols:
            stat, p = normaltest(self.df[col], nan_policy='omit')
            print(f"D'Agostino's K-squared Test for {col}")
            print('Statistics=%.3f, p=%.3f' % (stat, p))
            print('----------------------------------')

    ## def plot_tool_wear_vs_features(self):

        cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]']
        for col in cols:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=col, y='Tool wear [min]', data=self.df)
            plt.xlabel(col)
            plt.ylabel('Tool wear [min]')
            plt.title(f'Tool wear vs. {col}')
            plt.show()

    def test_log(self, df):
        cols = ['Rotational speed [rpm]']
        for col in cols:
            log_column = df[col].map(lambda i: np.log(i) if i > 0 else 0)
            t=sns.histplot(log_column,label="Skewness: %.2f"%(log_column.skew()) )
            t.legend()

    def test_boxcox(self, df):
        cols = ['Rotational speed [rpm]']
        for col in cols:
            boxcox_column = df[col]
            boxcox_column = stats.boxcox(boxcox_column)
            boxcox_column = pd.Series(boxcox_column[0])
            t=sns.histplot(boxcox_column,label="Skewness: %.2f"%(boxcox_column.skew()) )
            t.legend()

    def calculate_vif(self, df):
        """
        Calculate Variance Inflation Factor (VIF) for each variable in the selected columns.

        Parameters:
        - df (pd.DataFrame): The DataFrame containing the data.
        - selected_columns (list of str): List of column names to check for multicollinearity.

        Returns:
        - pd.DataFrame: DataFrame containing VIF values for each variable.
        """
        selected_columns = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
        # Subset the DataFrame with only the selected columns
        X = df[selected_columns]
        
        # Add a constant column to the data for VIF calculation
        X = sm.add_constant(X)

        # Calculate VIF for each feature
        vif_data = pd.DataFrame({
            "Variable": X.columns,
            "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        })

        # Drop the constant term from the result
        vif_data = vif_data[vif_data['Variable'] != 'const']

        return vif_data
