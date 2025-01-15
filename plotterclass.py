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
        """Creates box-plots of all desired columns

        Args:
            df
        """        
        cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
        for col in cols:
            fig = px.box(df, y=col)
            fig.show()     

    def hist_plotter(self, df):
        """Creates histograms of all desired columns

        Args:
            df
        """        
        cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
        fig, axes = plt.subplots(nrows=len(cols), ncols=1, figsize=(12, 8))  # Create a single subplot for each column

        for i, col in enumerate(cols):
            df[col].hist(bins=40, ax=axes[i])  # Plot histograms on each subplot
            axes[i].set_title(col)  # Set title for each subplot

        plt.tight_layout()  # Adjust spacing between subplots
        plt.show()

    def qq_plotter(self, df):
        """Creates qq-plots of all desired columns

        Args:
            df 
        """        
        cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
        for col in cols:
            fig, ax = plt.subplots()
            smg.qqplot(self.df[col], line='q', ax=ax)  # Create q-q plot with diagonal line
            ax.set_title(f'Q-Q Plot of {col}')
            ax.set_xlabel('Quantiles of Standardized Data')
            ax.set_ylabel('Quantiles of {col}')
            plt.show()
            
    def d_agostino_k2_test(self, df):
        """ Perfomes D'Agostino's K^2 Test on the desired columns

        Args:
            df
        """        
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
        """Performes a log transformation on the desired column and plots a new histogram, but does not make the change permanent

        Args:
            df
        """        
        cols = ['Rotational speed [rpm]']
        for col in cols:
            log_column = df[col].map(lambda i: np.log(i) if i > 0 else 0)
            t=sns.histplot(log_column,label="Skewness: %.2f"%(log_column.skew()) )
            t.legend()

    def test_boxcox(self, df):
        """Performes a Box-Cox transformation on the desired column and plots a new histogram, but does not make the change permanent
        Args:
            df 
        """        
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
    
    def calculate_operating_ranges(self, df):
        cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    # Ensure columns exist in the DataFrame
        for col in cols:
            if col not in cols:
                raise ValueError(f"Column '{col}' is not in the DataFrame.")
        
        # Create a summary table
        summary_table = df[cols].agg(['min', 'max', 'mean', 'std', 'median']).T
        summary_table.rename(columns={
            'min': 'Minimum',
            'max': 'Maximum',
            'mean': 'Mean',
            'std': 'Standard Deviation',
            'median': 'Median'
        }, inplace=True)
        summary_table.index.name = 'Variable'
        
        return summary_table
    
    def count_true_values(self, df):
        cols = ['Machine failure']
        for col in cols:
            if col not in cols:
                raise ValueError(f"Column '{col}' does not exist in the DataFrame.")
            true_count = df[col].sum()
            return true_count

    def plot_failure_ranges_with_comparison(self, df, failure_col, specific_failure_name):
        """
        Visualize and compare the ranges of operational parameters during general 
        machine failures and a specific type of failure.

        Parameters:
        - df (pd.DataFrame): The original DataFrame containing all failures.
        - specific_failure_df (pd.DataFrame): The DataFrame filtered for a specific type of failure.
        - specific_failure_name (str): Name of the specific failure type (e.g., "Tool Wear Failure").
        """
        # Filter the DataFrame for rows where machine failures occurred
        failure_df = df[df['Machine failure'] == True]
        specific_failure_df = failure_df[failure_df[failure_col] == True]

        # Columns to analyze
        columns_to_plot = [
            'Rotational speed [rpm]', 
            'Torque [Nm]', 
            'Process temperature [K]', 
            'Air temperature [K]'
        ]

        # Set up the plot grid
        plt.figure(figsize=(16, 12))
        for i, col in enumerate(columns_to_plot, start=1):
            plt.subplot(2, 2, i)

            # Plot general machine failure distribution
            sns.histplot(
                failure_df[col], kde=True, bins=20, color='skyblue', 
                edgecolor='black', label='General Failures', alpha=0.6
            )

            # Overlay specific failure distribution
            sns.histplot(
                specific_failure_df[col], kde=True, bins=20, color='orange', 
                edgecolor='black', label=specific_failure_name, alpha=0.6
            )

            plt.title(f'Distribution of {col} During Failures', fontsize=14)
            plt.xlabel(col, fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Adjust layout for the plots
        plt.tight_layout()
        plt.show()

    def plot_failure_ranges_with_comparison_to_all(
        self, df, failure_col_1, specific_failure_name_1, 
        failure_col_2, specific_failure_name_2, 
        failure_col_3, specific_failure_name_3, 
        failure_col_4, specific_failure_name_4, 
        failure_col_5, specific_failure_name_5
        ):
        """
        Visualise and compare the ranges of operational parameters during general 
        machine failures and a specific type of failure.

        Parameters:
        - df (pd.DataFrame): The original DataFrame containing all failures.
        - specific_failure_df (pd.DataFrame): The DataFrame filtered for a specific type of failure.
        - specific_failure_name (str): Name of the specific failure type (e.g., "Tool Wear Failure").
        """
        # Filter the DataFrame for rows where machine failures occurred
        failure_df = df[df['Machine failure'] == True]
        specific_failure_df_1 = failure_df[failure_df[failure_col_1] == True]
        specific_failure_df_2 = failure_df[failure_df[failure_col_2] == True]
        specific_failure_df_3 = failure_df[failure_df[failure_col_3] == True]
        specific_failure_df_4 = failure_df[failure_df[failure_col_4] == True]
        specific_failure_df_5 = failure_df[failure_df[failure_col_5] == True]

        # Columns to analyze
        columns_to_plot = [
            'Rotational speed [rpm]', 
            'Torque [Nm]', 
            'Process temperature [K]', 
            'Air temperature [K]'
        ]

        # Set up the plot grid
        plt.figure(figsize=(16, 12))
        for i, col in enumerate(columns_to_plot, start=1):
            plt.subplot(2, 2, i)

            # Plot general machine failure distribution
            sns.histplot(
                failure_df[col], kde=True, bins=20, color='skyblue', 
                edgecolor='black', label='All Failures', alpha=0.6
            )

            # Overlay specific failure distribution
            sns.histplot(
                specific_failure_df_1[col], kde=True, bins=20, color='orange', 
                edgecolor='black', label=specific_failure_name_1, alpha=0.6
            )

            sns.histplot(
                specific_failure_df_2[col], kde=True, bins=20, color='red', 
                edgecolor='black', label=specific_failure_name_2, alpha=0.6
            )

            sns.histplot(
                specific_failure_df_3[col], kde=True, bins=20, color='green', 
                edgecolor='black', label=specific_failure_name_3, alpha=0.6
            )

            sns.histplot(
                specific_failure_df_4[col], kde=True, bins=20, color='pink', 
                edgecolor='black', label=specific_failure_name_4, alpha=0.6
            )
            sns.histplot(
                specific_failure_df_5[col], kde=True, bins=20, color='yellow', 
                edgecolor='black', label=specific_failure_name_5, alpha=0.6
            )

    def rot_speed_failure_comparison(self, df, failure_col_1, specific_failure_name_1, failure_col_2, specific_failure_name_2, failure_col_3, specific_failure_name_3, failure_col_4, specific_failure_name_4, failure_col_5, specific_failure_name_5,):
        """
        Visualise and compare the ranges of operational parameters during general 
        machine failures and a specific type of failure.

        Parameters:
        - df (pd.DataFrame): The original DataFrame containing all failures.
        - specific_failure_df (pd.DataFrame): The DataFrame filtered for a specific type of failure.
        - specific_failure_name (str): Name of the specific failure type (e.g., "Tool Wear Failure").
        """
        # Filter the DataFrame for rows where machine failures occurred
        failure_df = df[df['Machine failure'] == True]
        specific_failure_df_1 = failure_df[failure_df[failure_col_1] == True]
        specific_failure_df_2 = failure_df[failure_df[failure_col_2] == True]
        specific_failure_df_3 = failure_df[failure_df[failure_col_3] == True]
        specific_failure_df_4 = failure_df[failure_df[failure_col_4] == True]
        specific_failure_df_5 = failure_df[failure_df[failure_col_5] == True]

        # Columns to analyze
        columns_to_plot = [
            'Rotational speed [rpm]', 
        ]

        # Set up the plot grid
        plt.figure(figsize=(16, 12))
        for i, col in enumerate(columns_to_plot, start=1):
            plt.subplot(2, 2, i)

            # Plot general machine failure distribution
            sns.histplot(
                failure_df[col], kde=True, bins=20, color='skyblue', 
                edgecolor='black', label='All Failures', alpha=0.6
            )

            # Overlay specific failure distribution
            sns.histplot(
                specific_failure_df_1[col], kde=True, bins=20, color='orange', 
                edgecolor='black', label=specific_failure_name_1, alpha=0.6
            )

            sns.histplot(
                specific_failure_df_2[col], kde=True, bins=20, color='red', 
                edgecolor='black', label=specific_failure_name_2, alpha=0.6
            )

            sns.histplot(
                specific_failure_df_3[col], kde=True, bins=20, color='green', 
                edgecolor='black', label=specific_failure_name_3, alpha=0.6
            )

            sns.histplot(
                specific_failure_df_4[col], kde=True, bins=20, color='pink', 
                edgecolor='black', label=specific_failure_name_4, alpha=0.6
            )
            sns.histplot(
                specific_failure_df_5[col], kde=True, bins=20, color='yellow', 
                edgecolor='black', label=specific_failure_name_5, alpha=0.6
            )



            plt.title(f'Distribution of {col} During Failures', fontsize=14)
            plt.xlabel(col, fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Adjust layout for the plots
        plt.tight_layout()
        plt.show()

        
def plot_failure_histogram(df_twf, df_hdf, df_pwf, df_osf, df_rnf, total_failures):
    """Creates histogram pltos of all dataframes input

    Args:
        df_twf : Dataframe of only Tool wear failure entries
        df_hdf : Dataframe of only Heat dissapation failure entries
        df_pwf : Dataframe of only Power failure entries
        df_osf : Dataframe of only Heat overstrain failure entries
        df_rnf : Dataframe of only Random failure entries
        total_failures : A dataframe of all failure entries
    """    
    # Prepare data for the histogram
    failure_types = ['Tool Wear Failure', 'Heat Dissipation Failure', 'Power Failure', 
                     'Heat Overstrain Failure', 'Random Failure']
    failure_counts = [
        df_twf.shape[0], 
        df_hdf.shape[0], 
        df_pwf.shape[0], 
        df_osf.shape[0], 
        df_rnf.shape[0]
    ]
    failure_percentages = [round(count / total_failures * 100, 2) for count in failure_counts]
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the histogram
    ax.bar(failure_types, failure_counts, color='skyblue', edgecolor='black')

    # Add labels and title
    for i, count in enumerate(failure_counts):
        ax.text(i, count + 1, f'{count} ({failure_percentages[i]}%)', ha='center', fontsize=12)

    ax.set_xlabel('Failure Type', fontsize=14)
    ax.set_ylabel('Number of Failures', fontsize=14)
    ax.set_title('Number of Failures by Type', fontsize=16)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Show the plot
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()