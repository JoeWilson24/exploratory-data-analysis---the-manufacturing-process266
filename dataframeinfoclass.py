import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import statsmodels.graphics.gofplots as smg
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import normaltest
from scipy import stats

class DataFrameInfo:
    def __init__(self, df):
        self.df = df

    def describe_columns(self):
        """Describe all columns to check their data types."""
        return self.df.dtypes

    def extract_statistics(self):
        """Extract median, standard deviation, and mean for numeric columns."""
        stats = self.df.describe().T[['mean', 'std']].copy()
        stats['median'] = self.df.median(numeric_only=True)
        return stats

    def count_distinct(self):
        """Count distinct values in each column."""
        return self.df.nunique()

    def get_shape(self):
        """Return the shape of the DataFrame."""
        return self.df.shape

    def count_nulls(self):
        """Generate a count and percentage count of NULL values in each column."""
        null_counts = self.df.isnull().sum()
        null_percentage = (null_counts / len(self.df)) * 100
        return pd.DataFrame({'null_count': null_counts, 'null_percentage': null_percentage})

    def top_n_categories(self, column, n=5):
        """Show the top N most common values in a categorical column."""
        if column in self.df.select_dtypes(include='object').columns:
            return self.df[column].value_counts().head(n)
        else:
            raise ValueError(f"Column '{column}' is not categorical.")

    def get_summary(self):
        """Print a summary of the DataFrame including basic info."""
        return {
            "Shape": self.get_shape(),
            "Column Data Types": self.describe_columns(),
            "Null Values": self.count_nulls()
        }