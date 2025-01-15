class DataTransform:
    #def __init__ (self, df):
        #self.df = df

    def convert_to_booleen(self, df):
        """Converts values in selected columns into booleen type values, True and False

        Args:
            df : dataframe

        Returns:
           df
        """        
        bool_cols = ['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF','RNF'] 
        for col in bool_cols:
            df[col] = df[col].astype(bool)
        return df