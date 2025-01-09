class DataTransform:
    #def __init__ (self, df):
        #self.df = df

    def convert_to_booleen(self, df):
        bool_cols = ['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF','RNF'] 
        for col in bool_cols:
            df[col] = df[col].astype(bool)
        return df