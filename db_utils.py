import yaml
from sqlalchemy import create_engine
import pandas as pd 

class RDSDatabaseConnector:
    def __init__(self, credentials):
        self.credentials = credentials
        self.engine = self.db_engine_init() 
    
    def db_engine_init (self):
        RDS_HOST = self.credentials["RDS_HOST"]
        RDS_PASSWORD = self.credentials["RDS_PASSWORD"]
        RDS_USER = self.credentials["RDS_USER"]
        RDS_DATABASE = self.credentials["RDS_DATABASE"]
        RDS_PORT = self.credentials["RDS_PORT"]

        engine = create_engine(f"postgresql://{RDS_USER}:{RDS_PASSWORD}@{RDS_HOST}:{RDS_PORT}/{RDS_DATABASE}")
        engine.execution_options(isolation_level='AUTOCOMMIT').connect()
        return engine

    def extract_data(self):
        if self.engine:
                df = pd.read_sql_table("failure_data", con=self.engine) 
                return df
        else:
            print("No engine available. Please initialise the engine first.")
            return None 

    def save_data_to_csv(self, file_path):

        df = self.extract_data()
        if df is not None:
            df.to_csv(file_path, index=False) 
            print(f"Data saved to {file_path}")
        else:
            print("Failed to extract data. Data not saved.")


def loadcredentials():
    with open(f"credentials.yaml", "r") as f:
        credentials = yaml.safe_load(f)
    return credentials

if __name__ == "__main__":
    credentials = loadcredentials()
    connector = RDSDatabaseConnector(credentials)
    engine = connector.engine 

    if engine:
        df = pd.read_sql("SELECT * FROM failure_data", con=engine) 
        print(df) 
        file_path = "/Users/joe/Documents/AiCore/VS Code/DataAnalysisProject/failure_data.csv"
        connector.save_data_to_csv(file_path)