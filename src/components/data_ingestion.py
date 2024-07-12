import os, sys
import pandas as pd
import numpy as np
from src.exceptions import CustomException
from src.logger import logging
from src import utils

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer

class DataIngestion:

    def __init__(self):
        self.train_data_path = os.path.join("artifacts","train.csv")
        self.test_data_path  = os.path.join("artifacts", "test.csv")
        self.raw_data_path   = os.path.join("artifacts", "raw.csv")

    def handling_outliers(self, df):

        ###################################### Train Dataframe outliers handling #################################################
        # ------------------------- Handling "humidity" columns outliers --------------------------
        out = utils.detect_outliers_by_z_score(df['humidity'])
        # Capping "humidity" column outliers by upper_limit and lower_limit
        upper_limit = df['humidity'].mean() + (3 * df['humidity'].std())
        lower_limit = df['humidity'].mean() - (3 * df['humidity'].std())
        df['humidity'] = np.where(df['humidity'] > upper_limit, upper_limit, np.where(df['humidity'] < lower_limit, lower_limit, df['humidity']))

        # ------------------------- Handling "wind_speed" columns outliers --------------------------
        outlier_data, iqr_upper_limit, iqr_lower_limit = utils.outliers_by_iqr(df['wind_speed'])
        # Capping outliers of "wind_speed" column
        df['wind_speed'] = np.where((df['wind_speed'] > iqr_upper_limit), iqr_upper_limit,np.where(df['wind_speed'] < iqr_lower_limit, iqr_lower_limit,df['wind_speed']))

        # ------------------------- Handling "temperature" columns outliers --------------------------
        outlier_data, iqr_upper_limit, iqr_lower_limit = utils.outliers_by_iqr(df['temperature'])
        # Capping outliers of "temperature" column
        df['temperature'] = np.where((df['temperature'] > iqr_upper_limit), iqr_upper_limit,np.where(df['temperature'] < iqr_lower_limit, iqr_lower_limit,df['temperature']))

    def initiate_data_ingestion(self):

        logging.info("Data ingestion process started.")
        try:
            logging.info("Reading the csv file.")
            df = pd.read_csv(r"notebook\data\train.csv")
            # train_df = pd.read_csv(r"notebook\data\train.csv")
            # test_df  = pd.read_csv(r"notebook\data\test.csv")

            # Outlier detection and handling
            self.handling_outliers(df)

            os.makedirs(os.path.dirname(self.raw_data_path), exist_ok=True)
            # df = pd.concat([train_df, test_df], axis=0)
            df.to_csv(self.raw_data_path, index=False, header=True)

            logging.info("Separating train and test dataset")
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

            train_df.to_csv(self.train_data_path, index=False, header=True)
            test_df.to_csv(self.test_data_path, index=False, header=True)
            logging.info(f"Train and test files are saved to '{os.path.dirname(self.raw_data_path)}'folder")
            logging.info("Ingestion process is complete.")

            return (
                self.train_data_path,
                self.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)

# if __name__ == "__main__":
#     obj = DataIngestion()
#     train_path, test_path = obj.initiate_data_ingestion()
# 
#     transformation_obj    = DataTransformation(train_path, test_path)
#     train_arr, test_arr,_ =  transformation_obj.data_transformation()
#
#     trainer_obj = ModelTrainer()
#     trainer_obj.train_model(train_arr, test_arr)