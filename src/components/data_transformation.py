import sys, os
import numpy as np
import pandas as pd
from src import utils

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.exceptions import CustomException
from src.logger import logging


class DataTransformation:

    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path  = test_path
        self.train_df = pd.read_csv(self.train_path)
        self.test_df  = pd.read_csv(self.test_path)
        self.preprocessor_file_path = os.path.join("artifacts", "preprocessor.pkl")

    def data_cleaning_process(self):

        # Creating 3 new columns "date_year", "date_month", "date_day" from "date_time" column
        self.train_df['date_time']  = pd.to_datetime(self.train_df['date_time'])
        self.train_df['date_year']  = self.train_df['date_time'].dt.year
        self.train_df['date_month'] = self.train_df['date_time'].dt.month
        self.train_df['date_day']   = self.train_df['date_time'].dt.day

        self.test_df['date_time']  = pd.to_datetime(self.test_df['date_time'])
        self.test_df['date_year']  = self.test_df['date_time'].dt.year
        self.test_df['date_month'] = self.test_df['date_time'].dt.month
        self.test_df['date_day']   = self.test_df['date_time'].dt.day

        # ------------------------------------ Dropping columns -------------------------------------
        # Dropping columns -> "rain_p_h", "snow_p_h", "dew_point", "date_time"
        self.train_df.drop(columns=['rain_p_h', 'snow_p_h', 'dew_point', 'date_time'], inplace=True)
        self.test_df.drop(columns=['rain_p_h', 'snow_p_h', 'dew_point', 'date_time'], inplace=True)

        # ------------------------------------ Converting "is_holiday" column to numerical ------------------------
        self.train_df['is_holiday'] = np.where(self.train_df['is_holiday'] == "None", 0, 1)
        self.test_df['is_holiday']  = np.where(self.test_df['is_holiday'] == "None", 0, 1)

    def get_proprocessor_obj(self):

        try:
            num_columns = ['humidity', 'wind_speed', 'wind_direction', 'visibility_in_miles', 'temperature', 'clouds_all', 'traffic_volume']
            cat_columns = ['weather_type']

            num_pipeline = Pipeline(steps=[
                ('num_imputer', SimpleImputer(strategy='median')),
                ('num_scaler', StandardScaler())
            ])
            logging.info("Created numerical pipeline.")

            cat_pipeline = Pipeline(steps=[
                ('cat_imputer', SimpleImputer(strategy='most_frequent')),
                ('cat_ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            logging.info("Created categorical pipeline.")

            ct = ColumnTransformer([
                ('num_pip', num_pipeline, num_columns),
                ('cat_pip', cat_pipeline, cat_columns)
            ])
            logging.info("Created column transformer object.")

            return ct

        except Exception as e:
            raise CustomException(e, sys)

    def data_transformation(self):
        try:

            logging.info("Data cleaning process")
            self.data_cleaning_process()

            logging.info("Getting column transformer object.")
            ct = self.get_proprocessor_obj()

            #Seperate independent(input) and dependent(output) variables
            output_column_name = "air_pollution_index"
            train_df_input     = self.train_df.drop(columns=[output_column_name])
            train_df_output    = self.train_df[output_column_name]
            test_df_input      = self.test_df.drop(columns=[output_column_name])
            test_df_output     = self.test_df[output_column_name]

            # Transforming train and test dataframe independent features
            logging.info("Transforming train and test dataframe independent features.")
            train_df_input_transformed = ct.fit_transform(train_df_input)
            test_df_input_transformed  = ct.transform(test_df_input)

            self.train_df = np.c_[ train_df_input_transformed, np.array(train_df_output) ]
            self.test_df  = np.c_[test_df_input_transformed, np.array(test_df_output)]
            logging.info("Data transformation process complete.")

            utils.save_object_as_pkl(self.preprocessor_file_path, ct)
            logging.info("Column transformer object saved as pickle file")

            return (self.train_df, self.test_df, self.preprocessor_file_path)

        except Exception as e:
            raise CustomException(e, sys)


# if __name__ == "__main__":
#     obj = DataTransformation()