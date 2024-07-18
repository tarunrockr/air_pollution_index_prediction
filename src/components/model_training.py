import os, sys
import numpy as np
import pandas as pd
from src.exceptions import CustomException
from src.logger import logging
from src import utils

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

class ModelTrainer:

    def __init__(self, train_array, test_array):
        self.model_file_path = os.path.join('artifacts', 'model.pkl')
        self.train_array = train_array
        self.test_array  = test_array

    def train_model(self):

        try:
            logging.info("Splitting independent and dependent variables.")
            X_train, X_test, y_train, y_test = ( self.train_array[:,:-1], self.test_array[:,:-1], self.train_array[:,-1], self.test_array[:,-1] )

            models = {
                "KNN": KNeighborsRegressor(),
                "DTR": DecisionTreeRegressor(),
                "RFR": RandomForestRegressor(),
                "ABR": AdaBoostRegressor(),
                "SVR": SVR(),
                "LR":  LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "CB":  CatBoostRegressor(),
                "XGB": XGBRegressor()
            }

            logging.info("Evaluating best model")
            model_dict = utils.model_evaluation(X_train, X_test, y_train, y_test, models)
            r2_scr_df = pd.DataFrame(model_dict).sort_values(by='R2 Score', ascending=False)

            best_model_name = r2_scr_df['Model Name'][0]
            best_r2_score   = r2_scr_df['R2 Score'][0]
            best_model      = models[best_model_name]
            print("Model DF\n", r2_scr_df)

            # Saving best model as pickle file
            utils.save_object_as_pkl(self.model_file_path ,best_model)

        except Exception as e:
            raise CustomException(e, sys)


