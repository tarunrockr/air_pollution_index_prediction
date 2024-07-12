import os, sys
import numpy as np
import pandas as pd
from src.exceptions import CustomException
from src.logger import logging

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

    def __init__(self):
        self.model_file_path = os.path.join('artifacts', 'model.pkl')

    def train_model(self, train_array, test_array):

        try:
            # print("Train df: \n", pd.DataFrame(train_array))
            logging.info("Splitting independent and dependent variables.")
            X_train, X_test, y_train, y_test = ( train_array[:,:-1], test_array[:,:-1], train_array[:,-1], test_array[:,-1] )

            models = {
                "KNN": KNeighborsRegressor(),
                "DTR": DecisionTreeRegressor(),
                "RFR": RandomForestRegressor(),
                "ABR": AdaBoostRegressor(),
                "SVR": SVR(),
                "LR": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "CB": CatBoostRegressor(),
                "XGB": XGBRegressor()
            }

            logging.info("Evaluating best model")




        except Exception as e:
            raise CustomException(e, sys)

