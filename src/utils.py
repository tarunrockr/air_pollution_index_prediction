import pandas as pd
import numpy as np
import pickle
from src.logger import logging
import os, sys


# Detecting outliers in column by z_score method as it close to normal distribution
def detect_outliers_by_z_score(data):
    outliers = []
    threshold = 3
    std_val  = data.std()
    mean_val = data.mean()

    for value in data:
        z_score = (value-mean_val)/std_val
        if abs(z_score) > 3:
            outliers.append(value)
    return outliers


# Detecting outliers in column by IQR method as it is not normally distributed
def outliers_by_iqr(data):
    outliers = []

    percentile_25 = np.quantile(data, 0.25)
    percentile_75 = np.quantile(data, 0.75)
    IQR = percentile_75 - percentile_25
    upper_limit = percentile_75 + (1.5 * IQR)
    lower_limit = percentile_25 - (1.5 * IQR)

    for item in data:
        if (item > upper_limit) or (item < lower_limit):
            outliers.append(item)

    return outliers, upper_limit, lower_limit

def save_object_as_pkl(file_path, object):

    dir_name = os.path.dirname(file_path)
    logging.info(f"Dirname: {os.path.dirname(file_path)}")
    # Creating artifacts directory if not exists
    os.makedirs(dir_name, exist_ok=True)

    with open(file_path, 'wb') as file:
        pickle.dump(object, file)

    head, tail = os.path.split(file_path)
    logging.info(f"Pickle file '{tail}' saved at path: {dir_name}")