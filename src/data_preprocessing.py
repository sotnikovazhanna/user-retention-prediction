import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selectiion import train_test_split
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        self.scaler = RobustScaler()
        self.is_fitted = False
