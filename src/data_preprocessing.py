import os

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

    def load_data(self, train_path, test_path):
        train = pd.read_csv(train_path, sep=',', decimal='.', encoding='utf-8')
        test = pd.read_csv(test_path, sep=',', decimal='.', encoding='utf-8')

        print(f'загружена обучающая выборка: {train.shape}')
        print(f'загружена тестовая выборка: {test.shape}')

        return train, test

    def split_data(self,X, y, test_size=0.2, random_state=42):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state,stratify=y)
        print(f"\nРазделение данных:")
        print(f"  Train: {X_train.shape[0]} samples")
        print(f"  Val: {X_val.shape[0]} samples")
        print(f"  Train retention: {y_train.mean():.3f}")
        print(f"  Val retention: {y_val.mean():.3f}")

        return X_train, X_val, y_train, y_val

    def scale_data(self, X_train, X_val=None, X_test=None):
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.is_fitted = True

        result=[X_train_scale]

        if X_val is not None:
            X_test_scaled = self.scaler.transform(X_test)
            result.append(X_test_scaled)

        print('Данные масштабированы')

        if len(result) == 1:
            return result[0]
        return tuple(result)

    def save_scaler(self, path='models/scaler.pkl'):
        import joblib
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.scaler, path)
        print(f'scaler сохранён в {path}')

    def load_scaler(self, path='models/scaler.pkl'):
        import joblib
        self.scaler = joblib.load(path)
        self.is_fitted = True
        print(f'scaler загружен из {path}')


