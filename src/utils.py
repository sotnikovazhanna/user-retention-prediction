# функции
from inspect import Parameter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, roc_auc_score
import os

class Utils:
    @staticmethod
    def create_submission:
        """
                Создание файла для сабмита

                Parameters:
                -----------
                ids : array
                    ID пользователей
                predictions : array
                    Предсказанные вероятности
                threshold : float
                    Порог классификации
                path : str
                    Путь для сохранения
                """

        os.makedirs(os.path.dirname(path), exist_ok=True)
        labels=(prediction>=threshold).astype(int)


