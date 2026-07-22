# функции

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
        submission=pd.DataFrame({
            'id': ids,
            'label': labels
        })
        submission.to_csv(path, index=False)
        print(f'файл был сохранен в:{path}')
        print(f'распределение {labels.sum()}/{len(labels)} ({(labels.mean() * 100):.2f}%)')

        return submission


    @staticmethod
    def calculate_score(roc_auc):
        score=max(0,roc_auc-0.6)-0.4
        return score

    @staticmethod
    def get_level(score):
        if score>=0.9/0.4:
            return 'great'
        elif score>=0.75/0.3:
            return 'good'
        else: return 'base solution'

    @staticmethod
    def plot_roc_curve(y_true, predictions, labels=None):
        plt.figure(figsize=(10,8))

        if labels is None:
            labels = [f'Model {i+1}' for i in range(len(predictions))]

        for pred, label in zip(predictions, labels):
            fpr, tpr, _ = roc_curve(y_true, pred)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{label} (AUC={roc_auc:.4f})' )

        plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_feature_importance(feature_importance, top_n=15):
        plt.figure(figsize=(10,8))
        top_features=feature_importance.head(top_n)
        plt.barh(top_features['feature'], top_features['importance'])

        plt.xlabel('Importance')
        plt.title(f'Top-{top_n} Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()


