# src/model_training.py
"""
Модуль для обучения моделей
ЗАВИСИТ ОТ: sklearn, numpy
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import joblib
import os
import warnings

warnings.filterwarnings('ignore')


class ModelTrainer:
    """Класс для обучения моделей"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.weights = {}

        # Инициализация моделей
        self._init_models()

    def _init_models(self):
        """Инициализация моделей с гиперпараметрами"""
        self.models = {
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=500,
                learning_rate=0.03,
                max_depth=4,
                min_samples_split=30,
                min_samples_leaf=15,
                subsample=0.7,
                random_state=self.random_state
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features='sqrt',
                random_state=self.random_state,
                n_jobs=-1
            ),
            'logistic_regression': LogisticRegression(
                C=0.5,
                max_iter=1000,
                random_state=self.random_state,
                class_weight='balanced'
            )
        }
        print("✅ Модели инициализированы")

    def train_models(self, X_train, y_train, X_val, y_val):
        """
        Обучение всех моделей

        Returns:
        --------
        dict: Результаты обучения
        """
        print("\n=== ОБУЧЕНИЕ МОДЕЛЕЙ ===")

        for name, model in self.models.items():
            print(f"\nОбучаем {name}...")

            # Обучение
            model.fit(X_train, y_train)

            # Предсказание на валидации
            y_pred = model.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, y_pred)

            # Сохраняем результаты
            self.results[name] = {
                'model': model,
                'val_auc': val_auc,
                'predictions': y_pred
            }

            print(f"  Val ROC-AUC: {val_auc:.4f}")

        # Вычисляем веса для ансамбля
        self._compute_ensemble_weights()

        return self.results

    def _compute_ensemble_weights(self):
        """Вычисление весов для ансамбля"""
        aucs = {name: res['val_auc'] for name, res in self.results.items()}
        total = sum(aucs.values())
        self.weights = {name: auc / total for name, auc in aucs.items()}

        print("\n=== ВЕСА ДЛЯ АНСАМБЛЯ ===")
        for name, weight in self.weights.items():
            print(f"  {name}: {weight:.3f}")

    def ensemble_predict(self, X):
        """
        Ансамблевое предсказание

        Parameters:
        -----------
        X : array
            Данные для предсказания

        Returns:
        --------
        array: Предсказанные вероятности
        """
        predictions = []

        for name, res in self.results.items():
            pred = res['model'].predict_proba(X)[:, 1]
            predictions.append(pred * self.weights[name])

        return np.sum(predictions, axis=0)

    def find_optimal_threshold(self, y_true, y_pred):
        """
        Поиск оптимального порога

        Parameters:
        -----------
        y_true : array
            Истинные значения
        y_pred : array
            Предсказанные вероятности

        Returns:
        --------
        float: Оптимальный порог
        """
        from sklearn.metrics import f1_score

        thresholds = np.linspace(0.1, 0.9, 50)
        best_threshold = 0.5
        best_score = 0

        for threshold in thresholds:
            y_pred_binary = (y_pred >= threshold).astype(int)
            score = f1_score(y_true, y_pred_binary)

            if score > best_score:
                best_score = score
                best_threshold = threshold

        print(f"✅ Оптимальный порог: {best_threshold:.3f}")
        return best_threshold

    def save_models(self, path='models/'):
        """Сохранение моделей"""
        os.makedirs(path, exist_ok=True)

        for name, model in self.models.items():
            joblib.dump(model, f'{path}{name}.pkl')

        joblib.dump(self.weights, f'{path}ensemble_weights.pkl')
        print(f"✅ Модели сохранены в {path}")