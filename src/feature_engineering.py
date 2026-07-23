# src/feature_engineering.py
"""
Модуль для создания новых признаков
ЗАВИСИТ ОТ: pandas, numpy
"""

import pandas as pd
import numpy as np


class FeatureEngineer:
    """Класс для создания новых признаков"""

    def __init__(self, train_df=None):
        """
        Parameters:
        -----------
        train_df : DataFrame, optional
            Обучающая выборка для вычисления статистик
        """
        self.train_df = train_df
        self.statistics = {}

        if train_df is not None:
            self._compute_statistics()

    def _compute_statistics(self):
        """Вычисление статистик для бинарных признаков"""
        if self.train_df is None:
            return

        # Вычисляем медианы для порогов
        self.statistics = {
            'median_sessions': self.train_df['sessions_count'].median(),
            'median_spent': (self.train_df['purchases_count'] *
                             self.train_df['avg_purchase_value']).median(),
            'median_std': self.train_df['session_std'].median()
        }

        print("✅ Вычислены статистики для бинарных признаков")

    def create_features(self, df, is_train=True):
        """
        Создание новых признаков

        Parameters:
        -----------
        df : DataFrame
            Данные для трансформации
        is_train : bool
            Являются ли данные обучающими

        Returns:
        --------
        DataFrame: Данные с новыми признаками
        """
        # Сохраняем id если есть
        if 'id' in df.columns:
            ids = df['id']
            df = df.drop('id', axis=1)

        # 1. Основные взаимодействия
        df['sessions_per_day'] = df['sessions_count'] / (df['active_days'] + 1)
        df['purchases_per_session'] = df['purchases_count'] / (df['sessions_count'] + 1)
        df['purchases_per_day'] = df['purchases_count'] / (df['active_days'] + 1)
        df['total_spent'] = df['purchases_count'] * df['avg_purchase_value']
        df['spent_per_session'] = df['total_spent'] / (df['sessions_count'] + 1)
        df['spent_per_day'] = df['total_spent'] / (df['active_days'] + 1)

        # 2. Композитные признаки
        df['engagement'] = (df['sessions_count'] * df['avg_session_time'] /
                            (df['days_since_last_activity'] + 1))
        df['purchase_rate'] = df['purchases_count'] / (df['sessions_count'] + 1)
        df['value_per_session'] = df['avg_purchase_value'] / (df['avg_session_time'] + 1)

        # 3. Бинарные признаки (используем статистики)
        stats = self.statistics

        df['is_active_recently'] = (df['days_since_last_activity'] <= 7).astype(int)
        df['is_high_engagement'] = (df['sessions_count'] > stats['median_sessions']).astype(int)
        df['is_big_spender'] = (df['total_spent'] > stats['median_spent']).astype(int)
        df['is_consistent'] = (df['session_std'] < stats['median_std']).astype(int)
        df['is_weekend'] = df['is_weekend_user']

        # 4. Логарифмические преобразования
        log_features = [
            'sessions_count', 'avg_session_time', 'days_since_last_activity',
            'purchases_count', 'avg_purchase_value', 'active_days', 'session_std',
            'total_spent', 'spent_per_session'
        ]
        for feature in log_features:
            if feature in df.columns:
                df[f'{feature}_log'] = np.log1p(df[feature])

        # 5. Квадраты признаков
        for feature in ['sessions_count', 'active_days', 'purchases_count']:
            if feature in df.columns:
                df[f'{feature}_sq'] = df[feature] ** 2

        # 6. Категоризация дней без активности
        df['days_category'] = pd.cut(
            df['days_since_last_activity'],
            bins=[-1, 3, 7, 14, 30, 60, float('inf')],
            labels=[0, 1, 2, 3, 4, 5]
        ).astype(int)

        print(f" Создано {df.shape[1]} признаков")

        if 'ids' in locals():
            return df, ids
        return df

    def get_feature_names(self):
        return self.feature_names if hasattr(self, 'feature_names') else []