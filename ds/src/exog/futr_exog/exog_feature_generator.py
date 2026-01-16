# exog_feature_generator.py
"""
外生変数生成ロジックをカプセル化するクラス。
日付範囲、カレンダー、トレンド、周期、イベント、ドメイン特化の特徴量を生成。
"""

import pandas as pd
import numpy as np
import jpholiday
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class ExogFeatureGenerator:
    """ロト予測用の外生変数（日時特徴量）を生成するクラス"""
    
    def __init__(self, start_date='1994-10-01', years_ahead=10, freq='W-WED', max_fourier_k=5, max_freq=100):
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp.now() + pd.DateOffset(years=years_ahead)
        self.freq = freq
        self.max_fourier_k = max_fourier_k
        self.max_freq = max_freq
        logger.info(f"ジェネレーター初期化: {self.start_date.date()} ～ {self.end_date.date()}, freq={self.freq}")

    def _create_date_range(self):
        """日付範囲（毎週水曜日）を生成"""
        dates = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq=self.freq
        )
        logger.info(f"日付範囲生成: {len(dates)}レコード")
        return pd.DataFrame({'ds': dates})

    def _create_calendar_features(self, df):
        """基本カレンダー特徴量"""
        df['year'] = df['ds'].dt.year
        df['month'] = df['ds'].dt.month
        df['day'] = df['ds'].dt.day
        df['quarter'] = df['ds'].dt.quarter
        df['week'] = df['ds'].dt.isocalendar().week.astype(int)
        df['dayofweek'] = df['ds'].dt.dayofweek
        df['dayofyear'] = df['ds'].dt.dayofyear
        df['week_of_month'] = ((df['day'] - 1) // 7) + 1
        
        df['is_month_start'] = df['ds'].dt.is_month_start.astype(int)
        df['is_leap_year'] = df['ds'].dt.is_leap_year.astype(int)
        df['is_month_end'] = df['ds'].dt.is_month_end.astype(int)
        df['days_in_month'] = df['ds'].dt.days_in_month
        df['days_to_month_end'] = df['days_in_month'] - df['day']
        df['days_since_month_start'] = df['day'] - 1
        df['days_in_year'] = 365 + df['is_leap_year']
        df['days_to_year_end'] = df['days_in_year'] - df['dayofyear']
        df['month_progress'] = df['day'] / df['days_in_month']
        df['year_progress'] = df['dayofyear'] / df['days_in_year']
        
        # オリジナルの残りのフラグ
        df['is_quarter_start'] = df['ds'].dt.is_quarter_start.astype(int)
        df['is_quarter_end'] = df['ds'].dt.is_quarter_end.astype(int)
        df['is_year_start'] = df['ds'].dt.is_year_start.astype(int)
        df['is_year_end'] = df['ds'].dt.is_year_end.astype(int)
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        df['is_weekday'] = (df['dayofweek'] < 5).astype(int)
        
        return df

    def _create_cyclical_features(self, df):
        """周期的Sin/Cos特徴量"""
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
        df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25)
        df['week_sin'] = np.sin(2 * np.pi * df['week'] / 52)
        df['week_cos'] = np.cos(2 * np.pi * df['week'] / 52)
        df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
        return df

    def _create_trend_features(self, df):
        """トレンド特徴量"""
        base_date = df['ds'].min()
        df['trend_linear'] = (df['ds'] - base_date).dt.days
        df['trend_squared'] = df['trend_linear'] ** 2
        df['trend_cubic'] = df['trend_linear'] ** 3
        df['trend_log'] = np.log1p(df['trend_linear'])
        df['trend_sqrt'] = np.sqrt(df['trend_linear'])
        df['trend_normalized'] = df['trend_linear'] / df['trend_linear'].max()
        return df

    def _create_fourier_features(self, df):
        """フーリエ項（k=1～max_k）"""
        t = np.arange(len(df))
        # 年周期（52週）
        for k in range(1, self.max_fourier_k + 1):
            df[f'fourier_year_sin_{k}'] = np.sin(2*np.pi*k*t/52)
            df[f'fourier_year_cos_{k}'] = np.cos(2*np.pi*k*t/52)
        # 月周期（12ヶ月）
        for k in range(1, self.max_fourier_k + 1):
            df[f'fourier_month_sin_{k}'] = np.sin(2*np.pi*k*df['month']/12)
            df[f'fourier_month_cos_{k}'] = np.cos(2*np.pi*k*df['month']/12)
        return df

    def _create_event_features(self, df):
        """イベント・祝日特徴量"""
        df['is_holiday'] = df['ds'].apply(lambda x: 1 if jpholiday.is_holiday(x) else 0)
        df['is_before_holiday'] = df['is_holiday'].shift(-1, fill_value=0)
        df['is_after_holiday'] = df['is_holiday'].shift(1, fill_value=0)
        
        df['is_golden_week'] = (((df['month'] == 4) & (df['day'] >= 29)) | ((df['month'] == 5) & (df['day'] <= 5))).astype(int)
        df['is_obon'] = ((df['month'] == 8) & (df['day'] >= 13) & (df['day'] <= 16)).astype(int)
        df['is_new_year_holiday'] = (((df['month'] == 12) & (df['day'] >= 29)) | ((df['month'] == 1) & (df['day'] <= 3))).astype(int)
        df['is_silver_week'] = ((df['month'] == 9) & (df['day'] >= 19) & (df['day'] <= 23)).astype(int)
        
        df['is_payday_25th'] = (df['day'] == 25).astype(int)
        df['is_payday_month_end'] = df['is_month_end']
        df['is_bonus_month'] = df['month'].isin([6, 12]).astype(int)
        df['is_fiscal_year_end'] = (df['month'] == 3).astype(int)
        df['is_fiscal_quarter_end'] = df['month'].isin([3,6,9,12]).astype(int)
        
        df['fiscal_year_jp'] = np.where(df['month'] >= 4, df['year'], df['year'] - 1)
        df['fiscal_quarter_jp'] = ((df['month'] - 4) % 12 // 3) + 1
        df['fiscal_month_jp'] = np.where(df['month'] >= 4, df['month'] - 3, df['month'] + 9)
        
        return df

    def _create_bingo5_features(self, df):
        """Bingo5特有の特徴量"""
        df['draw_number'] = range(len(df))
        df['draw_index_log'] = np.log1p(df['draw_number'])
        
        df['nth_wednesday_of_month'] = df['week_of_month']
        
        df['is_last_wednesday_of_month'] = ((df['dayofweek'] == 2) & ((df['day'] + 7) > df['days_in_month'])).astype(int)
        
        df['is_first_draw_of_month'] = (df['nth_wednesday_of_month'] == 1).astype(int)
        df['is_last_draw_of_month'] = df['is_last_wednesday_of_month']
        
        df['draws_this_month'] = df.groupby(
            [df['ds'].dt.year, df['ds'].dt.month]
        )['nth_wednesday_of_month'].transform('max')
        
        df['is_year_first_draw'] = (
            (df['ds'].dt.dayofyear <= 7) & (df['dayofweek'] == 2)
        ).astype(int)
        return df

    def _create_frequency_features(self, df):
        """周期変数生成"""
        n = len(df)
        for freq in range(2, self.max_freq + 1):
            df[f'freq{freq}'] = (np.arange(n) % freq) + 1
        return df

    def generate(self) -> pd.DataFrame:
        """全ての特徴量を統合的に生成し、結果のDataFrameを返す"""
        logger.info("特徴量生成開始...")
        df = self._create_date_range()
        
        df = self._create_calendar_features(df)
        df = self._create_cyclical_features(df)
        df = self._create_trend_features(df)
        df = self._create_fourier_features(df)
        df = self._create_event_features(df)
        df = self._create_bingo5_features(df)
        df = self._create_frequency_features(df)
        
        # 'futr_' 接頭辞付与
        rename_map = {col: f"futr_{col}" for col in df.columns if col != 'ds'}
        df = df.rename(columns=rename_map)
        logger.info(f"特徴量生成完了。カラム数: {len(df.columns)}")
        
        return df