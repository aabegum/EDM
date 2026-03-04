import pandas as pd
import numpy as np
import logging
import os
from dateutil.relativedelta import relativedelta

# A more robust DatasetCreator class tailored to the specific request

class DatasetFactory:
    def __init__(self, data_input):
        self.data_input = data_input
        self.df = self._load_data()
        self.original_df = self.df.copy()

    def _load_data(self):
        # Allow passing a DataFrame directly
        if isinstance(self.data_input, pd.DataFrame):
            df = self.data_input.copy()
            if 'time' in df.columns:
                df = df.sort_values('time').reset_index(drop=True)
            return df
            
        # Load logic handling parquet or csv
        try:
            if self.data_input.endswith('.parquet'):
                df = pd.read_parquet(self.data_input)
            elif self.data_input.endswith('.csv'):
                df = pd.read_csv(self.data_input, parse_dates=['time'], low_memory=False)
            
            # Ensure sorting
            if 'time' in df.columns:
                df = df.sort_values('time').reset_index(drop=True)
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()

    def create_dataset(self, 
                       horizon='day_ahead',
                       feature_set='standard',
                       split_strategy='chronological',
                       condition='none',
                       target_transform='raw'):
        
        # 1. Reset
        df = self.original_df.copy()

        # 2. Apply Conditions (Scenario 4)
        df = self._apply_condition(df, condition)

        # 3. Apply Transformations (Scenario 5)
        df, target_col = self._apply_transform(df, target_transform)

        # 4. Feature Selection (Scenario 2)
        features = self._select_features(df, feature_set)

        # 5. Create Horizon Targets (Scenario 1)
        # Shift target by horizon length
        h_hours = {'short_term': 1, 'near_term': 6, 'day_ahead': 24, 'week_ahead': 168}.get(horizon, 24)
        if 'city' in df.columns:
            df['target'] = df.groupby('city')[target_col].shift(-h_hours)
        else:
            df['target'] = df[target_col].shift(-h_hours)
        df = df.dropna(subset=['target'])

        # 6. Splitting (Scenario 3)
        X_train, y_train, X_val, y_val, X_test, y_test = self._split_data(df, features, split_strategy)

        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'train_df': df.loc[X_train.index].copy(),
            'val_df': df.loc[X_val.index].copy(),
            'test_df': df.loc[X_test.index].copy(),
            'features': features,
            'horizon_hours': h_hours
        }

    def _apply_condition(self, df, condition):
        if condition == 'ramadan_special' and 'is_ramadan' in df.columns:
            return df[df['is_ramadan'] == 1]
        elif condition == 'holiday_excluded' and 'is_holiday' in df.columns:
            return df[df['is_holiday'] == 0]
        elif condition == 'extreme_weather':
            # Top/Bottom 5% temps
            if 'temperature_2m' in df.columns:
                upper = df['temperature_2m'].quantile(0.95)
                lower = df['temperature_2m'].quantile(0.05)
                return df[(df['temperature_2m'] > upper) | (df['temperature_2m'] < lower)]
        elif condition == 'covid_adjusted':
            # Exclude lockdown period
            mask = ~((df['time'] >= '2020-03-15') & (df['time'] <= '2020-06-01'))
            return df[mask]
        elif condition == 'pre_pandemic':
            return df[df['time'] < '2020-03-15']
        elif condition == 'post_pandemic':
            return df[df['time'] > '2021-12-31'] # or another post-pandemic marker
        elif condition == 'pandemic_only':
            return df[(df['time'] >= '2020-03-15') & (df['time'] <= '2021-12-31')]
        return df

    def _apply_transform(self, df, method):
        target = 'demand'
        if method == 'log':
            df['demand_log'] = np.log1p(df[target])
            return df, 'demand_log'
        elif method == 'differenced':
            df['demand_diff'] = df[target].diff()
            return df.dropna(), 'demand_diff'
        elif method == 'normalized_population' and 'population' in df.columns:
            df['demand_per_capita'] = df[target] / df['population']
            return df, 'demand_per_capita'
        elif method == 'detrended':
             # Simple detrend via rolling mean subtraction (yearly)
             df['trend'] = df[target].rolling(window=24*365, min_periods=1).mean()
             df['demand_detrended'] = df[target] - df['trend']
             return df.dropna(), 'demand_detrended'
        return df, target

    def _select_features(self, df, variant):
        all_cols = df.columns
        base = ['hour', 'day_of_week', 'month', 'is_holiday', 'is_weekend']
        
        # Helper lists
        weather = [c for c in all_cols if any(x in c for x in ['temp', 'humid', 'solar', 'wind', 'cloud', 'radiation'])]
        lags = [c for c in all_cols if 'lag' in c]
        cyclical = [c for c in all_cols if 'sin' in c or 'cos' in c]

        if variant == 'minimal':
            # Approx 25 important ones
            selected = base + ['temperature_2m'] + [c for c in lags if '24h' in c or '168h' in c][:5]
        elif variant == 'standard':
            # ~50 features
            selected = base + weather[:5] + lags[:20] + cyclical
        elif variant == 'extended':
            # All available
            selected = [c for c in all_cols if c not in ['time', 'demand', 'target'] and df[c].dtype in [np.int64, np.int32, np.float64, np.float32]]
        elif variant == 'weather_heavy':
            selected = weather + cyclical + ['is_holiday']
        elif variant == 'demand_heavy':
            selected = lags + cyclical + ['is_holiday']
        else:
            selected = base # Fallback

        # Automatic Anti-Leakage: Filter unsafe lags
        safe_selected = []
        for col in selected:
            is_unsafe = False
            # We assume day_ahead = 24h horizon -> safety means lags >= 25h
            try:
                if 'lag_' in col:
                    lag_val = int(col.split('_lag_')[-1].replace('h', ''))
                    if lag_val < 25:
                        is_unsafe = True
            except: pass
            if not is_unsafe:
                safe_selected.append(col)

        return safe_selected

    def _split_data(self, df, features, strategy):
        # Time-based splits
        total = len(df)
        
        if strategy == 'seasonal_aware':
            # Last full year as test
            max_date = df['time'].max()
            test_start = max_date - relativedelta(years=1)
            val_start = test_start - relativedelta(years=1)
            
            train = df[df['time'] < val_start]
            val = df[(df['time'] >= val_start) & (df['time'] < test_start)]
            test = df[df['time'] >= test_start]
            
        elif strategy == 'recent_bias':
            # 50/25/25
            train_idx = int(total * 0.5)
            val_idx = int(total * 0.75)
            train = df.iloc[:train_idx]
            val = df.iloc[train_idx:val_idx]
            test = df.iloc[val_idx:]
            
        else: # Chronological standard 70/15/15
            if 'time' in df.columns:
                unique_times = np.sort(df['time'].unique())
                train_time = unique_times[int(len(unique_times) * 0.7)]
                val_time = unique_times[int(len(unique_times) * 0.85)]
                train = df[df['time'] < train_time]
                val = df[(df['time'] >= train_time) & (df['time'] < val_time)]
                test = df[df['time'] >= val_time]
            else:
                train_idx = int(total * 0.7)
                val_idx = int(total * 0.85)
                train = df.iloc[:train_idx]
                val = df.iloc[train_idx:val_idx]
                test = df.iloc[val_idx:]

        return (train[features], train['target'],
                val[features], val['target'],
                test[features], test['target'])

if __name__ == "__main__":
    # Test execution
    print("DatasetFactory ready.")
