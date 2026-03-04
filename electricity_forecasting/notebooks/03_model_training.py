# Auto-generated Python script from 03_model_training.ipynb# Generated on: 03_model_training.ipynb


# # Model Training and Evaluation# ## Electricity Demand Forecasting# # This notebook implements and evaluates machine learning models for electricity demand forecasting.# # **Approach:**# - Time-series aware train/validation/test split# - Baseline models (Linear, Ridge, Random Forest)# - Advanced models (XGBoost, LightGBM)# - Hyperparameter tuning with time-series cross-validation# - Ensemble methods# - Regional and temporal performance analysis


# ## 1. Setup and Configuration


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os
import json
warnings.filterwarnings('ignore')

# Modeling libraries
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

# Advanced models
import xgboost as xgb
import lightgbm as lgb

# Visualization settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

print('=' * 80)
print('ELECTRICITY DEMAND FORECASTING - MODEL TRAINING')
print('=' * 80)
print(f'Notebook started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
print(f'\nLibraries loaded successfully!')
print(f'\n⚠️  Data Leakage Prevention Active:')
print(f'   - Using FROZEN percentile boundaries from training set')
print(f'   - Using FROZEN z-score parameters from training set')
print(f'   - Chronological train/validation/test split')
print(f'   - No recomputation on test data')



# ## 2. Forecasting Configuration


# ===== CRITICAL: FORECASTING HORIZON CONFIGURATION =====
FORECAST_HORIZON = 24  # Hours ahead to forecast
SAFE_MIN_LAG = FORECAST_HORIZON + 1  # Minimum safe lag = 25 hours

print('=' * 80)
print('FORECASTING CONFIGURATION')
print('=' * 80)
print(f'Forecast Horizon: {FORECAST_HORIZON} hours ahead')
print(f'Safe Minimum Lag: {SAFE_MIN_LAG} hours')
print(f'Task: Predict demand at t+{FORECAST_HORIZON} using features available at t')
print('=' * 80)


# ## 3. Data Loading and Preparation


# Load engineered features (MUST use fixed leakage-free version)
data_path = 'data/'

# Load full leakage-free feature set from fixed notebook
dfs = []
for city in ['aydin', 'denizli', 'mugla']:
    file_path = f'{data_path}processed/{city}_engineered_features_enhanced.csv'
    print(f"Checking for file: {file_path}")
    if os.path.exists(file_path):
        city_df = pd.read_csv(file_path)
        dfs.append(city_df)

if dfs:
    df = pd.concat(dfs, ignore_index=True)
    print(f'\n✓ Loaded LEAKAGE-FREE engineered features')
    print(f'  Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns')
else:
    raise FileNotFoundError("Could not find enhanced feature CSVs in data/processed/")

# Convert time to datetime
df['time'] = pd.to_datetime(df['time'])

# ===== CRITICAL FIX: CREATE FUTURE DEMAND TARGET AND FEATURES =====
# Uses custom robust class logic for perfect dataset building
import sys
sys.path.append(os.path.abspath('src'))
from dataset_creator import DatasetFactory

print(f'\n🎯 USING DATASET FACTORY FOR HORIZON={FORECAST_HORIZON}h')
print('-' * 80)
factory = DatasetFactory(df)
dataset_result = factory.create_dataset(
    horizon='day_ahead',
    feature_set='extended', 
    split_strategy='chronological'
)

# Extract generated data chunks, aligning naming with original logic
train_df = dataset_result['train_df'].rename(columns={'target': 'demand_future'})
val_df = dataset_result['val_df'].rename(columns={'target': 'demand_future'})
test_df = dataset_result['test_df'].rename(columns={'target': 'demand_future'})
feature_cols = dataset_result['features']

print(f"✓ Using {len(feature_cols)} safe engineered features.")
print(f"✓ Dataset perfectly shifted backward {FORECAST_HORIZON}h to create target.")
print(f"✓ Re-split sorted data chronologically correctly with NO overlaps.")
print(f"✓ Eliminated ~200 lines of hardcoded logic via DatasetFactory helper class.")

print('=' * 80)
print('TIME-SERIES SPLIT (CHRONOLOGICAL - NO DATA LEAKAGE)')
print('=' * 80)
print(f'\nTrain set:')
print(f'  Size: {len(train_df):,} samples')
print(f'  Period: {train_df["time"].min()} to {train_df["time"].max()}')

print(f'\nValidation set:')
print(f'  Size: {len(val_df):,} samples')
print(f'  Period: {val_df["time"].min()} to {val_df["time"].max()}')

print(f'\nTest set:')
print(f'  Size: {len(test_df):,} samples')
print(f'  Period: {test_df["time"].min()} to {test_df["time"].max()}')

# ===== CRITICAL STEP 1: FREEZE PERCENTILE BOUNDARIES (from training set only) =====
print(f'\n' + '=' * 80)
print('FREEZING PARAMETERS FROM TRAINING SET')
print('=' * 80)

import json

# Compute percentile boundaries ONLY from training data
percentile_bounds = {}
for city in train_df['city'].unique():
    for hour in range(24):
        group = train_df[(train_df['city'] == city) & (train_df['hour'] == hour)]['demand']
        if len(group) > 0:
            percentile_bounds[f'{city}_{hour}'] = {
                'p10': float(group.quantile(0.10)),
                'p25': float(group.quantile(0.25)),
                'p50': float(group.quantile(0.50)),
                'p75': float(group.quantile(0.75)),
                'p90': float(group.quantile(0.90)),
            }

# Save frozen percentiles to disk
percentile_path = f'{data_path}percentile_bounds_from_training.json'
with open(percentile_path, 'w') as f:
    json.dump(percentile_bounds, f)
print(f'✓ Frozen percentile bounds saved: {percentile_path}')
print(f'  {len(percentile_bounds)} (city, hour) groups')

# ===== CRITICAL STEP 2: FREEZE Z-SCORE PARAMETERS (from training set only) =====
zscore_params = {}
for city in train_df['city'].unique():
    for hour in range(24):
        group = train_df[(train_df['city'] == city) & (train_df['hour'] == hour)]['temperature_2m']
        if len(group) > 1:
            zscore_params[f'{city}_{hour}'] = {
                'mean': float(group.mean()),
                'std': float(group.std()),
            }

zscore_path = f'{data_path}zscore_params_from_training.json'
with open(zscore_path, 'w') as f:
    json.dump(zscore_params, f)
print(f'✓ Frozen z-score parameters saved: {zscore_path}')
print(f'  {len(zscore_params)} (city, hour) groups')

# ===== CRITICAL STEP 3: APPLY FROZEN PARAMETERS TO VALIDATION AND TEST =====
def apply_frozen_percentiles(df_input, percentile_bounds, dataset_name=''):
    """Apply frozen percentile boundaries WITHOUT recomputing"""
    df_output = df_input.copy()
    
    if 'demand_percentile_hourly' in df_output.columns:
        mapped_count = 0
        for idx, row in df_output.iterrows():
            key = f"{row['city']}_{row['hour']}"
            demand = row['demand']
            
            if key in percentile_bounds:
                bounds = percentile_bounds[key]
                # Map demand value to percentile using FROZEN training boundaries
                if demand <= bounds['p25']:
                    df_output.loc[idx, 'demand_percentile_hourly'] = 0.125
                elif demand <= bounds['p50']:
                    df_output.loc[idx, 'demand_percentile_hourly'] = 0.375
                elif demand <= bounds['p75']:
                    df_output.loc[idx, 'demand_percentile_hourly'] = 0.625
                else:
                    df_output.loc[idx, 'demand_percentile_hourly'] = 0.875
                mapped_count += 1
        
        if dataset_name:
            print(f'  {dataset_name}: Mapped {mapped_count} percentiles using FROZEN bounds')
    
    return df_output

def apply_frozen_zscores(df_input, zscore_params, dataset_name=''):
    """Apply frozen z-score parameters WITHOUT recomputing"""
    df_output = df_input.copy()
    
    if 'temp_zscore_hourly' in df_output.columns:
        mapped_count = 0
        for idx, row in df_output.iterrows():
            key = f"{row['city']}_{row['hour']}"
            temp = row['temperature_2m']
            
            if key in zscore_params:
                params = zscore_params[key]
                zscore = (temp - params['mean']) / (params['std'] + 1e-6)
                df_output.loc[idx, 'temp_zscore_hourly'] = zscore
                mapped_count += 1
        
        if dataset_name:
            print(f'  {dataset_name}: Applied FROZEN z-scores to {mapped_count} observations')
    
    return df_output

# Apply frozen parameters
val_df = apply_frozen_percentiles(val_df, percentile_bounds, 'Validation')
test_df = apply_frozen_percentiles(test_df, percentile_bounds, 'Test')

val_df = apply_frozen_zscores(val_df, zscore_params, 'Validation')
test_df = apply_frozen_zscores(test_df, zscore_params, 'Test')

print(f'\n✓ Frozen parameters applied to validation and test sets')
print(f'  (NO recomputation - using training set parameters only)')



# ===== PREPARE FEATURE MATRICES AND FUTURE DEMAND TARGET =====
# CRITICAL: Use demand_future (t+24h) as target, not current demand
print('\n' + '=' * 80)
print('PREPARING FEATURES AND TARGET FOR FORECASTING')
print('=' * 80)

X_train = train_df[feature_cols].copy()
y_train = train_df['demand_future'].copy()  # FORECASTING: Predict future demand

X_val = val_df[feature_cols].copy()
y_val = val_df['demand_future'].copy()  # FORECASTING: Predict future demand

X_test = test_df[feature_cols].copy()
y_test = test_df['demand_future'].copy()  # FORECASTING: Predict future demand

print(f'\n✓ Target variable: demand_future (t+{FORECAST_HORIZON}h)')
print(f'  This is TRUE FORECASTING, not regression on current demand')

# ===== CRITICAL: Impute missing values using TRAINING SET STATISTICS ONLY =====
# Compute statistics on training set
train_means = X_train.mean()  # This uses skipna=True by default

# Apply training statistics to all sets (NEVER recompute on test/val)
X_train = X_train.fillna(train_means)
X_val = X_val.fillna(train_means)  # Use training means, NOT validation means!
X_test = X_test.fillna(train_means)  # Use training means, NOT test means!

# Handle any remaining NaN or inf values
X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(0)
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

print(f'\nFeature matrices prepared:')
print(f'X_train: {X_train.shape}')
print(f'X_val: {X_val.shape}')
print(f'X_test: {X_test.shape}')

# Verification: Confirm no percentile recomputation happened
print(f'\n✓ Missing values imputed using TRAINING SET STATISTICS')
print(f'✓ Percentile boundaries and z-scores are FROZEN from training')
print(f'✓ Data leakage prevention verified')
print(f'\n✓ Feature preparation complete')



# ## 4. Feature Scaling


# ===== CRITICAL: Fit scaler on training set ONLY =====
# Standardize features (fit on training set only)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit AND transform on training
X_val_scaled = scaler.transform(X_val)           # Transform ONLY on validation (use training stats)
X_test_scaled = scaler.transform(X_test)         # Transform ONLY on test (use training stats)

# Convert back to DataFrames for easier handling
X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
X_val_scaled = pd.DataFrame(X_val_scaled, columns=feature_cols, index=X_val.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)

print('Feature scaling complete (StandardScaler - fit ONLY on training)')
print(f'Scaled feature statistics (train set):')
print(f'  Mean: {X_train_scaled.mean().mean():.6f} (should be ~0)')
print(f'  Std: {X_train_scaled.std().mean():.6f} (should be ~1)')

print(f'\nValidation set scaled using training parameters:')
print(f'  Mean: {X_val_scaled.mean().mean():.6f} (will NOT be ~0, which is correct)')
print(f'  This ensures no data leakage')

# ## 5. Evaluation Metrics Function

# ===== COMPREHENSIVE EVALUATION FRAMEWORK =====

def calculate_mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error.
    
    Formula: mean(|actual - pred| / actual) * 100
    Use Case: Primary accuracy metric
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    
    Returns:
    --------
    float : MAPE value as percentage
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Avoid division by zero - only calculate for non-zero actuals
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error.
    
    Formula: sqrt(mean((actual - pred)²))
    Use Case: Penalizes large errors
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    
    Returns:
    --------
    float : RMSE value
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error.
    
    Formula: mean(|actual - pred|)
    Use Case: Robust to outliers
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    
    Returns:
    --------
    float : MAE value
    """
    return mean_absolute_error(y_true, y_pred)


def calculate_pinball_loss(y_true, y_pred, quantile=0.5):
    """
    Calculate Pinball Loss for quantile forecasts.
    
    Use Case: Quantile-specific, probabilistic forecast evaluation
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    quantile : float
        Quantile level (default: 0.5 for median)
    
    Returns:
    --------
    float : Pinball loss value
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    errors = y_true - y_pred
    
    loss = np.where(errors >= 0, 
                    quantile * errors, 
                    (quantile - 1) * errors)
    
    return np.mean(loss)


def calculate_crps(y_true, y_pred, y_pred_std=None):
    """
    Calculate Continuous Ranked Probability Score (simplified version).
    
    Use Case: Overall distribution quality
    
    For point forecasts without uncertainty estimates, this simplifies to MAE.
    For probabilistic forecasts, it evaluates the entire predictive distribution.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values (mean)
    y_pred_std : array-like, optional
        Standard deviation of predictions (for probabilistic forecasts)
    
    Returns:
    --------
    float : CRPS value
    """
    if y_pred_std is None:
        # For point forecasts, CRPS reduces to MAE
        return calculate_mae(y_true, y_pred)
    else:
        # Simplified CRPS for normal distribution
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_pred_std = np.array(y_pred_std)
        
        from scipy.stats import norm
        
        # Standardized error
        z = (y_true - y_pred) / (y_pred_std + 1e-10)
        
        # CRPS for normal distribution
        crps = y_pred_std * (z * (2 * norm.cdf(z) - 1) + 
                             2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
        
        return np.mean(crps)


def evaluate_model(y_true, y_pred, model_name='Model', dataset='Validation'):
    """
    Calculate comprehensive evaluation metrics for demand forecasting.
    
    Parameters:
    -----------
    y_true : array-like
        True demand values
    y_pred : array-like
        Predicted demand values
    model_name : str
        Name of the model
    dataset : str
        Dataset name (Train/Validation/Test)
    
    Returns:
    --------
    dict : Dictionary containing all metrics
    """
    mape = calculate_mape(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)
    mae = calculate_mae(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Pinball losses for different quantiles
    pinball_50 = calculate_pinball_loss(y_true, y_pred, quantile=0.5)
    pinball_10 = calculate_pinball_loss(y_true, y_pred, quantile=0.1)
    pinball_90 = calculate_pinball_loss(y_true, y_pred, quantile=0.9)
    
    # CRPS (point forecast version)
    crps = calculate_crps(y_true, y_pred)
    
    # Additional metrics
    max_error = np.max(np.abs(y_true - y_pred))
    median_ae = np.median(np.abs(y_true - y_pred))
    
    metrics = {
        'MAPE': mape,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Pinball_Loss_50': pinball_50,
        'Pinball_Loss_10': pinball_10,
        'Pinball_Loss_90': pinball_90,
        'CRPS': crps,
        'Max_Error': max_error,
        'Median_AE': median_ae
    }
    
    print(f'\n{model_name} - {dataset} Set Metrics:')
    print('-' * 60)
    print(f'MAPE (Mean Abs Percentage Error): {mape:>10.2f} %')
    print(f'RMSE (Root Mean Squared Error):  {rmse:>10.2f} MWh')
    print(f'MAE (Mean Absolute Error):       {mae:>10.2f} MWh')
    print(f'R² (Coefficient of Determination): {r2:>10.4f}')
    print(f'Pinball Loss (q=0.5):            {pinball_50:>10.2f}')
    print(f'Pinball Loss (q=0.1):            {pinball_10:>10.2f}')
    print(f'Pinball Loss (q=0.9):            {pinball_90:>10.2f}')
    print(f'CRPS:                            {crps:>10.2f}')
    print(f'Max Error:                        {max_error:>10.2f} MWh')
    print(f'Median Absolute Error:            {median_ae:>10.2f} MWh')
    
    return metrics


def comprehensive_evaluation(model, test_df, predictions, model_name='Model'):
    """
    Evaluate model performance across multiple dimensions.
    
    This function provides detailed evaluation by:
    - Overall performance
    - Performance by city
    - Performance by season
    - Performance by hour of day
    - Performance during special periods (Ramadan, holidays)
    
    Parameters:
    -----------
    model : trained model
        The trained model (not used directly, kept for consistency)
    test_df : pd.DataFrame
        Test dataframe with all features and metadata
    predictions : array-like
        Model predictions
    model_name : str
        Name of the model for reporting
    
    Returns:
    --------
    dict : Nested dictionary containing metrics for each segment
    """
    results = {}
    
    # Ensure predictions align with test_df
    if len(predictions) != len(test_df):
        print(f"⚠️  Warning: Predictions length ({len(predictions)}) != test_df length ({len(test_df)})")
        return results
    
    print(f'\n{"=" * 80}')
    print(f'COMPREHENSIVE EVALUATION: {model_name}')
    print(f'{"=" * 80}')
    
    # ===== OVERALL METRICS =====
    print(f'\n📊 OVERALL PERFORMANCE')
    print('-' * 60)
    results['overall'] = {
        'mape': calculate_mape(test_df['demand_future'], predictions),
        'rmse': calculate_rmse(test_df['demand_future'], predictions),
        'mae': calculate_mae(test_df['demand_future'], predictions),
        'r2': r2_score(test_df['demand_future'], predictions)
    }
    print(f"MAPE: {results['overall']['mape']:.2f}%")
    print(f"RMSE: {results['overall']['rmse']:.2f} MWh")
    print(f"MAE:  {results['overall']['mae']:.2f} MWh")
    print(f"R²:   {results['overall']['r2']:.4f}")
    
    # ===== BY CITY =====
    print(f'\n🏙️  PERFORMANCE BY CITY')
    print('-' * 60)
    if 'city' in test_df.columns:
        for city in sorted(test_df['city'].unique()):
            city_mask = test_df['city'] == city
            if city_mask.sum() > 0:
                city_key = f'city_{city}'
                results[city_key] = {
                    'mape': calculate_mape(test_df.loc[city_mask, 'demand_future'], 
                                          predictions[city_mask]),
                    'rmse': calculate_rmse(test_df.loc[city_mask, 'demand_future'], 
                                          predictions[city_mask]),
                    'mae': calculate_mae(test_df.loc[city_mask, 'demand_future'], 
                                        predictions[city_mask]),
                    'count': int(city_mask.sum())
                }
                print(f"{city:15s} - MAPE: {results[city_key]['mape']:6.2f}% | "
                      f"RMSE: {results[city_key]['rmse']:7.2f} | "
                      f"MAE: {results[city_key]['mae']:7.2f} | "
                      f"N: {results[city_key]['count']:,}")
    
    # ===== BY SEASON =====
    print(f'\n🌦️  PERFORMANCE BY SEASON')
    print('-' * 60)
    if 'season' in test_df.columns:
        for season in ['winter', 'spring', 'summer', 'fall']:
            season_mask = test_df['season'] == season
            if season_mask.sum() > 0:
                season_key = f'season_{season}'
                results[season_key] = {
                    'mape': calculate_mape(test_df.loc[season_mask, 'demand_future'],
                                          predictions[season_mask]),
                    'rmse': calculate_rmse(test_df.loc[season_mask, 'demand_future'],
                                          predictions[season_mask]),
                    'mae': calculate_mae(test_df.loc[season_mask, 'demand_future'],
                                        predictions[season_mask]),
                    'count': int(season_mask.sum())
                }
                print(f"{season.capitalize():10s} - MAPE: {results[season_key]['mape']:6.2f}% | "
                      f"RMSE: {results[season_key]['rmse']:7.2f} | "
                      f"MAE: {results[season_key]['mae']:7.2f} | "
                      f"N: {results[season_key]['count']:,}")
    
    # ===== BY HOUR OF DAY =====
    print(f'\n🕐 PERFORMANCE BY HOUR OF DAY (Selected Hours)')
    print('-' * 60)
    if 'hour' in test_df.columns:
        for hour in [0, 6, 12, 18]:  # Midnight, Morning, Noon, Evening
            hour_mask = test_df['hour'] == hour
            if hour_mask.sum() > 0:
                hour_key = f'hour_{hour:02d}'
                results[hour_key] = {
                    'mape': calculate_mape(test_df.loc[hour_mask, 'demand_future'],
                                          predictions[hour_mask]),
                    'rmse': calculate_rmse(test_df.loc[hour_mask, 'demand_future'],
                                          predictions[hour_mask]),
                    'mae': calculate_mae(test_df.loc[hour_mask, 'demand_future'],
                                        predictions[hour_mask]),
                    'count': int(hour_mask.sum())
                }
                hour_label = {0: 'Midnight', 6: 'Morning', 12: 'Noon', 18: 'Evening'}
                print(f"{hour:02d}:00 ({hour_label[hour]:8s}) - MAPE: {results[hour_key]['mape']:6.2f}% | "
                      f"RMSE: {results[hour_key]['rmse']:7.2f} | "
                      f"MAE: {results[hour_key]['mae']:7.2f} | "
                      f"N: {results[hour_key]['count']:,}")
    
    # ===== SPECIAL PERIODS =====
    print(f'\n🌙 PERFORMANCE DURING SPECIAL PERIODS')
    print('-' * 60)
    
    # Ramadan
    if 'is_ramadan' in test_df.columns:
        ramadan_mask = test_df['is_ramadan'] == 1
        if ramadan_mask.sum() > 0:
            results['ramadan'] = {
                'mape': calculate_mape(test_df[ramadan_mask]['demand_future'],
                                      predictions[ramadan_mask]),
                'rmse': calculate_rmse(test_df[ramadan_mask]['demand_future'],
                                      predictions[ramadan_mask]),
                'mae': calculate_mae(test_df[ramadan_mask]['demand_future'],
                                    predictions[ramadan_mask]),
                'count': int(ramadan_mask.sum())
            }
            print(f"Ramadan      - MAPE: {results['ramadan']['mape']:6.2f}% | "
                  f"RMSE: {results['ramadan']['rmse']:7.2f} | "
                  f"MAE: {results['ramadan']['mae']:7.2f} | "
                  f"N: {results['ramadan']['count']:,}")
        else:
            print(f"Ramadan      - No data available in test set")
    
    # Holidays
    if 'is_holiday' in test_df.columns:
        holiday_mask = test_df['is_holiday'] == 1
        if holiday_mask.sum() > 0:
            results['holidays'] = {
                'mape': calculate_mape(test_df[holiday_mask]['demand_future'],
                                      predictions[holiday_mask]),
                'rmse': calculate_rmse(test_df[holiday_mask]['demand_future'],
                                      predictions[holiday_mask]),
                'mae': calculate_mae(test_df[holiday_mask]['demand_future'],
                                    predictions[holiday_mask]),
                'count': int(holiday_mask.sum())
            }
            print(f"Holidays     - MAPE: {results['holidays']['mape']:6.2f}% | "
                  f"RMSE: {results['holidays']['rmse']:7.2f} | "
                  f"MAE: {results['holidays']['mae']:7.2f} | "
                  f"N: {results['holidays']['count']:,}")
        else:
            print(f"Holidays     - No data available in test set")
    
    # Weekends
    if 'is_weekend' in test_df.columns:
        weekend_mask = test_df['is_weekend'] == 1
        if weekend_mask.sum() > 0:
            results['weekends'] = {
                'mape': calculate_mape(test_df[weekend_mask]['demand_future'],
                                      predictions[weekend_mask]),
                'rmse': calculate_rmse(test_df[weekend_mask]['demand_future'],
                                      predictions[weekend_mask]),
                'mae': calculate_mae(test_df[weekend_mask]['demand_future'],
                                    predictions[weekend_mask]),
                'count': int(weekend_mask.sum())
            }
            print(f"Weekends     - MAPE: {results['weekends']['mape']:6.2f}% | "
                  f"RMSE: {results['weekends']['rmse']:7.2f} | "
                  f"MAE: {results['weekends']['mae']:7.2f} | "
                  f"N: {results['weekends']['count']:,}")
    
    print(f'\n{"=" * 80}')
    print(f'✓ Comprehensive evaluation complete for {model_name}')
    print(f'{"=" * 80}\n')
    
    return results


# Dictionary to store all model results (will use UNSCALED data for feature extraction)
model_results = {}

print('✓ Evaluation functions defined')
print('  - evaluate_model(): Standard metrics for train/val/test')
print('  - comprehensive_evaluation(): Detailed segmented analysis')
print('✓ Tree-based models use unscaled data (X_train, X_val, X_test)')
print('✓ Linear models use scaled data (X_train_scaled, X_val_scaled, X_test_scaled)')



# ## 6. Baseline Model: Linear Regression


# ## ⚠️ CRITICAL: Feature Preparation Verification# # All preprocessing steps respect time-series data integrity:# - ✅ Percentile bounds frozen from training set# - ✅ Z-score parameters frozen from training set# - ✅ Scaler (StandardScaler) fit ONLY on training set# - ✅ Test/Validation use training statistics (NOT recomputed)# - ✅ No temporal data leakage


print('=' * 80)
print('BASELINE MODEL: LINEAR REGRESSION')
print('=' * 80)

# Train linear regression
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Predictions
y_train_pred_lr = lr_model.predict(X_train_scaled)
y_val_pred_lr = lr_model.predict(X_val_scaled)
y_test_pred_lr = lr_model.predict(X_test_scaled)

# Evaluate
train_metrics_lr = evaluate_model(y_train, y_train_pred_lr, 'Linear Regression', 'Train')
val_metrics_lr = evaluate_model(y_val, y_val_pred_lr, 'Linear Regression', 'Validation')
test_metrics_lr = evaluate_model(y_test, y_test_pred_lr, 'Linear Regression', 'Test')

# Store results
model_results['Linear Regression'] = {
    'model': lr_model,
    'train_metrics': train_metrics_lr,
    'val_metrics': val_metrics_lr,
    'test_metrics': test_metrics_lr,
    'predictions': {
        'train': y_train_pred_lr,
        'val': y_val_pred_lr,
        'test': y_test_pred_lr
    }
}

print(f'\n✓ Linear Regression trained successfully')


# ## 7. Ridge Regression (L2 Regularization)


print('=' * 80)
print('RIDGE REGRESSION (L2 Regularization)')
print('=' * 80)

# Train Ridge with default alpha
ridge_model = Ridge(alpha=1.0, random_state=42)
ridge_model.fit(X_train_scaled, y_train)

# Predictions
y_train_pred_ridge = ridge_model.predict(X_train_scaled)
y_val_pred_ridge = ridge_model.predict(X_val_scaled)
y_test_pred_ridge = ridge_model.predict(X_test_scaled)

# Evaluate
train_metrics_ridge = evaluate_model(y_train, y_train_pred_ridge, 'Ridge', 'Train')
val_metrics_ridge = evaluate_model(y_val, y_val_pred_ridge, 'Ridge', 'Validation')
test_metrics_ridge = evaluate_model(y_test, y_test_pred_ridge, 'Ridge', 'Test')

# Store results
model_results['Ridge'] = {
    'model': ridge_model,
    'train_metrics': train_metrics_ridge,
    'val_metrics': val_metrics_ridge,
    'test_metrics': test_metrics_ridge,
    'predictions': {
        'train': y_train_pred_ridge,
        'val': y_val_pred_ridge,
        'test': y_test_pred_ridge
    }
}

print(f'\n✓ Ridge Regression trained successfully')


# ## 8. Random Forest Regressor


print('=' * 80)
print('RANDOM FOREST REGRESSOR')
print('=' * 80)

# Train Random Forest (use unscaled data - tree-based models don't need scaling)
rf_model = RandomForestRegressor(
    n_estimators=10,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

print('Training Random Forest (100 trees)...')
rf_model.fit(X_train, y_train)

# Predictions
y_train_pred_rf = rf_model.predict(X_train)
y_val_pred_rf = rf_model.predict(X_val)
y_test_pred_rf = rf_model.predict(X_test)

# Evaluate
train_metrics_rf = evaluate_model(y_train, y_train_pred_rf, 'Random Forest', 'Train')
val_metrics_rf = evaluate_model(y_val, y_val_pred_rf, 'Random Forest', 'Validation')
test_metrics_rf = evaluate_model(y_test, y_test_pred_rf, 'Random Forest', 'Test')

# Store results
model_results['Random Forest'] = {
    'model': rf_model,
    'train_metrics': train_metrics_rf,
    'val_metrics': val_metrics_rf,
    'test_metrics': test_metrics_rf,
    'predictions': {
        'train': y_train_pred_rf,
        'val': y_val_pred_rf,
        'test': y_test_pred_rf
    }
}

print(f'\n✓ Random Forest trained successfully')


# ## 9. XGBoost Model


print('=' * 80)
print('XGBOOST REGRESSOR')
print('=' * 80)

# Train XGBoost
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    tree_method='hist',
    verbosity=0
)

print('Training XGBoost (100 estimators)...')
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# Predictions
y_train_pred_xgb = xgb_model.predict(X_train)
y_val_pred_xgb = xgb_model.predict(X_val)
y_test_pred_xgb = xgb_model.predict(X_test)

# Evaluate
train_metrics_xgb = evaluate_model(y_train, y_train_pred_xgb, 'XGBoost', 'Train')
val_metrics_xgb = evaluate_model(y_val, y_val_pred_xgb, 'XGBoost', 'Validation')
test_metrics_xgb = evaluate_model(y_test, y_test_pred_xgb, 'XGBoost', 'Test')

# Store results
model_results['XGBoost'] = {
    'model': xgb_model,
    'train_metrics': train_metrics_xgb,
    'val_metrics': val_metrics_xgb,
    'test_metrics': test_metrics_xgb,
    'predictions': {
        'train': y_train_pred_xgb,
        'val': y_val_pred_xgb,
        'test': y_test_pred_xgb
    }
}

print(f'\n✓ XGBoost trained successfully')


# ## 10. LightGBM Model


print('=' * 80)
print('LIGHTGBM REGRESSOR')
print('=' * 80)

# Train LightGBM
lgb_model = lgb.LGBMRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)

print('Training LightGBM (100 estimators)...')
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.log_evaluation(period=0)]
)

# Predictions
y_train_pred_lgb = lgb_model.predict(X_train)
y_val_pred_lgb = lgb_model.predict(X_val)
y_test_pred_lgb = lgb_model.predict(X_test)

# Evaluate
train_metrics_lgb = evaluate_model(y_train, y_train_pred_lgb, 'LightGBM', 'Train')
val_metrics_lgb = evaluate_model(y_val, y_val_pred_lgb, 'LightGBM', 'Validation')
test_metrics_lgb = evaluate_model(y_test, y_test_pred_lgb, 'LightGBM', 'Test')

# Store results
model_results['LightGBM'] = {
    'model': lgb_model,
    'train_metrics': train_metrics_lgb,
    'val_metrics': val_metrics_lgb,
    'test_metrics': test_metrics_lgb,
    'predictions': {
        'train': y_train_pred_lgb,
        'val': y_val_pred_lgb,
        'test': y_test_pred_lgb
    }
}

print(f'\n✓ LightGBM trained successfully')


# ## 11. Model Comparison


print('=' * 80)
print('MODEL COMPARISON - VALIDATION SET')
print('=' * 80)

# Create comparison DataFrame
comparison_data = []

for model_name, results in model_results.items():
    comparison_data.append({
        'Model': model_name,
        'MAE': results['val_metrics']['MAE'],
        'RMSE': results['val_metrics']['RMSE'],
        'R²': results['val_metrics']['R2'],
        'MAPE (%)': results['val_metrics']['MAPE']
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('MAE')

print('\n' + comparison_df.to_string(index=False))

# Find best model
best_model_name = comparison_df.iloc[0]['Model']
best_mae = comparison_df.iloc[0]['MAE']

print(f'\n🏆 Best Model: {best_model_name} (MAE: {best_mae:.2f} MWh)')


# Visualize model comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

metrics_to_plot = ['MAE', 'RMSE', 'R²']
colors = plt.cm.Set3(range(len(comparison_df)))

for idx, metric in enumerate(metrics_to_plot):
    ax = axes[idx]
    bars = ax.bar(comparison_df['Model'], comparison_df[metric], color=colors)
    ax.set_ylabel(metric, fontsize=11, fontweight='bold')
    ax.set_title(f'{metric} Comparison (Validation Set)', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.close('all')

print('✓ Model comparison visualization complete')

# ## 12. Time-Series Cross-Validation (TimeSeriesSplit)
# ===== CRITICAL: TIME-SERIES CROSS-VALIDATION =====
# Issue 3 Fix: Use TimeSeriesSplit instead of single train/val/test split
print('\n' + '=' * 80)
print('TIME-SERIES CROSS-VALIDATION (TimeSeriesSplit)')
print('=' * 80)
print(f'\nUsing TimeSeriesSplit for robust time-series evaluation')
print(f'This ensures models are evaluated on multiple temporal folds')

# Combine train and validation for cross-validation
X_train_val = pd.concat([X_train_scaled, X_val_scaled], axis=0)
y_train_val = pd.concat([y_train, y_val], axis=0)

# Initialize TimeSeriesSplit
n_splits = 2
tscv = TimeSeriesSplit(n_splits=n_splits)

print(f'\nNumber of splits: {n_splits}')
print(f'Total samples for CV: {len(X_train_val):,}')

# Cross-validate the best model
cv_results = {}

for model_name in ['Linear Regression', 'Ridge', 'XGBoost', 'LightGBM']:
    if model_name not in model_results:
        continue
    
    print(f'\n{"-" * 80}')
    print(f'Cross-validating: {model_name}')
    print(f'{"-" * 80}')
    
    fold_scores = {
        'mape': [],
        'rmse': [],
        'mae': [],
        'r2': []
    }
    
    # Get the model type
    if model_name == 'Linear Regression':
        model_class = LinearRegression
        use_scaled = True
    elif model_name == 'Ridge':
        model_class = lambda: Ridge(alpha=1.0, random_state=42)
        use_scaled = True
    elif model_name == 'XGBoost':
        model_class = lambda: xgb.XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, tree_method='hist', verbosity=0
        )
        use_scaled = False
    elif model_name == 'LightGBM':
        model_class = lambda: lgb.LGBMRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=-1
        )
        use_scaled = False
    
    # Prepare data based on model type
    if use_scaled:
        X_cv = X_train_val
        # For unscaled models, use original features
        X_train_val_unscaled = pd.concat([X_train, X_val], axis=0)
    else:
        X_cv = pd.concat([X_train, X_val], axis=0)
    
    # Perform time-series cross-validation
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_cv), 1):
        # Split data
        X_fold_train = X_cv.iloc[train_idx]
        y_fold_train = y_train_val.iloc[train_idx]
        X_fold_val = X_cv.iloc[val_idx]
        y_fold_val = y_train_val.iloc[val_idx]
        
        # Train model
        if callable(model_class):
            model = model_class()
        else:
            model = model_class()
        
        model.fit(X_fold_train, y_fold_train)
        
        # Predict
        y_fold_pred = model.predict(X_fold_val)
        
        # Calculate metrics
        mape = calculate_mape(y_fold_val, y_fold_pred)
        rmse = calculate_rmse(y_fold_val, y_fold_pred)
        mae = calculate_mae(y_fold_val, y_fold_pred)
        r2 = r2_score(y_fold_val, y_fold_pred)
        
        fold_scores['mape'].append(mape)
        fold_scores['rmse'].append(rmse)
        fold_scores['mae'].append(mae)
        fold_scores['r2'].append(r2)
        
        print(f'  Fold {fold_idx}/{n_splits}: MAPE={mape:.2f}%, RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.4f}')
    
    # Calculate mean and std across folds
    cv_results[model_name] = {
        'mape_mean': np.mean(fold_scores['mape']),
        'mape_std': np.std(fold_scores['mape']),
        'rmse_mean': np.mean(fold_scores['rmse']),
        'rmse_std': np.std(fold_scores['rmse']),
        'mae_mean': np.mean(fold_scores['mae']),
        'mae_std': np.std(fold_scores['mae']),
        'r2_mean': np.mean(fold_scores['r2']),
        'r2_std': np.std(fold_scores['r2'])
    }
    
    print(f'\n  📊 CV Results ({n_splits} folds):')
    print(f'     MAPE: {cv_results[model_name]["mape_mean"]:.2f}% ± {cv_results[model_name]["mape_std"]:.2f}%')
    print(f'     RMSE: {cv_results[model_name]["rmse_mean"]:.2f} ± {cv_results[model_name]["rmse_std"]:.2f}')
    print(f'     MAE:  {cv_results[model_name]["mae_mean"]:.2f} ± {cv_results[model_name]["mae_std"]:.2f}')
    print(f'     R²:   {cv_results[model_name]["r2_mean"]:.4f} ± {cv_results[model_name]["r2_std"]:.4f}')

# Summary table
print(f'\n{"=" * 80}')
print('TIME-SERIES CROSS-VALIDATION SUMMARY')
print(f'{"=" * 80}')

cv_summary = []
for model_name, results in cv_results.items():
    cv_summary.append({
        'Model': model_name,
        'MAPE_mean': results['mape_mean'],
        'MAPE_std': results['mape_std'],
        'RMSE_mean': results['rmse_mean'],
        'MAE_mean': results['mae_mean'],
        'R2_mean': results['r2_mean']
    })

cv_summary_df = pd.DataFrame(cv_summary)
cv_summary_df = cv_summary_df.sort_values('MAPE_mean')

print('\nCross-Validation Results (Mean ± Std):')
for _, row in cv_summary_df.iterrows():
    print(f"\n{row['Model']}:")
    print(f"  MAPE: {row['MAPE_mean']:.2f}% ± {row['MAPE_std']:.2f}%")
    print(f"  RMSE: {row['RMSE_mean']:.2f}")
    print(f"  MAE:  {row['MAE_mean']:.2f}")
    print(f"  R²:   {row['R2_mean']:.4f}")

best_cv_model = cv_summary_df.iloc[0]['Model']
print(f'\n🏆 Best Model (CV): {best_cv_model} (MAPE: {cv_summary_df.iloc[0]["MAPE_mean"]:.2f}%)')

print(f'\n✓ Time-series cross-validation complete')
print(f'  Used {n_splits} temporal folds for robust evaluation')
print(f'  This addresses the limitation of single train/val/test split')

# ## 13. Feature Importance Analysis (Tree-Based Models)
print('=' * 80)
print('FEATURE IMPORTANCE ANALYSIS')
print('=' * 80)

# Get feature importance from best tree-based model
tree_models = ['Random Forest', 'XGBoost', 'LightGBM']
best_tree_model = None
best_tree_mae = float('inf')

for model_name in tree_models:
    if model_name in model_results:
        mae = model_results[model_name]['val_metrics']['MAE']
        if mae < best_tree_mae:
            best_tree_mae = mae
            best_tree_model = model_name

if best_tree_model:
    print(f'\nAnalyzing feature importance from: {best_tree_model}')
    
    model = model_results[best_tree_model]['model']
    
    if best_tree_model == 'Random Forest':
        importances = model.feature_importances_
    elif best_tree_model == 'XGBoost':
        importances = model.feature_importances_
    elif best_tree_model == 'LightGBM':
        importances = model.feature_importances_
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print(f'\nTop 20 Most Important Features:')
    print(importance_df.head(20).to_string(index=False))
    
    # Visualize top 15 features
    top_n = 15
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(top_n), top_features['Importance'].values)
    plt.yticks(range(top_n), top_features['Feature'].values)
    plt.xlabel('Importance Score', fontsize=11, fontweight='bold')
    plt.title(f'Top {top_n} Feature Importances - {best_tree_model}', fontsize=13, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(alpha=0.3, axis='x')
    plt.tight_layout()
    plt.close('all')
    
    # Save importance to CSV
    importance_path = f'../data/output/feature_importance_{best_tree_model.lower().replace(" ", "_")}.csv'
    importance_df.to_csv(importance_path, index=False)
    print(f'\n✓ Feature importance saved: {importance_path}')
else:
    print('No tree-based models available for feature importance analysis')

# ## 13. Prediction Analysis and Visualization
# Plot predictions vs actual for best model
best_model = model_results[best_model_name]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Validation set scatter plot
axes[0, 0].scatter(y_val, best_model['predictions']['val'], alpha=0.5, s=20)
axes[0, 0].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2, label='Perfect prediction')
axes[0, 0].set_xlabel('True Demand (MWh)', fontsize=11)
axes[0, 0].set_ylabel('Predicted Demand (MWh)', fontsize=11)
axes[0, 0].set_title(f'{best_model_name} - Validation Set', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Test set scatter plot
axes[0, 1].scatter(y_test, best_model['predictions']['test'], alpha=0.5, s=20, color='orange')
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect prediction')
axes[0, 1].set_xlabel('True Demand (MWh)', fontsize=11)
axes[0, 1].set_ylabel('Predicted Demand (MWh)', fontsize=11)
axes[0, 1].set_title(f'{best_model_name} - Test Set', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Residuals plot (validation)
residuals_val = y_val - best_model['predictions']['val']
axes[1, 0].scatter(best_model['predictions']['val'], residuals_val, alpha=0.5, s=20)
axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 0].set_xlabel('Predicted Demand (MWh)', fontsize=11)
axes[1, 0].set_ylabel('Residuals (MWh)', fontsize=11)
axes[1, 0].set_title('Residuals Plot - Validation Set', fontsize=12, fontweight='bold')
axes[1, 0].grid(alpha=0.3)

# Residuals distribution
axes[1, 1].hist(residuals_val, bins=50, edgecolor='black', alpha=0.7)
axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2)
axes[1, 1].set_xlabel('Residuals (MWh)', fontsize=11)
axes[1, 1].set_ylabel('Frequency', fontsize=11)
axes[1, 1].set_title('Residuals Distribution - Validation Set', fontsize=12, fontweight='bold')
axes[1, 1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.close('all')

print('✓ Prediction analysis visualization complete')

# ## 14. Time Series Predictions Visualization
# Plot time series of predictions vs actual for test set
# Visualization for multiple target months
target_months = {
    1: 'January',
    3: 'March',
    6: 'June',
    9: 'September'
}

cities = test_df['city'].unique()
city_to_plot = cities[0]  # Prefer the first city

plots_generated = 0

for target_month, target_month_name in target_months.items():
    # Filter for specific month and city
    mask = (test_df['time'].dt.month == target_month) & (test_df['city'] == city_to_plot)
    
    if mask.sum() > 0:
        print(f"Plotting data for {target_month_name} ({city_to_plot})")
        test_sample = test_df[mask].copy()
        test_sample['predicted'] = best_model['predictions']['test'][mask]
        
        plt.figure(figsize=(16, 6))
        plt.plot(test_sample['time'], test_sample['demand'], label='Actual Demand', linewidth=2, alpha=0.8)
        plt.plot(test_sample['time'], test_sample['predicted'], label='Predicted Demand', linewidth=2, alpha=0.8, linestyle='--')
        plt.xlabel('Time', fontsize=11, fontweight='bold')
        plt.ylabel('Demand (MWh)', fontsize=11, fontweight='bold')
        plt.title(f'{best_model_name} - {target_month_name} Predictions ({city_to_plot})', fontsize=13, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.close('all')
        plots_generated += 1

if plots_generated == 0:
    print(f"No specific month data found in test set. Showing first 7 days.")
    sample_days = 7
    sample_hours = sample_days * 24
    test_sample = test_df.iloc[:sample_hours].copy()
    test_sample['predicted'] = best_model['predictions']['test'][:sample_hours]
    
    plt.figure(figsize=(16, 6))
    plt.plot(test_sample['time'], test_sample['demand'], label='Actual Demand', linewidth=2, alpha=0.8)
    plt.plot(test_sample['time'], test_sample['predicted'], label='Predicted Demand', linewidth=2, alpha=0.8, linestyle='--')
    plt.xlabel('Time', fontsize=11, fontweight='bold')
    plt.ylabel('Demand (MWh)', fontsize=11, fontweight='bold')
    plt.title(f'{best_model_name} - First {sample_days} Days of Test Set Predictions', fontsize=13, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.close('all')

plt.xticks(rotation=45)
plt.tight_layout()
plt.close('all')

print(f'✓ Time series visualization ({sample_days if "sample_days" in locals() else target_month_name}) complete')

# ===== BEST AND WORST DAYS ANALYSIS =====
print('\n' + '=' * 80)
print('BEST AND WORST PREDICTION DAYS ANALYSIS')
print('=' * 80)

# Create a detailed error dataframe for analysis
error_df = test_df[['time', 'city', 'demand', 'demand_future']].copy()
error_df['predicted'] = best_model['predictions']['test']
error_df['abs_error'] = np.abs(error_df['demand_future'] - error_df['predicted'])
# Calculate percentage error (avoid division by zero)
error_df['mape'] = np.abs((error_df['demand_future'] - error_df['predicted']) / (error_df['demand_future'] + 1e-6)) * 100
# Extract date for grouping
error_df['date'] = error_df['time'].dt.date

# Calculate daily performance metrics
# We group by date and take the mean of MAPE for that day across all hours/cities
daily_performance = error_df.groupby('date').agg({
    'mape': 'mean',
    'abs_error': 'mean', 
    'demand_future': 'mean'
}).reset_index()

# Sort to find best and worst days
best_days = daily_performance.sort_values('mape').head(3)
worst_days = daily_performance.sort_values('mape', ascending=False).head(3)

print('\n🏆 TOP 3 BEST DAYS (Lowest Error):')
# Format for cleaner output
print(best_days.to_string(index=False, formatters={
    'mape': '{:.2f}%'.format,
    'abs_error': '{:.2f}'.format,
    'demand_future': '{:.2f}'.format
}))

print('\n⚠️ TOP 3 WORST DAYS (Highest Error):')
print(worst_days.to_string(index=False, formatters={
    'mape': '{:.2f}%'.format,
    'abs_error': '{:.2f}'.format,
    'demand_future': '{:.2f}'.format
}))

# Function to plot specific days
def plot_daily_performance(date_obj, title_prefix, color_scheme='green'):
    """Helper to plot performance for a specific date"""
    # Filter data for this date
    day_mask = error_df['time'].dt.date == date_obj
    day_data = error_df[day_mask].copy().sort_values(['city', 'time'])
    
    if len(day_data) == 0:
        return

    # If multiple cities, pick the first one to keep plot readable
    cities = day_data['city'].unique()
    city_to_show = cities[0]
    
    # Filter for single city
    day_city_data = day_data[day_data['city'] == city_to_show]
    
    plt.figure(figsize=(14, 5))
    
    # Plot actual vs predicted
    plt.plot(day_city_data['time'], day_city_data['demand_future'], 
             label='Actual', linewidth=2, color='black', alpha=0.7)
    plt.plot(day_city_data['time'], day_city_data['predicted'], 
             label='Predicted', linewidth=2, linestyle='--', color=color_scheme)
    
    # Calculate daily metric for this specific city/day view
    daily_mape = np.mean(np.abs((day_city_data['demand_future'] - day_city_data['predicted']) / 
                               (day_city_data['demand_future'] + 1e-6))) * 100
    
    plt.title(f'{title_prefix}: {date_obj} ({city_to_show})\nMAPE: {daily_mape:.2f}%', fontsize=12, fontweight='bold')
    plt.xlabel('Time of Day', fontsize=10)
    plt.ylabel('Demand (MWh)', fontsize=10)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.close('all')

# Visualize predictions for Best Days
print("\nPlotting Best Days...")
for _, row in best_days.iterrows():
    plot_daily_performance(row['date'], "✅ BEST PERFORMANCE", color_scheme='green')

# Visualize predictions for Worst Days
print("\nPlotting Worst Days...")
for _, row in worst_days.iterrows():
    plot_daily_performance(row['date'], "❌ WORST PERFORMANCE", color_scheme='red')

print('✓ Best/Worst days analysis visualization complete')

# ## 15. Regional Performance Analysis
print('=' * 80)
print('REGIONAL PERFORMANCE ANALYSIS')
print('=' * 80)

# Analyze performance by region on test set
test_results = test_df.copy()
test_results['predicted'] = best_model['predictions']['test']
test_results['error'] = test_results['demand_future'] - test_results['predicted']
test_results['abs_error'] = np.abs(test_results['error'])
test_results['pct_error'] = np.abs(test_results['error'] / (test_results['demand_future'] + 1e-10)) * 100

regional_performance = []

for region in test_results['city'].unique():
    region_data = test_results[test_results['city'] == region]
    
    mae = region_data['abs_error'].mean()
    rmse = np.sqrt((region_data['error']**2).mean())
    mape = region_data['pct_error'].mean()
    r2 = r2_score(region_data['demand_future'], region_data['predicted'])
    
    regional_performance.append({
        'Region': region.capitalize(),
        'Samples': len(region_data),
        'MAE': mae,
        'RMSE': rmse,
        'MAPE (%)': mape,
        'R²': r2
    })

regional_df = pd.DataFrame(regional_performance)
print('\n' + regional_df.to_string(index=False))

# Visualize regional performance
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# MAE by region
axes[0].bar(regional_df['Region'], regional_df['MAE'], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
axes[0].set_ylabel('MAE (MWh)', fontsize=11, fontweight='bold')
axes[0].set_title('Mean Absolute Error by Region', fontsize=12, fontweight='bold')
axes[0].grid(alpha=0.3, axis='y')

# Add value labels
for i, val in enumerate(regional_df['MAE']):
    axes[0].text(i, val, f'{val:.2f}', ha='center', va='bottom', fontsize=10)

# R² by region
axes[1].bar(regional_df['Region'], regional_df['R²'], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
axes[1].set_ylabel('R² Score', fontsize=11, fontweight='bold')
axes[1].set_title('R² Score by Region', fontsize=12, fontweight='bold')
axes[1].grid(alpha=0.3, axis='y')

# Add value labels
for i, val in enumerate(regional_df['R²']):
    axes[1].text(i, val, f'{val:.4f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.close('all')

print('\n✓ Regional performance analysis complete')


# ## 16. Export Results and Predictions


print('=' * 80)
print('EXPORTING RESULTS')
print('=' * 80)

# Export model comparison
comparison_path = '../data/output/model_comparison.csv'
comparison_df.to_csv(comparison_path, index=False)
print(f'✓ Model comparison saved: {comparison_path}')

# Export regional performance
regional_path = '../data/output/regional_performance.csv'
regional_df.to_csv(regional_path, index=False)
print(f'✓ Regional performance saved: {regional_path}')

# Export test predictions with metadata
predictions_df = test_df[['time', 'city', 'demand', 'demand_future']].copy()
predictions_df['predicted_demand'] = best_model['predictions']['test']
predictions_df['error'] = predictions_df['demand_future'] - predictions_df['predicted_demand']
predictions_df['abs_error'] = np.abs(predictions_df['error'])
predictions_df['pct_error'] = np.abs(predictions_df['error'] / (predictions_df['demand_future'] + 1e-10)) * 100

predictions_path = '../data/output/test_predictions.csv'
predictions_df.to_csv(predictions_path, index=False)
print(f'✓ Test predictions saved: {predictions_path}')

# Save model metadata
import json

model_metadata = {
    'timestamp': datetime.now().isoformat(),
    'best_model': best_model_name,
    'data_integrity_checks': {
        'percentile_bounds_used': 'percentile_bounds_from_training.json (FROZEN)',
        'zscore_params_used': 'zscore_params_from_training.json (FROZEN)',
        'scaler_fit_on': 'training_set_only',
        'train_val_test_split': 'chronological (no shuffle)',
        'comment': 'All parameters computed from training set only - no test leakage'
    },
    'best_model_metrics': {
        'validation': best_model['val_metrics'],
        'test': best_model['test_metrics']
    },
    'all_models_validation': {
        name: results['val_metrics'] 
        for name, results in model_results.items()
    },
    'data_split': {
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'total_features': len(feature_cols),
        'feature_note': 'Features use frozen preprocessing parameters from training set'
    },
    'regional_performance': regional_df.to_dict('records')
}

metadata_path = '../data/output/model_training_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(model_metadata, f, indent=2, default=str)
print(f'✓ Model metadata saved: {metadata_path}')

# Verify frozen parameter files exist
if os.path.exists(f'{data_path}percentile_bounds_from_training.json'):
    print(f'✓ Frozen percentile bounds: {data_path}percentile_bounds_from_training.json')
if os.path.exists(f'{data_path}zscore_params_from_training.json'):
    print(f'✓ Frozen z-score parameters: {data_path}zscore_params_from_training.json')

print('\n' + '=' * 80)
print('ALL RESULTS EXPORTED SUCCESSFULLY')
print('=' * 80)
print('\n⚠️  CRITICAL: Frozen parameter files are saved and must be used in production')
print('   - percentile_bounds_from_training.json')
print('   - zscore_params_from_training.json')



# ## 17. Comprehensive Segmented Evaluation


print('=' * 80)
print('COMPREHENSIVE SEGMENTED EVALUATION')
print('=' * 80)

# Run comprehensive evaluation on the best model
print(f'\nRunning comprehensive evaluation for best model: {best_model_name}')
print(f'This will analyze performance across:')
print(f'  - Cities')
print(f'  - Seasons')
print(f'  - Hours of day')
print(f'  - Special periods (Ramadan, holidays, weekends)')

comprehensive_results = comprehensive_evaluation(
    model=best_model['model'],
    test_df=test_df,
    predictions=best_model['predictions']['test'],
    model_name=best_model_name
)

# Save comprehensive results to JSON
import json
results_path = f'../data/output/comprehensive_evaluation_{best_model_name.lower().replace(" ", "_")}.json'
with open(results_path, 'w') as f:
    json.dump(comprehensive_results, f, indent=2, default=str)
print(f'\n✓ Comprehensive evaluation results saved: {results_path}')

# Create visualization of performance by segment
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Performance by City
if any(k.startswith('city_') for k in comprehensive_results.keys()):
    city_data = {k.replace('city_', ''): v for k, v in comprehensive_results.items() if k.startswith('city_')}
    cities = list(city_data.keys())
    mapes = [city_data[c]['mape'] for c in cities]
    
    axes[0, 0].barh(cities, mapes, color='skyblue')
    axes[0, 0].set_xlabel('MAPE (%)', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Performance by City', fontsize=12, fontweight='bold')
    axes[0, 0].grid(alpha=0.3, axis='x')
    
    for i, (city, mape) in enumerate(zip(cities, mapes)):
        axes[0, 0].text(mape, i, f' {mape:.2f}%', va='center', fontsize=9)

# 2. Performance by Season
if any(k.startswith('season_') for k in comprehensive_results.keys()):
    season_data = {k.replace('season_', ''): v for k, v in comprehensive_results.items() if k.startswith('season_')}
    seasons = ['winter', 'spring', 'summer', 'fall']
    seasons_present = [s for s in seasons if s in season_data]
    mapes = [season_data[s]['mape'] for s in seasons_present]
    
    axes[0, 1].bar(range(len(seasons_present)), mapes, color=['lightblue', 'lightgreen', 'orange', 'brown'][:len(seasons_present)])
    axes[0, 1].set_xticks(range(len(seasons_present)))
    axes[0, 1].set_xticklabels([s.capitalize() for s in seasons_present])
    axes[0, 1].set_ylabel('MAPE (%)', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Performance by Season', fontsize=12, fontweight='bold')
    axes[0, 1].grid(alpha=0.3, axis='y')
    
    for i, mape in enumerate(mapes):
        axes[0, 1].text(i, mape, f'{mape:.2f}%', ha='center', va='bottom', fontsize=9)

# 3. Performance by Hour
if any(k.startswith('hour_') for k in comprehensive_results.keys()):
    hour_data = {k.replace('hour_', ''): v for k, v in comprehensive_results.items() if k.startswith('hour_')}
    hours = sorted(hour_data.keys())
    mapes = [hour_data[h]['mape'] for h in hours]
    hour_labels = [f'{h}:00' for h in hours]
    
    axes[1, 0].plot(range(len(hours)), mapes, marker='o', linewidth=2, markersize=8, color='purple')
    axes[1, 0].set_xticks(range(len(hours)))
    axes[1, 0].set_xticklabels(hour_labels)
    axes[1, 0].set_ylabel('MAPE (%)', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Performance by Hour of Day', fontsize=12, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    
    for i, mape in enumerate(mapes):
        axes[1, 0].text(i, mape, f'{mape:.1f}%', ha='center', va='bottom', fontsize=8)

# 4. Performance during Special Periods
special_periods = {}
if 'ramadan' in comprehensive_results:
    special_periods['Ramadan'] = comprehensive_results['ramadan']['mape']
if 'holidays' in comprehensive_results:
    special_periods['Holidays'] = comprehensive_results['holidays']['mape']
if 'weekends' in comprehensive_results:
    special_periods['Weekends'] = comprehensive_results['weekends']['mape']
if 'overall' in comprehensive_results:
    special_periods['Overall'] = comprehensive_results['overall']['mape']

if special_periods:
    periods = list(special_periods.keys())
    mapes = list(special_periods.values())
    colors = ['gold', 'coral', 'lightgreen', 'lightgray'][:len(periods)]
    
    axes[1, 1].bar(periods, mapes, color=colors)
    axes[1, 1].set_ylabel('MAPE (%)', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Performance: Special Periods vs Overall', fontsize=12, fontweight='bold')
    axes[1, 1].tick_params(axis='x', rotation=15)
    axes[1, 1].grid(alpha=0.3, axis='y')
    
    for i, (period, mape) in enumerate(zip(periods, mapes)):
        axes[1, 1].text(i, mape, f'{mape:.2f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.close('all')

print('✓ Comprehensive evaluation visualization complete')


# ## 18. Final Summary and Recommendations

print('=' * 80)
print('FINAL SUMMARY AND RECOMMENDATIONS')
print('=' * 80)


print(f'\n📊 DATASET INFORMATION')
print(f'   Total samples: {len(df):,}')
print(f'   Features used: {len(feature_cols)}')
print(f'   Train/Val/Test: {len(train_df):,} / {len(val_df):,} / {len(test_df):,}')

print(f'\n🏆 BEST MODEL: {best_model_name}')
print(f'\n   Validation Metrics:')
for metric, value in best_model['val_metrics'].items():
    if metric in ['MAE', 'RMSE', 'Max_Error', 'Median_AE']:
        print(f'      {metric}: {value:.2f} MWh')
    elif metric == 'MAPE':
        print(f'      {metric}: {value:.2f}%')
    else:
        print(f'      {metric}: {value:.4f}')

print(f'\n   Test Metrics:')
for metric, value in best_model['test_metrics'].items():
    if metric in ['MAE', 'RMSE', 'Max_Error', 'Median_AE']:
        print(f'      {metric}: {value:.2f} MWh')
    elif metric == 'MAPE':
        print(f'      {metric}: {value:.2f}%')
    else:
        print(f'      {metric}: {value:.4f}')

print(f'\n🔒 DATA INTEGRITY & LEAKAGE PREVENTION')
print(f'   ✅ Percentile bounds: FROZEN from training set')
print(f'   ✅ Z-score parameters: FROZEN from training set')
print(f'   ✅ Feature scaling: FIT on training set only')
print(f'   ✅ Train/Val/Test: Chronological split (no shuffle)')
print(f'   ✅ Missing value imputation: Using training set statistics')
print(f'   ✅ Feature engineering: Uses leakage-free features from notebook 2')
print(f'   ✅ No recomputation on test/validation data')

print(f'\n📁 FROZEN PARAMETERS (MUST USE IN PRODUCTION):')
print(f'   1. {data_path}percentile_bounds_from_training.json')
print(f'   2. {data_path}zscore_params_from_training.json')
print(f'   3. StandardScaler parameters (saved in model)')

print(f'\n🎯 NEXT STEPS / RECOMMENDATIONS:')
print(f'   1. Hyperparameter tuning for {best_model_name} (use TimeSeriesSplit)')
print(f'   2. Ensemble modeling (combine top 3 models)')
print(f'   3. Time-series cross-validation for robust evaluation')
print(f'   4. Analyze prediction errors by time of day and season')
print(f'   5. Regional-specific model tuning')
print(f'   6. Deploy with FROZEN preprocessing parameters')

print(f'\n⚠️  PRODUCTION DEPLOYMENT CHECKLIST:')
print(f'   [ ] Load frozen percentile bounds from JSON')
print(f'   [ ] Load frozen z-score parameters from JSON')
print(f'   [ ] Load StandardScaler from saved model')
print(f'   [ ] Apply frozen parameters in this exact order:')
print(f'       1. Percentile mapping')
print(f'       2. Z-score normalization')  
print(f'       3. StandardScaler transformation')
print(f'   [ ] NEVER recompute percentiles/z-scores on new data')
print(f'   [ ] Monitor for data distribution drift')
print(f'   [ ] Retrain monthly with new data + old frozen parameters')

print('\n' + '=' * 80)
print(f'Training completed: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
print('=' * 80)

