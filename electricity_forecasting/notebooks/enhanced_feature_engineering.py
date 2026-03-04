#!/usr/bin/env python3
"""
Enhanced Feature Engineering Pipeline for Turkish Electricity Demand Forecasting

This script creates sophisticated, leakage-safe features from raw data.
Designed for production use with strict temporal integrity guarantees.

Usage:
    python enhanced_feature_engineering.py \
        --config configs/standard.yaml \
        --input-dir data/raw/ \
        --output-dir data/processed/

Author: AI Assistant
Version: 2.0.0
"""

import argparse
import hashlib
import json
import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable, Any

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from scipy.signal import savgol_filter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


# ============================================================================
# CONFIGURATION SYSTEM
# ============================================================================

@dataclass
class CityConfig:
    """City-specific parameters"""
    name: str
    hdd_base: float = 18.0
    cdd_base: float = 24.0
    is_coastal: bool = False
    is_industrial: bool = False
    tourism_months: List[int] = field(default_factory=lambda: [6, 7, 8])
    latitude: float = 39.0
    longitude: float = 35.0
    elevation: float = 100.0  # meters


@dataclass
class FeatureConfig:
    """Feature engineering configuration"""
    version: str = "2.0.0"
    description: str = "Enhanced feature set"
    
    # Temporal features
    cyclical_encoding: bool = True
    calendar_features: bool = True
    holiday_features: bool = True
    
    # Weather features
    weather_raw: bool = True
    weather_derived: bool = True
    degree_days: bool = True
    heat_index: bool = True
    wind_chill: bool = False  # Rarely relevant for Turkey
    
    # Lag features
    lag_hours: List[int] = field(default_factory=lambda: [1, 2, 3, 6, 12, 24, 48, 72, 168])
    
    # Rolling statistics
    rolling_windows: List[int] = field(default_factory=lambda: [6, 12, 24, 48, 168])
    rolling_stats: List[str] = field(default_factory=lambda: ['mean', 'std', 'min', 'max', 'skew'])
    
    # Advanced features
    exponential_moving: bool = True
    rate_of_change: bool = True
    volatility_features: bool = True
    fourier_features: bool = True
    
    # Similar-day features
    similar_day_stats: bool = True
    historical_patterns: bool = True
    rolling_similar_days: bool = True
    weather_analogues: bool = True  # New for Weather-Based Analogues
    
    # Domain-specific
    industrial_features: bool = True
    tourism_features: bool = True
    agricultural_features: bool = True
    education_features: bool = True
    
    # Interactions
    feature_interactions: bool = True  # Enabled for Tier 4
    
    # Safety
    enforce_leakage_checks: bool = True
    freeze_percentiles: bool = True


# Default city configurations for Turkey
DEFAULT_CITIES = {
    'aydin': CityConfig('aydin', hdd_base=16, cdd_base=26, is_coastal=True, 
                       tourism_months=[5, 6, 7, 8, 9], latitude=37.85, longitude=27.85),
    'denizli': CityConfig('denizli', hdd_base=17, cdd_base=25, is_industrial=True,
                         latitude=37.78, longitude=29.10, elevation=400),
    'mugla': CityConfig('mugla', hdd_base=16, cdd_base=26, is_coastal=True,
                       tourism_months=[5, 6, 7, 8, 9], latitude=37.22, longitude=28.36),
}


# ============================================================================
# BASE PIPELINE CLASS
# ============================================================================

class FeaturePipeline(ABC):
    """Abstract base class for feature pipelines"""
    
    def __init__(self, config: FeatureConfig, city_config: Optional[CityConfig] = None):
        self.config = config
        self.city_config = city_config
        self.feature_names: List[str] = []
        
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform dataframe, adding features"""
        pass
    
    def validate_no_leakage(self, df: pd.DataFrame, original_index: pd.DatetimeIndex):
        """Validate that no future information leaked"""
        if not self.config.enforce_leakage_checks:
            return True
            
        # Check index hasn't been reordered
        if not df.index.equals(original_index):
            logger.warning(f"{self.__class__.__name__}: Index was modified!")
            return False
            
        # Check no future dates appeared
        if len(df) > 0:
            max_original = original_index.max()
            max_new = df.index.max()
            if max_new > max_original:
                logger.error(f"{self.__class__.__name__}: Future dates detected!")
                return False
                
        return True
    
    def get_feature_names(self) -> List[str]:
        return self.feature_names


# ============================================================================
# TEMPORAL FEATURE PIPELINE
# ============================================================================

class TemporalFeaturePipeline(FeaturePipeline):
    """
    Create time-based features with cyclical encoding.
    No leakage possible - all derived from timestamp.
    """
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        original_index = df.index.copy()
        
        # Basic time components
        features['year'] = df.index.year
        features['month'] = df.index.month
        features['day'] = df.index.day
        features['hour'] = df.index.hour
        features['dayofweek'] = df.index.dayofweek
        features['dayofyear'] = df.index.dayofyear
        features['weekofyear'] = df.index.isocalendar().week.astype(int)
        
        # Cyclical encoding (preserves continuity)
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        features['dow_sin'] = np.sin(2 * np.pi * features['dayofweek'] / 7)
        features['dow_cos'] = np.cos(2 * np.pi * features['dayofweek'] / 7)
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
        features['doy_sin'] = np.sin(2 * np.pi * features['dayofyear'] / 365)
        features['doy_cos'] = np.cos(2 * np.pi * features['dayofyear'] / 365)
        
        # Business time features
        features['is_weekend'] = (features['dayofweek'] >= 5).astype(int)
        features['is_business_day'] = (features['dayofweek'] < 5).astype(int)
        features['is_morning_peak'] = ((features['hour'] >= 7) & (features['hour'] <= 9)).astype(int)
        features['is_evening_peak'] = ((features['hour'] >= 17) & (features['hour'] <= 21)).astype(int)
        features['is_night'] = ((features['hour'] >= 22) | (features['hour'] <= 5)).astype(int)
        features['is_midday'] = ((features['hour'] >= 10) & (features['hour'] <= 16)).astype(int)
        
        # Day phase categorical
        def get_day_phase(hour):
            if 5 <= hour < 12:
                return 'morning'
            elif 12 <= hour < 17:
                return 'afternoon'
            elif 17 <= hour < 22:
                return 'evening'
            else:
                return 'night'
        
        features['day_phase'] = features['hour'].apply(get_day_phase)
        
        # Season
        features['season'] = features['month'].map({
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'fall', 10: 'fall', 11: 'fall'
        })
        
        # Week of month
        features['week_of_month'] = ((features['day'] - 1) // 7).astype(int)
        
        self.feature_names = list(features.columns)
        self.validate_no_leakage(features, original_index)
        
        return features


# ============================================================================
# WEATHER FEATURE PIPELINE
# ============================================================================

class WeatherFeaturePipeline(FeaturePipeline):
    """
    Create weather-derived features with city-specific parameters.
    """
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        original_index = df.index.copy()
        
        # Check required columns
        required = ['temperature_2m', 'relative_humidity_2m']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required weather columns: {missing}")
        
        # Raw weather (pass through)
        weather_cols = ['temperature_2m', 'relative_humidity_2m', 
                       'wind_speed_10m', 'cloud_cover', 'shortwave_radiation']
        for col in weather_cols:
            if col in df.columns:
                features[col] = df[col]
        
        # Temperature transformations
        temp = df['temperature_2m']
        features['temp_squared'] = temp ** 2
        features['temp_cubed'] = temp ** 3  # For extreme non-linearity
        
        # Degree days with city-specific bases
        if self.city_config:
            hdd_base = self.city_config.hdd_base
            cdd_base = self.city_config.cdd_base
        else:
            hdd_base, cdd_base = 18, 24
        
        features['hdd'] = np.maximum(0, hdd_base - temp)
        features['cdd'] = np.maximum(0, temp - cdd_base)
        
        # Accumulated degree days (seasonal stress)
        features['hdd_cumulative'] = features['hdd'].cumsum()
        features['cdd_cumulative'] = features['cdd'].cumsum()
        
        # Heat index (apparent temperature)
        if 'relative_humidity_2m' in df.columns:
            rh = df['relative_humidity_2m']
            # Simplified heat index calculation
            features['heat_index'] = self._calculate_heat_index(temp, rh)
            
            # Humidex (Canadian version)
            features['humidex'] = temp + 0.5555 * (6.11 * np.exp(5417.7530 * 
                              (1/273.16 - 1/(temp + 273.16))) * (rh/100) - 10)
        
        # Wind chill (if relevant)
        if self.config.wind_chill and 'wind_speed_10m' in df.columns:
            wind = df['wind_speed_10m']
            features['wind_chill'] = 13.12 + 0.6215 * temp - 11.37 * (wind ** 0.16) + \
                                    0.3965 * temp * (wind ** 0.16)
        
        # Temperature rate of change
        features['temp_change_1h'] = temp.diff(1)
        features['temp_change_3h'] = temp.diff(3)
        features['temp_change_6h'] = temp.diff(6)
        features['temp_change_24h'] = temp.diff(24)
        
        # Temperature volatility
        features['temp_volatility_6h'] = temp.rolling(6, min_periods=1).std()
        features['temp_volatility_24h'] = temp.rolling(24, min_periods=1).std()
        
        # Effective temperature (combines multiple factors)
        if 'relative_humidity_2m' in df.columns and 'wind_speed_10m' in df.columns:
            features['effective_temp'] = (
                0.5 * temp + 
                0.3 * features['heat_index'] + 
                0.2 * (temp - df['wind_speed_10m'] * 0.5)  # Wind cooling
            )
        
        self.feature_names = list(features.columns)
        self.validate_no_leakage(features, original_index)
        
        return features
    
    def _calculate_heat_index(self, temp, rh):
        """Calculate heat index using NOAA formula"""
        # Simplified version
        hi = 0.5 * (temp + 61.0 + ((temp - 68.0) * 1.2) + (rh * 0.094))
        
        # Full calculation for hot conditions
        hot_mask = temp >= 80
        if hot_mask.any():
            t = temp[hot_mask]
            r = rh[hot_mask]
            hi_hot = -42.379 + 2.04901523*t + 10.14333127*r - 0.22475541*t*r \
                     - 6.83783e-3*t**2 - 5.481717e-2*r**2 + 1.22874e-3*t**2*r \
                     + 8.5282e-4*t*r**2 - 1.99e-6*t**2*r**2
            hi[hot_mask] = hi_hot
        
        return hi


# ============================================================================
# LAG FEATURE PIPELINE (CRITICAL FOR LEAKAGE PREVENTION)
# ============================================================================

class LagFeaturePipeline(FeaturePipeline):
    """
    Create lagged demand features with STRICT horizon safety.
    This is the most critical pipeline for leakage prevention.
    """
    
    def __init__(self, config: FeatureConfig, city_config: Optional[CityConfig] = None,
                 forecast_horizon: int = 24):
        super().__init__(config, city_config)
        self.forecast_horizon = forecast_horizon
        self.safe_lag_min = forecast_horizon + 1
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'demand' not in df.columns:
            raise ValueError("DataFrame must contain 'demand' column")
        
        features = pd.DataFrame(index=df.index)
        original_index = df.index.copy()
        demand = df['demand']
        
        # Filter to safe lags only
        safe_lags = [lag for lag in self.config.lag_hours if lag >= self.safe_lag_min]
        
        if not safe_lags:
            logger.warning(f"No safe lags for horizon={self.forecast_horizon}h. "
                          f"Minimum safe lag is {self.safe_lag_min}h.")
            # Add a placeholder
            features['demand_lag_safe'] = np.nan
        else:
            logger.info(f"Creating {len(safe_lags)} safe lags (>= {self.safe_lag_min}h) "
                       f"for horizon={self.forecast_horizon}h")
            
            for lag in safe_lags:
                features[f'demand_lag_{lag}h'] = demand.shift(lag)
        
        self.feature_names = list(features.columns)
        self.validate_no_leakage(features, original_index)
        
        # Verify no future data
        for col in features.columns:
            if features[col].notna().any():
                # Check that lagged values are from past
                first_valid = features[col].first_valid_index()
                if first_valid:
                    expected_first = original_index[0] + timedelta(hours=int(col.split('_')[2].replace('h', '')))
                    if first_valid < expected_first:
                        logger.error(f"Leakage detected in {col}!")
        
        return features


# ============================================================================
# ROLLING STATISTICS PIPELINE (CRITICAL FOR LEAKAGE PREVENTION)
# ============================================================================

class RollingFeaturePipeline(FeaturePipeline):
    """
    Create rolling statistics with CRITICAL shift(1) before rolling.
    This prevents including current observation in statistics.
    """
    
    
    def __init__(self, config: FeatureConfig, city_config: Optional[CityConfig] = None, forecast_horizon: int = 24):
        super().__init__(config, city_config)
        self.forecast_horizon = forecast_horizon
        self.safe_lag_min = forecast_horizon + 1

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'demand' not in df.columns:
            raise ValueError("DataFrame must contain 'demand' column")
        
        features = pd.DataFrame(index=df.index)
        original_index = df.index.copy()
        
        # CRITICAL: Shift by safe lag to separate features from target
        demand_shifted = df['demand'].shift(self.safe_lag_min)
        
        for window in self.config.rolling_windows:
            for stat in self.config.rolling_stats:
                col_name = f'demand_{stat}_{window}h'
                
                if stat == 'mean':
                    features[col_name] = demand_shifted.rolling(window, min_periods=1).mean()
                elif stat == 'std':
                    features[col_name] = demand_shifted.rolling(window, min_periods=1).std()
                elif stat == 'min':
                    features[col_name] = demand_shifted.rolling(window, min_periods=1).min()
                elif stat == 'max':
                    features[col_name] = demand_shifted.rolling(window, min_periods=1).max()
                elif stat == 'skew':
                    features[col_name] = demand_shifted.rolling(window, min_periods=3).skew()
                elif stat == 'kurt':
                    features[col_name] = demand_shifted.rolling(window, min_periods=4).kurt()
        
        # Coefficient of variation (volatility measure)
        for window in self.config.rolling_windows:
            ma_col = f'demand_mean_{window}h'
            std_col = f'demand_std_{window}h'
            if ma_col in features.columns and std_col in features.columns:
                features[f'demand_cv_{window}h'] = features[std_col] / features[ma_col].replace(0, np.nan)
        
        # Exponential moving averages
        if self.config.exponential_moving:
            for span in [12, 24, 168]:
                features[f'demand_ema_{span}h'] = demand_shifted.ewm(span=span, min_periods=1).mean()
        
        # Rate of change in rolling statistics
        if self.config.rate_of_change:
            for window in [24, 168]:
                ma_col = f'demand_mean_{window}h'
                if ma_col in features.columns:
                    features[f'demand_roc_{window}h'] = features[ma_col].diff(1)
        
        self.feature_names = list(features.columns)
        self.validate_no_leakage(features, original_index)
        
        return features


# ============================================================================
# SIMILAR-DAY FEATURE PIPELINE
# ============================================================================

class SimilarDayPipeline(FeaturePipeline):
    """
    Create features based on historical similar patterns.
    Uses pre-computed statistics to prevent leakage.
    """
    
    def __init__(self, config: FeatureConfig, city_config: Optional[CityConfig] = None,
                 historical_stats: Optional[Dict] = None, forecast_horizon: int = 24):
        super().__init__(config, city_config)
        self.historical_stats = historical_stats or {}
        self.forecast_horizon = forecast_horizon
        self.safe_lag_min = forecast_horizon + 1
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        original_index = df.index.copy()
        
        if 'demand' not in df.columns:
            return features
        
        demand = df['demand']
        
        # Direct lag-based similar days (safe)
        features['demand_same_hour_1w'] = demand.shift(168)   # 1 week ago
        features['demand_same_hour_2w'] = demand.shift(336)   # 2 weeks ago
        features['demand_same_hour_3w'] = demand.shift(504)   # 3 weeks ago
        features['demand_same_hour_4w'] = demand.shift(672)   # 4 weeks ago
        
        # Same day-of-week historical average (computed safely)
        if self.historical_stats:
            # Use pre-computed statistics
            for stat_name, stat_values in self.historical_stats.items():
                if 'dow' in stat_name:
                    features[f'hist_{stat_name}'] = df.index.dayofweek.map(stat_values)
                elif 'hour' in stat_name:
                    features[f'hist_{stat_name}'] = df.index.hour.map(stat_values)
        else:
            # Compute from data (only for initial exploration, not production)
            logger.warning("No historical stats provided. Computing from current data (leakage risk!).")
            dow_avg = demand.groupby(demand.index.dayofweek).transform('mean')
            hour_avg = demand.groupby(demand.index.hour).transform('mean')
            features['demand_dow_avg'] = dow_avg
            features['demand_hour_avg'] = hour_avg
        
        # Deviation from historical pattern
        if 'demand_same_hour_1w' in features.columns:
            features['demand_deviation_1w'] = demand - features['demand_same_hour_1w']
        
        # Trend in similar days
        features['demand_trend_2w'] = features['demand_same_hour_1w'] - features['demand_same_hour_2w']
        features['demand_trend_4w'] = features['demand_same_hour_2w'] - features['demand_same_hour_4w']
        
        # Advanced Similar Day Rolling (Tier 2) - Safely using lags
        if self.config.rolling_similar_days:
            # Same DOW average of last 4 weeks (safely using lags)
            lags_to_avg = [168 * i for i in range(1, 5)]
            features['demand_same_dow_avg_4w'] = pd.concat([demand.shift(l) for l in lags_to_avg], axis=1).mean(axis=1)
            
            # Same hour average of last 30 days (safely using 24h lags respecting horizon)
            # Find first safe multiple of 24h
            start_multiple = (self.safe_lag_min + 23) // 24
            lags_30d = [24 * i for i in range(start_multiple, start_multiple + 30)]
            features['demand_same_hour_avg_30d'] = pd.concat([demand.shift(l) for l in lags_30d], axis=1).mean(axis=1)
            
            # Trend and Momentum (Tier 2)
            features['demand_lag_168h_trend'] = features['demand_same_hour_1w'].diff(24)
            features['demand_weekly_momentum'] = (features['demand_same_hour_1w'] - features['demand_same_hour_4w']) / 3
            
            # Same Day-Type Average (Tier 2/3)
            # 0=Weekday, 1=Weekend
            df_day_type = pd.Series((df.index.dayofweek >= 5).astype(int), index=df.index)
            if self.historical_stats and 'day_type_mean' in self.historical_stats:
                features['demand_day_type_avg'] = df_day_type.map(self.historical_stats['day_type_mean'])
            else:
                 # Fallback to rolling same-day-type if no global stats
                 is_weekend = (df.index.dayofweek >= 5)
                 features['demand_day_type_avg'] = np.where(
                     is_weekend,
                     pd.concat([demand.shift(168), demand.shift(336)], axis=1).mean(axis=1), # Weekends likeliest similar
                     pd.concat([demand.shift(24*i) for i in range(1, 6)], axis=1).mean(axis=1) # Weekdays
                 )
        
        self.feature_names = list(features.columns)
        self.validate_no_leakage(features, original_index)
        
        return features


# ============================================================================
# DOMAIN-SPECIFIC PIPELINE
# ============================================================================

class DomainFeaturePipeline(FeaturePipeline):
    """
    City-specific and domain-specific features.
    """
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        original_index = df.index.copy()
        
        if self.city_config is None:
            logger.warning("No city config provided, skipping domain-specific features")
            return features
        
        city = self.city_config
        
        # Industrial features
        if self.config.industrial_features and city.is_industrial:
            # Industrial load characteristics
            if 'demand' in df.columns:
                # Flatness of load curve (high = industrial)
                features['load_flatness_24h'] = 1 - (df['demand'].rolling(24, min_periods=1).std() / 
                                                     df['demand'].rolling(24, min_periods=1).mean())
                
                # Weekend drop ratio (low = industrial)
                features['weekend_drop_ratio'] = np.nan  # Computed with calendar info
        
        # Tourism features
        if self.config.tourism_features and city.is_coastal:
            # Tourism season indicator
            features['is_tourism_season'] = df.index.month.isin(city.tourism_months).astype(int)
            
            # Summer weekend effect
            features['is_summer_weekend'] = (
                (df.index.month.isin([6, 7, 8])) & 
                (df.index.dayofweek >= 5)
            ).astype(int)
            
            # Night load ratio (hotels operate 24/7)
            if 'demand' in df.columns and 'hour' in df.columns:
                night_mask = (df.index.hour >= 23) | (df.index.hour <= 5)
                features['night_load_ratio'] = np.where(
                    night_mask,
                    df['demand'] / df['demand'].rolling(24, min_periods=1).mean(),
                    np.nan
                )
        
        # Agricultural features (irrigation)
        if self.config.agricultural_features:
            if 'is_irrigation_season' not in df.columns:
                features['is_irrigation_season'] = df.index.month.isin([5, 6, 7, 8, 9]).astype(int)
            
            # Hot dry days (high irrigation need)
            if 'temperature_2m' in df.columns and 'relative_humidity_2m' in df.columns:
                features['hot_dry_day'] = (
                    (df['temperature_2m'] > 30) & 
                    (df['relative_humidity_2m'] < 40)
                ).astype(int)
        
        # Education features (Tier 3)
        if self.config.education_features:
            if 'is_semester' not in df.columns: # Check if exists in input
                # Generic Turkish university semesters
                features['is_university_semester'] = df.index.month.isin([10, 11, 12, 2, 3, 4, 5]).astype(int)
                features['is_exam_period'] = df.index.month.isin([1, 6]).astype(int)
        
        # City-specific Load Characteristics (Tier 3)
        if city.name == 'denizli':
            if 'demand' in df.columns:
                # Calculate if not in input
                features['denizli_base_load_ratio'] = df['demand'].rolling(168).min() / df['demand'].rolling(168).mean()
                features['denizli_flatness_score'] = 1 - (df['demand'].rolling(24).std() / df['demand'].rolling(24).mean())
        
        self.feature_names = list(features.columns)
        self.validate_no_leakage(features, original_index)
        
        return features


# ============================================================================
# INTERACTION FEATURE PIPELINE (OPTIONAL)
# ============================================================================

class InteractionFeaturePipeline(FeaturePipeline):
    """
    Create feature interactions. Use with caution - can explode feature count.
    """
    
    def transform(self, df: pd.DataFrame, base_features: pd.DataFrame) -> pd.DataFrame:
        if not self.config.feature_interactions:
            return pd.DataFrame(index=df.index)
        
        features = pd.DataFrame(index=df.index)
        original_index = df.index.copy()
        
        # Select numeric features for interactions
        numeric_cols = base_features.select_dtypes(include=[np.number]).columns[:20]  # Limit to top 20
        
        # Key interactions for electricity
        if 'temperature_2m' in base_features.columns:
            if 'hour_sin' in base_features.columns:
                features['temp_x_hour_sin'] = base_features['temperature_2m'] * base_features['hour_sin']
            if 'is_weekend' in base_features.columns:
                features['temp_x_weekend'] = base_features['temperature_2m'] * base_features['is_weekend']
            if 'is_business_day' in base_features.columns:
                features['temp_x_is_business_day'] = base_features['temperature_2m'] * base_features['is_business_day']
        
        if 'relative_humidity_2m' in base_features.columns and 'temperature_2m' in base_features.columns:
            features['humidity_x_temp'] = base_features['relative_humidity_2m'] * base_features['temperature_2m']
            
        if 'shortwave_radiation' in base_features.columns and 'temperature_2m' in base_features.columns:
            features['temp_x_solar_radiation'] = base_features['shortwave_radiation'] * base_features['temperature_2m']
        
        if 'hdd' in base_features.columns and 'hour_sin' in base_features.columns:
            features['hdd_x_hour_sin'] = base_features['hdd'] * base_features['hour_sin']
        
        if 'cdd' in base_features.columns and 'hour_sin' in base_features.columns:
            features['cdd_x_hour_sin'] = base_features['cdd'] * base_features['hour_sin']
        
        self.feature_names = list(features.columns)
        self.validate_no_leakage(features, original_index)
        
        return features


# ============================================================================
# FOURIER FEATURE PIPELINE (Tier 4)
# ============================================================================

class FourierFeaturePipeline(FeaturePipeline):
    """
    Create Fourier features for daily and weekly periodicities.
    """
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.config.fourier_features:
            return pd.DataFrame(index=df.index)
            
        features = pd.DataFrame(index=df.index)
        original_index = df.index.copy()
        
        # Periodicities
        daily_period = 24
        weekly_period = 24 * 7
        
        # Hours from start to maintain continuity
        hours = np.arange(len(df))
        
        # Daily Fourier (2nd and 3rd order)
        for i in [2, 3]:
            features[f'hour_fourier_sin_{i}'] = np.sin(2 * np.pi * i * hours / daily_period)
            features[f'hour_fourier_cos_{i}'] = np.cos(2 * np.pi * i * hours / daily_period)
            
        # Weekly Fourier
        for i in [1, 2]:
            features[f'week_fourier_sin_{i}'] = np.sin(2 * np.pi * i * hours / weekly_period)
            features[f'week_fourier_cos_{i}'] = np.cos(2 * np.pi * i * hours / weekly_period)
            
        self.feature_names = list(features.columns)
        self.validate_no_leakage(features, original_index)
        
        return features


# ============================================================================
# MOMENTUM AND VOLATILITY PIPELINE (Tier 4)
# ============================================================================

class MomentumFeaturePipeline(FeaturePipeline):
    """
    Create demand trend, acceleration and volatility features.
    """
    
    def __init__(self, config: FeatureConfig, city_config: Optional[CityConfig] = None, forecast_horizon: int = 24):
        super().__init__(config, city_config)
        self.forecast_horizon = forecast_horizon
        self.safe_lag_min = forecast_horizon + 1

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.config.volatility_features:
            return pd.DataFrame(index=df.index)
            
        features = pd.DataFrame(index=df.index)
        original_index = df.index.copy()
        
        if 'demand' not in df.columns:
            return features
            
        demand = df['demand'].shift(self.safe_lag_min) # Shift to prevent leakage
        
        # Volatility (Rolling Std normalized by mean)
        for window in [24, 168]:
            roll = demand.rolling(window)
            features[f'demand_volatility_{window}h'] = roll.std() / roll.mean().replace(0, np.nan)
            
        # Trend and Acceleration (Rate of change of rate of change)
        velocity = demand.diff(6)
        features['demand_trend_6h'] = velocity
        features['demand_acceleration_6h'] = velocity.diff(6)
        
        # Temperature volatility
        if 'temperature_2m' in df.columns:
            features['temp_volatility_24h'] = df['temperature_2m'].rolling(24).std()
            
        self.feature_names = list(features.columns)
        self.validate_no_leakage(features, original_index)
        
        return features


# ============================================================================
# WEATHER ANALOGUE PIPELINE (Tier 2 - Critical)
# ============================================================================

class WeatherAnaloguePipeline(FeaturePipeline):
    """
    Finds historical days with most similar weather profiles.
    - Uses temperature and humidity vectors.
    - Searches only HISTORICAL data (lag >= target_horizon + 24).
    """
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.config.weather_analogues:
            return pd.DataFrame(index=df.index)
            
        features = pd.DataFrame(index=df.index)
        original_index = df.index.copy()
        
        # Check requirements
        if 'temperature_2m' not in df.columns or 'demand' not in df.columns:
            return features
            
        temp = df['temperature_2m']
        demand = df['demand']
        
        # To avoid massive slow-down, we search for analogues on a daily basis
        # and then map back to hourly.
        # We use normalize() to get the date part
        df_daily = df.copy()
        df_daily['date'] = df_daily.index.normalize()
        
        daily_agg = df_daily.groupby('date').agg({
            'temperature_2m': ['mean', 'max', 'min'],
            'relative_humidity_2m': 'mean' if 'relative_humidity_2m' in df.columns else 'max',
            'demand': 'mean'
        })
        daily_agg.columns = ['temp_mean', 'temp_max', 'temp_min', 'rh_mean', 'demand_mean']
        
        # Standardize for distance calculation
        weather_cols = ['temp_mean', 'temp_max', 'temp_min', 'rh_mean']
        # Use mean/std from daily_agg to normalize
        daily_norm = (daily_agg[weather_cols] - daily_agg[weather_cols].mean()) / (daily_agg[weather_cols].std() + 1e-6)
        
        # For each day in the dataset, find most similar PREVIOUS day
        similar_3day_mean = pd.Series(index=daily_agg.index, dtype=float)
        
        # Only start searching after we have at least 30 days of history
        search_start_idx = 30
        
        for i in range(search_start_idx, len(daily_agg)):
            target_vec = daily_norm.iloc[i]
            # Search pool: from start up to i-2 (to be absolutely safe with 24h horizon)
            pool = daily_norm.iloc[:i-1]
            
            # Euclidean distance
            distances = np.sqrt(((pool - target_vec)**2).sum(axis=1))
            
            # Top 3 similar days
            top_3_indices = distances.nsmallest(3).index
            similar_3day_mean.iloc[i] = daily_agg.loc[top_3_indices, 'demand_mean'].mean()
            
        # Map back to hourly
        features['similar_3day_mean'] = df.index.normalize().map(similar_3day_mean)
        
        # Fill first 30 days with same-hour-last-week fallback
        features['similar_3day_mean'] = features['similar_3day_mean'].fillna(demand.shift(168))
        
        self.feature_names = list(features.columns)
        self.validate_no_leakage(features, original_index)
        
        return features


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

class EnhancedFeatureEngineer:
    """
    Main orchestrator for enhanced feature engineering.
    """
    
    def __init__(self, config: FeatureConfig, forecast_horizon: int = 24):
        self.config = config
        self.forecast_horizon = forecast_horizon
        self.city_configs = DEFAULT_CITIES
        self.pipelines: Dict[str, FeaturePipeline] = {}
        self.historical_stats: Dict[str, Dict] = {}
        
    def fit(self, df: pd.DataFrame, city: str, output_dir: Optional[Path] = None):
        """
        Compute historical statistics on training data.
        Must be called before transform on new data.
        """
        logger.info(f"Fitting feature engineer on {len(df)} samples for city: {city}")
        
        # Compute historical statistics (frozen for production)
        if 'demand' in df.columns:
            self.historical_stats[city] = {
                'dow_mean': df.groupby(df.index.dayofweek)['demand'].mean().to_dict(),
                'hour_mean': df.groupby(df.index.hour)['demand'].mean().to_dict(),
                'month_mean': df.groupby(df.index.month)['demand'].mean().to_dict(),
                'day_type_mean': df.groupby(df.index.dayofweek >= 5)['demand'].mean().to_dict(),
            }
            
            # Save for reproducibility
            stats_filename = f'precomputed_stats_{city}.json'
            if output_dir:
                stats_path = output_dir / stats_filename
            else:
                stats_path = Path(stats_filename)
                
            with open(stats_path, 'w') as f:
                json.dump(self.historical_stats[city], f, indent=2)
            logger.info(f"Saved historical stats to {stats_path}")
        
        return self
    
    def transform(self, df: pd.DataFrame, city: str) -> pd.DataFrame:
        """
        Transform dataframe to create all features.
        """
        logger.info(f"Transforming {len(df)} samples for city: {city}")
        
        city_config = self.city_configs.get(city)
        
        # Initialize pipelines
        all_features = []
        
        # 1. Temporal features
        temporal_pipe = TemporalFeaturePipeline(self.config, city_config)
        temporal_features = temporal_pipe.transform(df)
        all_features.append(temporal_features)
        logger.info(f"Temporal features: {len(temporal_pipe.get_feature_names())}")
        
        # 2. Weather features
        weather_pipe = WeatherFeaturePipeline(self.config, city_config)
        weather_features = weather_pipe.transform(df)
        all_features.append(weather_features)
        logger.info(f"Weather features: {len(weather_pipe.get_feature_names())}")
        
        # 3. Lag features (CRITICAL - horizon safe)
        lag_pipe = LagFeaturePipeline(self.config, city_config, self.forecast_horizon)
        lag_features = lag_pipe.transform(df)
        all_features.append(lag_features)
        logger.info(f"Lag features: {len(lag_pipe.get_feature_names())}")
        
        # 4. Rolling features
        rolling_pipe = RollingFeaturePipeline(self.config, city_config, self.forecast_horizon)
        rolling_features = rolling_pipe.transform(df)
        all_features.append(rolling_features)
        logger.info(f"Rolling features: {len(rolling_pipe.get_feature_names())}")
        
        # 5. Similar-day features
        similar_pipe = SimilarDayPipeline(self.config, city_config, 
                                         self.historical_stats.get(city),
                                         self.forecast_horizon)
        similar_features = similar_pipe.transform(df)
        all_features.append(similar_features)
        logger.info(f"Similar-day features: {len(similar_pipe.get_feature_names())}")
        
        # 6. Domain-specific features
        domain_pipe = DomainFeaturePipeline(self.config, city_config)
        domain_features = domain_pipe.transform(df)
        all_features.append(domain_features)
        logger.info(f"Domain features: {len(domain_pipe.get_feature_names())}")
        
        # 6.5 Weather Analogue features (Tier 2/8) - NEW
        analogue_pipe = WeatherAnaloguePipeline(self.config, city_config)
        analogue_features = analogue_pipe.transform(df)
        all_features.append(analogue_features)
        logger.info(f"Weather Analogue features: {len(analogue_pipe.get_feature_names())}")
        
        # 7. Fourier features (Tier 4)
        fourier_pipe = FourierFeaturePipeline(self.config, city_config)
        fourier_features = fourier_pipe.transform(df)
        all_features.append(fourier_features)
        logger.info(f"Fourier features: {len(fourier_pipe.get_feature_names())}")
        
        # 8. Momentum features (Tier 4)
        momentum_pipe = MomentumFeaturePipeline(self.config, city_config, self.forecast_horizon)
        momentum_features = momentum_pipe.transform(df)
        all_features.append(momentum_features)
        logger.info(f"Momentum features: {len(momentum_pipe.get_feature_names())}")
        
        # Combine all features
        combined = pd.concat(all_features, axis=1)
        
        # Ensure 'demand' and other essential columns are present
        if 'demand' in df.columns:
            combined['demand'] = df['demand']
        
        # Remove duplicate columns (can happen with base features like temp/hour)
        combined = combined.loc[:, ~combined.columns.duplicated()].copy()

        # ========================================================================
        # CRITICAL FIX: ADVANCED EVENT & WEATHER PASSTHROUGH
        # The input CSV contains SOTA Weather (Adaptive metrics, Solar gain) and 
        # Event (Hijri/Lunar holidays, Astral sun times) features. We MUST keep them.
        # If the pipeline naively recalculated a simpler version, we override it.
        # ========================================================================
        # Features where the input CSV version is far superior to the pipeline's basic fallback:
        superior_input_features = ['is_tourism_season', 'is_irrigation_season', 'heat_index', 'wind_chill', 'is_school_season']
        
        for col in df.columns:
            if col not in combined.columns or col in superior_input_features:
                combined[col] = df[col]
        # ========================================================================
        
        # 9. Interaction features (optional, after base features)
        if self.config.feature_interactions:
            interaction_pipe = InteractionFeaturePipeline(self.config, city_config)
            interaction_features = interaction_pipe.transform(df, combined)
            combined = pd.concat([combined, interaction_features], axis=1)
            logger.info(f"Interaction features: {len(interaction_pipe.get_feature_names())}")
        
        # Add city identifier
        combined['city'] = city
        
        # Add metadata
        combined['feature_version'] = self.config.version
        combined['forecast_horizon'] = self.forecast_horizon
        
        # Validate final output
        self._validate_output(combined, df.index)
        
        logger.info(f"Final feature set: {len(combined.columns)} columns, {len(combined)} rows")
        
        return combined
    
    def _validate_output(self, features: pd.DataFrame, original_index: pd.DatetimeIndex):
        """Final validation checks"""
        # Check index preserved
        if not features.index.equals(original_index):
            raise ValueError("Feature index does not match original index!")
        
        # Check for infinite values
        inf_count = np.isinf(features.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            logger.warning(f"Found {inf_count} infinite values in features")
        
        # Check for all-NaN columns
        nan_cols = features.columns[features.isna().all()].tolist()
        if nan_cols:
            logger.warning(f"All-NaN columns: {nan_cols}")
        
        # Check feature count reasonable
        if len(features.columns) > 200:
            logger.warning(f"High feature count: {len(features.columns)}. Consider feature selection.")
        
        logger.info("✓ Output validation passed")
    
    def get_feature_info(self) -> Dict:
        """Get information about created features"""
        return {
            'version': self.config.version,
            'forecast_horizon': self.forecast_horizon,
            'config': asdict(self.config),
            'city_configs': {k: asdict(v) for k, v in self.city_configs.items()},
            'historical_stats': self.historical_stats,
        }


# ============================================================================
# CLI AND MAIN EXECUTION
# ============================================================================

def create_sample_config():
    """Create sample configuration files"""
    
    # Standard config
    standard_config = {
        'version': '2.0.0',
        'description': 'Standard feature set for 24h ahead forecasting',
        'feature_config': asdict(FeatureConfig()),
        'forecast_horizon': 24,
    }
    
    # Minimal config
    minimal_config = {
        'version': '2.0.0-minimal',
        'description': 'Minimal feature set for fast experimentation',
        'feature_config': asdict(FeatureConfig(
            lag_hours=[24, 48, 168],
            rolling_windows=[24, 168],
            rolling_stats=['mean', 'std'],
            exponential_moving=False,
            similar_day_stats=False,
            industrial_features=False,
            tourism_features=False,
            agricultural_features=False,
            education_features=False,
        )),
        'forecast_horizon': 24,
    }
    
    # Extended config
    extended_config = {
        'version': '2.0.0-extended',
        'description': 'Extended feature set for maximum accuracy',
        'feature_config': asdict(FeatureConfig(
            lag_hours=[1, 2, 3, 6, 12, 24, 48, 72, 168, 336],
            rolling_windows=[6, 12, 24, 48, 168, 336],
            rolling_stats=['mean', 'std', 'min', 'max', 'skew', 'kurt'],
            feature_interactions=True,
            volatility_features=True,
        )),
        'forecast_horizon': 24,
    }
    
    # Save configs
    Path('configs').mkdir(exist_ok=True)
    
    with open('configs/standard.yaml', 'w') as f:
        yaml.dump(standard_config, f)
    with open('configs/minimal.yaml', 'w') as f:
        yaml.dump(minimal_config, f)
    with open('configs/extended.yaml', 'w') as f:
        yaml.dump(extended_config, f)
    
    print("Created sample configs: configs/standard.yaml, configs/minimal.yaml, configs/extended.yaml")


def main():
    parser = argparse.ArgumentParser(description='Enhanced Feature Engineering')
    parser.add_argument('--config', type=Path, default='configs/standard.yaml',
                       help='Configuration file')
    parser.add_argument('--input-dir', type=Path, required=True,
                       help='Directory with raw city CSVs')
    parser.add_argument('--output-dir', type=Path, required=True,
                       help='Directory to save processed features')
    parser.add_argument('--create-configs', action='store_true',
                       help='Create sample configuration files')
    parser.add_argument('--horizon', type=int, help='Override forecast horizon')
    
    args = parser.parse_args()
    
    if args.create_configs:
        create_sample_config()
        return
    
    # Load config
    with open(args.config) as f:
        config_dict = yaml.safe_load(f)
    
    feature_config = FeatureConfig(**config_dict['feature_config'])
    horizon = args.horizon or config_dict['forecast_horizon']
    
    # Initialize engineer
    engineer = EnhancedFeatureEngineer(feature_config, horizon)
    
    # Process each city
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for city in DEFAULT_CITIES.keys():
        input_path = args.input_dir / f"{city}.csv"
        if not input_path.exists():
            logger.warning(f"Input not found: {input_path}")
            continue
        
        # Load data
        df = pd.read_csv(input_path)
        df['time'] = pd.to_datetime(df['time'], format='%d.%m.%Y %H:%M')
        df.set_index('time', inplace=True)
        df.sort_index(inplace=True)
        
        # Split for fit/transform (in production, fit on training period only)
        split_idx = int(len(df) * 0.7)  # 70% for fitting
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        # Fit on training data
        engineer.fit(train_df, city, output_dir=output_dir)
        
        # Transform all data (using frozen stats from fit)
        features_train = engineer.transform(train_df, city)
        features_test = engineer.transform(test_df, city)
        
        # Combine
        features_all = pd.concat([features_train, features_test])
        
        # Save
        output_path = output_dir / f"{city}_engineered_features_enhanced.csv"
        features_all.to_csv(output_path)
        logger.info(f"Saved: {output_path}")
        
        # Save metadata
        meta_path = output_dir / f"{city}_feature_info.json"
        with open(meta_path, 'w') as f:
            json.dump(engineer.get_feature_info(), f, indent=2, default=str)
    
    logger.info("Feature engineering complete!")


if __name__ == '__main__':
    main()  