"""
Data Processing Module - Clean and Normalize Weather Data

This module processes raw weather data into the final BayScen format with:
- Hourly aggregation
- Missing value handling
- Discretization into CARLA-compatible ranges
- Final column naming and ordering

Author: BayScen Research
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


class WeatherDataProcessor:
    """Process and normalize weather data for BayScen"""
    
    def __init__(self):
        """Initialize processor with transformation rules"""
        
        # Road surface condition to wetness mapping (Frost API codes)
        self.road_condition_to_wetness = {
            0.0: 0,       # Error -> Dry
            256.0: 0,     # Dry -> 0%
            512.0: 20,    # Moist -> 20%
            1024.0: 40,   # Wet -> 40%
            2048.0: 60,   # Slush -> 60%
            4096.0: 80,   # Snow -> 80%
            8192.0: 100   # Ice -> 100%
        }
        
    def merge_data_sources(
        self,
        df_main: pd.DataFrame,
        df_cloud: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Merge main station data with cloudiness data
        
        Args:
            df_main: Main station dataframe with road conditions
            df_cloud: Optional cloud station dataframe
            
        Returns:
            Merged dataframe
        """
        print("\nMerging data sources...")
        
        df = df_main.copy()
        
        # Merge cloudiness if available
        if df_cloud is not None and not df_cloud.empty:
            # Rename cloud column if needed
            cloud_cols = [c for c in df_cloud.columns if 'cloud' in c.lower()]
            if cloud_cols:
                df_cloud_clean = df_cloud[['timestamp'] + cloud_cols].copy()
                df_cloud_clean = df_cloud_clean.rename(
                    columns={cloud_cols[0]: 'cloud_area_fraction1'}
                )
                
                # CHANGED: 'left' -> 'outer' to preserve timestamps where main station 
                # might be missing but cloud station has data.
                df = df.merge(df_cloud_clean, on='timestamp', how='outer')
                
                # CHANGED: Must sort by timestamp immediately after outer merge
                # to ensure ffill works correctly later.
                df = df.sort_values('timestamp').reset_index(drop=True)
                
                print(f"✓ Merged cloudiness data from secondary station")
        
        return df
    
    def resample_to_hourly(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample 10-minute data to hourly with proper aggregation
        """
        print("\nResampling to hourly frequency...")
        
        # FIX 1: Sort AND Reset Index. 
        # The notebook did this. Without reset_index, .loc slicing behaves unpredictably.
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Extract hour and minute
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        
        # Find the first row where minute is 00
        hourly_rows = df[df['minute'] == 0]
        if hourly_rows.empty:
            print("Warning: No hourly (:00) timestamps found, using all data")
            first_hourly_idx = 0
        else:
            first_hourly_idx = hourly_rows.index[0]
        
        print(f"Starting from first hourly row: {df.loc[first_hourly_idx, 'timestamp']}")
        
        # Process only from the first hourly row onwards
        df_processed = df.loc[first_hourly_idx:].copy()
        
        # FIX 2: Forward fill ONLY the columns the notebook filled.
        # We explicitly EXCLUDE 'wind_speed' from this list.
        # This ensures wind_speed remains NaN where data is missing, 
        # so those rows get dropped later (matching the notebook).
        cols_to_fill = [
            'cloud_area_fraction1',
            'road_friction',
            'road_surface_condition', 
            'visibility_in_air_mor',
            'road_water_film_thickness'
        ]
        
        for col in cols_to_fill:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].fillna(method='ffill')
        
        # Fill precipitation with 0 (not NaN)
        precip_col = 'sum(precipitation_amount PT10M)'
        if precip_col in df_processed.columns:
            df_processed[precip_col] = df_processed[precip_col].fillna(0)
        
        # Set timestamp as index
        df_processed = df_processed.set_index('timestamp')
        
        # Define aggregation rules
        other_cols = [
            col for col in df_processed.columns
            if col not in [precip_col, 'hour', 'minute']
        ]
        
        agg_dict = {}
        
        # Sum precipitation
        if precip_col in df_processed.columns:
            agg_dict[precip_col] = 'sum'
        
        # Take first value for all other columns (corresponds to :00 time)
        agg_dict.update({col: 'first' for col in other_cols})
        
        print("\nUsing aggregation rules:")
        print(agg_dict)
        
        # Resample to hourly frequency
        df_hourly = df_processed.resample('h').agg(agg_dict)
        
        # Reset index
        df_hourly = df_hourly.reset_index()
        
        print(f"\nHourly data shape: {df_hourly.shape}")
        
        # Drop any row that has at least one NaN value
        print(f"Shape before dropping NaNs: {df_hourly.shape}")
        
        # Because we didn't fill wind_speed, this will now drop the ~1450 rows 
        # where wind was missing, matching the notebook count.
        df_hourly = df_hourly.dropna()
        print(f"Shape after dropping NaNs: {df_hourly.shape}")
        
        print(f"\nFirst few rows of hourly data:")
        print(df_hourly.head())
        
        return df_hourly
    
    def transform_precipitation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform precipitation from mm to discrete levels (0-100)
        
        Ranges:
        - 0: No precipitation
        - 20: Light (0.1-1.0 mm)
        - 40: Light-Moderate (1.1-2.5 mm)
        - 60: Moderate (2.6-5.0 mm)
        - 80: Heavy (5.1-10.0 mm)
        - 100: Very Heavy (>10.0 mm)
        """
        precip_col = 'sum(precipitation_amount PT10M)'
        
        if precip_col not in df.columns:
            print("Warning: Precipitation column not found")
            df['Precipitation'] = 0
            return df
        
        conditions = [
            (df[precip_col] == 0),
            (df[precip_col] > 0) & (df[precip_col] <= 1.0),
            (df[precip_col] > 1.0) & (df[precip_col] <= 2.5),
            (df[precip_col] > 2.5) & (df[precip_col] <= 5.0),
            (df[precip_col] > 5.0) & (df[precip_col] <= 10.0),
            (df[precip_col] > 10.0)
        ]
        
        choices = [0, 20, 40, 60, 80, 100]
        
        df['Precipitation'] = np.select(conditions, choices, default=0)
        
        return df
    
    def transform_wetness(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform road surface condition to wetness percentage"""
        
        if 'road_surface_condition' not in df.columns:
            print("Warning: Road surface condition column not found")
            df['Wetness'] = 0
            return df
        
        df['Wetness'] = df['road_surface_condition'].map(
            self.road_condition_to_wetness
        ).fillna(0).astype(int)
        
        return df
    
    def transform_cloudiness(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform cloudiness from 0-8 scale to 0-100 scale"""
        
        cloud_col = 'cloud_area_fraction1'
        
        if cloud_col not in df.columns:
            print("Warning: Cloudiness column not found, defaulting to 50%")
            df['Cloudiness'] = 50
            return df
        
        # Multiply by 10 to get 0-80 scale, then convert to integer
        df['Cloudiness'] = (df[cloud_col] * 10).fillna(50).astype(int)
        
        return df
    
    def transform_precipitation_deposits(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform road water film thickness to discrete deposits using quintiles
        
        Uses percentile-based binning for non-zero values
        """
        water_col = 'road_water_film_thickness'
        
        if water_col not in df.columns:
            print("Warning: Water film thickness column not found")
            df['Precipitation_Deposits'] = 0
            return df
        
        # Default to 0
        df['Precipitation_Deposits'] = 0
        
        # Only bin non-zero values using quintiles
        non_zero_mask = df[water_col] > 0
        
        if non_zero_mask.sum() > 0:
            df.loc[non_zero_mask, 'Precipitation_Deposits'] = pd.qcut(
                df.loc[non_zero_mask, water_col],
                q=5,
                labels=[20, 40, 60, 80, 100],
                duplicates='drop'
            )
        
        df['Precipitation_Deposits'] = df['Precipitation_Deposits'].astype(int)
        
        return df
    
    def transform_wind_intensity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform wind speed to discrete intensity levels using quintiles
        
        Uses percentile-based binning for non-zero values
        """
        wind_col = 'wind_speed'
        
        if wind_col not in df.columns:
            print("Warning: Wind speed column not found")
            df['Wind_Intensity'] = 0
            return df
        
        df['Wind_Intensity'] = 0
        
        # Only bin non-zero values
        non_zero_mask = df[wind_col] > 0
        
        if non_zero_mask.sum() > 0:
            df.loc[non_zero_mask, 'Wind_Intensity'] = pd.qcut(
                df.loc[non_zero_mask, wind_col],
                q=5,
                labels=[20, 40, 60, 80, 100],
                duplicates='drop'
            )
        
        df['Wind_Intensity'] = df['Wind_Intensity'].astype(int)
        
        return df
    
    def transform_road_friction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform road friction to discrete levels
        
        Ranges:
        - 0.0: No friction
        - 0.1: Very low (0.01-0.15)
        - 0.2: Low (0.16-0.30)
        - 0.4: Moderate-Low (0.31-0.50)
        - 0.6: Moderate-High (0.51-0.70)
        - 0.8: High (0.71-0.90)
        - 1.0: Very High (>0.90)
        """
        friction_col = 'road_friction'
        
        if friction_col not in df.columns:
            print("Warning: Road friction column not found")
            df['Road_Friction'] = 0.8
            return df
        
        conditions = [
            (df[friction_col] == 0),
            (df[friction_col] > 0) & (df[friction_col] <= 0.15),
            (df[friction_col] > 0.15) & (df[friction_col] <= 0.3),
            (df[friction_col] > 0.3) & (df[friction_col] <= 0.5),
            (df[friction_col] > 0.5) & (df[friction_col] <= 0.7),
            (df[friction_col] > 0.7) & (df[friction_col] <= 0.9),
            (df[friction_col] > 0.9)
        ]
        
        choices = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
        
        df['Road_Friction'] = np.select(conditions, choices, default=0.8)
        
        return df
    
    def transform_fog(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Transform visibility to fog density and distance
        
        Fog Density (inverse of visibility):
        - 0: Clear (high visibility)
        - 20-100: Increasing fog density (decreasing visibility)
        
        Fog Distance (direct relationship):
        - 100: Clear (high visibility)
        - 80-0: Decreasing visibility distance
        """
        vis_col = 'visibility_in_air_mor'
        
        if vis_col not in df.columns:
            print("Warning: Visibility column not found, defaulting to clear conditions")
            df['Fog_Density'] = 0
            df['Fog_Distance'] = 100
            return df
        
        # Define fog threshold (e.g., visibility < 10km = 10000m)
        fog_threshold = 10000
        
        # Default to clear conditions
        df['Fog_Density'] = 0
        df['Fog_Distance'] = 100
        
        # Identify foggy conditions
        foggy_mask = df[vis_col] < fog_threshold
        
        if foggy_mask.sum() > 0:
            # Calculate percentile ranks for foggy conditions
            foggy_percentiles = df.loc[foggy_mask, vis_col].rank(pct=True)
            
            # Fog Density: Low visibility -> high density
            df.loc[foggy_mask, 'Fog_Density'] = pd.cut(
                foggy_percentiles,
                bins=5,
                labels=[100, 80, 60, 40, 20],  # Inverse
                include_lowest=True
            )
            
            # Fog Distance: Low visibility -> low distance
            df.loc[foggy_mask, 'Fog_Distance'] = pd.cut(
                foggy_percentiles,
                bins=5,
                labels=[0, 20, 40, 60, 80],  # Direct
                include_lowest=True
            )
        
        df['Fog_Density'] = df['Fog_Density'].astype(int)
        df['Fog_Distance'] = df['Fog_Distance'].astype(int)
        
        return df
    
    def transform_time_of_day(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform hour to sun altitude angle (-90 to 90)
        
        Hour ranges:
        - 0-5: -90 (night)
        - 6-8: -60 (dawn)
        - 9-11: -30 (early morning)
        - 12-14: 0 (midday)
        - 15-17: 30 (afternoon)
        - 18-20: 60 (dusk)
        - 21-23: 90 (late evening)
        """
        if 'timestamp' not in df.columns:
            print("Warning: Timestamp column not found")
            df['Time_of_Day'] = 0
            return df
        
        hour = df['timestamp'].dt.hour
        
        conditions = [
            (hour >= 0) & (hour <= 5),
            (hour >= 6) & (hour <= 8),
            (hour >= 9) & (hour <= 11),
            (hour >= 12) & (hour <= 14),
            (hour >= 15) & (hour <= 17),
            (hour >= 18) & (hour <= 20),
            (hour >= 21) & (hour <= 23)
        ]
        
        choices = [-90, -60, -30, 0, 30, 60, 90]
        
        df['Time_of_Day'] = np.select(conditions, choices, default=0)
        
        return df
    
    def process_full_pipeline(
        self,
        df_main: pd.DataFrame,
        df_cloud: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Run complete processing pipeline
        
        Args:
            df_main: Main station dataframe
            df_cloud: Optional cloud station dataframe
            
        Returns:
            Processed dataframe ready for BayScen
        """
        print("\n" + "="*60)
        print("PROCESSING PIPELINE")
        print("="*60)
        
        # Step 1: Merge data sources
        df = self.merge_data_sources(df_main, df_cloud)
        
        # Step 2: Resample to hourly
        df = self.resample_to_hourly(df)
        
        # Step 3: Transform all columns
        print("\nTransforming columns...")
        df = self.transform_precipitation(df)
        df = self.transform_wetness(df)
        df = self.transform_cloudiness(df)
        df = self.transform_precipitation_deposits(df)
        df = self.transform_wind_intensity(df)
        df = self.transform_road_friction(df)
        df = self.transform_fog(df)
        df = self.transform_time_of_day(df)
        
        # Step 4: Select and order final columns
        final_columns = [
            'Time_of_Day',
            'Cloudiness',
            'Precipitation',
            'Wind_Intensity',
            'Fog_Density',
            'Fog_Distance',
            'Wetness',
            'Precipitation_Deposits',
            'Road_Friction'
        ]
        
        # Keep only columns that exist
        available_cols = [col for col in final_columns if col in df.columns]
        df_final = df[available_cols].copy()
        
        # Reset index
        df_final = df_final.reset_index(drop=True)
        
        print("\n" + "="*60)
        print("PROCESSING COMPLETE")
        print("="*60)
        print(f"\nFinal dataset shape: {df_final.shape}")
        print(f"Columns: {list(df_final.columns)}")
        print(f"\nFirst few rows:")
        print(df_final.head())
        
        # Print value distributions
        print("\n" + "="*60)
        print("VALUE DISTRIBUTIONS")
        print("="*60)
        for col in df_final.columns:
            print(f"\n{col}:")
            print(df_final[col].value_counts().sort_index().head(10))
        
        return df_final


if __name__ == "__main__":
    print("Data Processing Module")
    print("Use the CLI tool (python cli.py process) or import this module")