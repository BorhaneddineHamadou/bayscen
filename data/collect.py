"""
Data Collection Module - Frost API Integration

This module handles fetching raw weather and road condition data from
Frost API (Norwegian Meteorological Institute) using two stations:
1. Main station - Road conditions, wind, precipitation, visibility
2. Secondary station - Cloudiness data

Author: BayScen Research
"""

import requests
from requests.auth import HTTPBasicAuth
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import time


class FrostAPICollector:
    """Collector for Frost API (Norwegian Meteorological Institute)"""
    
    def __init__(self, client_id: str, client_secret: str):
        """
        Initialize Frost API collector
        
        Args:
            client_id: Frost API client ID
            client_secret: Frost API client secret
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.base_url = "https://frost.met.no/observations/v0.jsonld"
        





    def create_time_chunks(self, start_time: str, end_time: str, chunk_months: int = 2) -> List[Tuple[str, str]]:
        """
        Split time range into chunks to avoid API limits
        
        Args:
            start_time: Start time in ISO format
            end_time: End time in ISO format
            chunk_months: Months per chunk (default: 2)
            
        Returns:
            List of (start, end) time tuples
        """
        start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        
        chunks = []
        current = start_dt
        
        while current < end_dt:
            # Calculate next chunk end (approximately chunk_months months)
            next_date = current + timedelta(days=chunk_months * 30)
            if next_date > end_dt:
                next_date = end_dt
            
            chunks.append((
                current.strftime('%Y-%m-%dT%H:%M:%SZ'),
                next_date.strftime('%Y-%m-%dT%H:%M:%SZ')
            ))
            current = next_date
        
        return chunks

    def fetch_station_data(
        self,
        station_id: str,
        elements: List[str],
        start_time: str,
        end_time: str,
        chunk_months: int = 2
    ) -> pd.DataFrame:
        """
        Fetch data from a Frost API station with automatic chunking
        
        Args:
            station_id: Station identifier (e.g., 'SN84770')
            elements: List of elements to fetch
            start_time: Start time in ISO format
            end_time: End time in ISO format
            chunk_months: Months per chunk to avoid API limits (default: 2)
            
        Returns:
            DataFrame with fetched data
        """
        print(f"\nFetching data from station {station_id}...")
        print(f"Elements: {elements}")
        print(f"Period: {start_time} to {end_time}")
        
        elements_str = ','.join(elements)
        
        # Create time chunks
        time_chunks = self.create_time_chunks(start_time, end_time, chunk_months)
        print(f"Splitting into {len(time_chunks)} chunks to avoid API limits...")
        
        all_records = []
        
        for i, (chunk_start, chunk_end) in enumerate(time_chunks, 1):
            print(f"  Chunk {i}/{len(time_chunks)}: {chunk_start[:10]} to {chunk_end[:10]}")
            
            url = (
                f"{self.base_url}?"
                f"sources={station_id}&"
                f"elements={elements_str}&"
                f"referencetime={chunk_start}/{chunk_end}"
            )
            
            response = requests.get(
                url,
                auth=HTTPBasicAuth(self.client_id, self.client_secret)
            )
            
            if response.status_code != 200:
                print(f"  Warning: Error in chunk {i}: {response.status_code}")
                continue
            
            data = response.json()
            
            if 'data' not in data:
                print(f"  Warning: No data in chunk {i}")
                continue
            
            # Parse observations from this chunk
            for obs in data['data']:
                timestamp = obs['referenceTime']
                
                obs_dict = {'timestamp': timestamp}
                
                for obs_data in obs['observations']:
                    element_id = obs_data['elementId']
                    value = obs_data['value']
                    obs_dict[element_id] = value
                
                all_records.append(obs_dict)
            
            print(f"  ✓ Fetched {len(data['data'])} observations")
        
        if not all_records:
            print(f"Warning: No data returned from station {station_id}")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_records)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"✓ Total: {len(df)} records from {station_id}")
        
        return df





def collect_full_dataset(
    frost_client_id: str,
    frost_client_secret: str,
    station_id_main: str,
    station_id_cloud: str,
    start_time: str,
    end_time: str,
    output_dir: Path
) -> Dict[str, Path]:
    """
    Collect complete dataset from Frost API stations
    
    Args:
        frost_client_id: Frost API client ID
        frost_client_secret: Frost API client secret
        station_id_main: Main Frost station ID (for road conditions)
        station_id_cloud: Secondary Frost station ID (for cloudiness)
        start_time: Start time (ISO format)
        end_time: End time (ISO format)
        output_dir: Directory to save raw data
        
    Returns:
        Dictionary of output file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_files = {}
    
    # 1. Fetch main station data (road conditions, wind, precipitation)
    print("\n" + "="*60)
    print("STEP 1: Fetching main station data (road conditions)")
    print("="*60)
    
    frost_collector = FrostAPICollector(frost_client_id, frost_client_secret)
    
    main_elements = [
        'wind_speed',
        'road_friction',
        'road_surface_condition',
        'visibility_in_air_mor',
        'road_water_film_thickness',
        'sum(precipitation_amount PT10M)'
    ]
    
    df_main = frost_collector.fetch_station_data(
        station_id=station_id_main,
        elements=main_elements,
        start_time=start_time,
        end_time=end_time
    )
    
    if not df_main.empty:
        main_file = output_dir / "frost_main_station.csv"
        df_main.to_csv(main_file, index=False)
        output_files['main_station'] = main_file
        print(f"\n✓ Saved main station data: {main_file}")
    
    # 2. Fetch cloudiness data from secondary station
    print("\n" + "="*60)
    print("STEP 2: Fetching cloudiness data (secondary station)")
    print("="*60)
    
    cloud_elements = ['cloud_area_fraction1']
    
    df_cloud = frost_collector.fetch_station_data(
        station_id=station_id_cloud,
        elements=cloud_elements,
        start_time=start_time,
        end_time=end_time
    )
    
    if not df_cloud.empty:
        cloud_file = output_dir / "frost_cloud_station.csv"
        df_cloud.to_csv(cloud_file, index=False)
        output_files['cloud_station'] = cloud_file
        print(f"\n✓ Saved cloud station data: {cloud_file}")
    
    print("\n" + "="*60)
    print("DATA COLLECTION COMPLETE")
    print("="*60)
    print(f"\nRaw data files saved in: {output_dir}")
    for name, path in output_files.items():
        print(f"  - {name}: {path.name}")
    
    return output_files


if __name__ == "__main__":
    # Example usage
    print("Data Collection Module")
    print("Use the CLI tool (python cli.py collect) or import this module")