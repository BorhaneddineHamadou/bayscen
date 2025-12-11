# BayScen Data Collection Module

Collect and process real-world weather data for autonomous vehicle testing scenario generation.

## Overview

This module fetches weather and road condition data from the Norwegian Meteorological Institute (Frost API) and processes it into a format suitable for training Bayesian Networks in the BayScen framework.

### Data Source

**Frost API** (Norwegian Meteorological Institute)
- Official documentation: https://frost.met.no/
- Credentials: https://frost.met.no/auth/requestCredentials.html
- **Main Station**: Road conditions, wind, precipitation, visibility
- **Secondary Station**: Cloudiness data

### Why Two Frost Stations?

The main station (e.g., SN84770) provides comprehensive road condition data but may lack cloudiness measurements. A secondary station with cloudiness data is used to supplement the dataset, ensuring complete weather parameter coverage.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

1. Register for Frost API credentials:
   - Visit: https://frost.met.no/auth/requestCredentials.html

2. Edit `config.yaml`:

```yaml
frost_api:
  client_id: "YOUR_FROST_CLIENT_ID"
  client_secret: "YOUR_FROST_CLIENT_SECRET"

location:
  frost_station_main: "SN84770"    # Main station with road data
  frost_station_cloud: "SN90450"   # Station with cloudiness

time_period:
  start: "2020-12-01T00:00:00Z"
  end: "2024-03-03T23:59:59Z"
```

### Command Line Usage

**Collect and process in one command:**
```bash
python cli.py full --config config.yaml
```

**Or run steps separately:**

```bash
# Step 1: Collect raw data
python cli.py collect --config config.yaml

# Step 2: Process raw data
python cli.py process --input raw/ --output processed/
```

**Custom directories:**
```bash
python cli.py full --config config.yaml --raw-dir data/raw --processed-dir data/processed
```

### Python API Usage

```python
from pathlib import Path
from collect import collect_full_dataset
from process import WeatherDataProcessor
import pandas as pd

# Step 1: Collect raw data
output_files = collect_full_dataset(
    frost_client_id="YOUR_CLIENT_ID",
    frost_client_secret="YOUR_CLIENT_SECRET",
    station_id_main="SN84770",
    station_id_cloud="SN90450",
    start_time="2020-12-01T00:00:00Z",
    end_time="2024-03-03T23:59:59Z",
    output_dir=Path("raw")
)

# Step 2: Load raw data
df_main = pd.read_csv("raw/frost_main_station.csv")
df_main['timestamp'] = pd.to_datetime(df_main['timestamp'], utc=True)

df_cloud = pd.read_csv("raw/frost_cloud_station.csv")
df_cloud['timestamp'] = pd.to_datetime(df_cloud['timestamp'], utc=True)

# Step 3: Process data
processor = WeatherDataProcessor()
df_processed = processor.process_full_pipeline(df_main, df_cloud)

# Step 4: Save processed data
df_processed.to_csv("processed/bayscen_final_data.csv", index=False)
```

## Folder Structure

```
data/
├── raw/                              # Raw data from Frost API
│   ├── frost_main_station.csv       # Road conditions, wind, precipitation, visibility
│   └── frost_cloud_station.csv      # Cloudiness data
│
├── processed/                        # Processed data
│   └── bayscen_final_data.csv       # Final dataset for BayScen
│
├── collect.py                        # Data collection module
├── process.py                        # Data processing module
├── cli.py                            # Command-line interface
├── config.yaml                       # Configuration file
├── data_collection_tutorial.ipynb   # Tutorial notebook
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Output Data Format

The final processed dataset (`bayscen_final_data.csv`) contains hourly weather observations with the following columns:

| Column | Range | Description |
|--------|-------|-------------|
| `Time of Day` | -90 to 90 | Sun altitude angle (degrees) |
| `Cloudiness` | 0-100 | Cloud coverage percentage |
| `Precipitation` | 0, 20, 40, 60, 80, 100 | Precipitation intensity level |
| `Wind Intensity` | 0, 20, 40, 60, 80, 100 | Wind strength level |
| `Fog Density` | 0, 20, 40, 60, 80, 100 | Fog thickness level |
| `Fog Distance` | 0, 20, 40, 60, 80, 100 | Visibility distance level |
| `Wetness` | 0, 20, 40, 60, 80, 100 | Road surface wetness |
| `Precipitation Deposits` | 0, 20, 40, 60, 80, 100 | Water/snow on road |
| `Road Friction` | 0.0-1.0 | Tire grip coefficient |

### Discretization Details

**Precipitation (mm/hour):**
- 0: No precipitation
- 20: Light (0.1-1.0 mm)
- 40: Light-Moderate (1.1-2.5 mm)
- 60: Moderate (2.6-5.0 mm)
- 80: Heavy (5.1-10.0 mm)
- 100: Very Heavy (>10.0 mm)

**Road Wetness (Frost API codes):**
- 0: Dry (256)
- 20: Moist (512)
- 40: Wet (1024)
- 60: Slush (2048)
- 80: Snow (4096)
- 100: Ice (8192)

**Wind Intensity:**
- Percentile-based quintiles (0, 20, 40, 60, 80, 100)
- 0 indicates no wind

**Fog Density & Distance:**
- Derived from visibility measurements
- Inverse relationship: high density = low distance
- Clear conditions: density=0, distance=100

**Road Friction:**
- 0.0: No friction
- 0.1: Very low (0.01-0.15)
- 0.2: Low (0.16-0.30)
- 0.4: Moderate-Low (0.31-0.50)
- 0.6: Moderate-High (0.51-0.70)
- 0.8: High (0.71-0.90)
- 1.0: Very High (>0.90)

**Time of Day (Sun altitude angles):**
- -90: Night (0-5h)
- -60: Dawn (6-8h)
- -30: Early morning (9-11h)
- 0: Midday (12-14h)
- 30: Afternoon (15-17h)
- 60: Dusk (18-20h)
- 90: Late evening (21-23h)

## Data Processing Pipeline

The processing pipeline consists of:

1. **Merging**: Combine main station and cloudiness data by timestamp
2. **Resampling**: Aggregate 10-minute data to hourly intervals
3. **Transformation**: Apply discretization rules to each parameter
4. **Validation**: Remove rows with missing critical values
5. **Formatting**: Output in BayScen-compatible format

## Troubleshooting

### Missing Cloudiness Data

If the secondary Frost station doesn't have cloudiness data:
- Verify the station ID has cloud measurements
- Check alternative nearby stations
- The processor will handle missing values gracefully

### API Rate Limits

Frost API: Generally no strict limits for authenticated requests
- Data collection is respectful with built-in delays
- Large time periods may take 20-40 minutes

### Invalid Station IDs

Search for valid stations at:
```
https://frost.met.no/sources/v0.jsonld?types=SensorSystem&country=NO
```

Filter by:
- `nearestmaxdistance`: Distance from coordinates
- `elements`: Required measurements (e.g., `road_friction`)

## Tutorial Notebook

See `data_collection_tutorial.ipynb` for a step-by-step walkthrough with:
- API setup and authentication
- Data collection from multiple sources
- Processing and transformation
- Data inspection and validation
- Saving final dataset

Run with:
```bash
jupyter notebook data_collection_tutorial.ipynb
```

## Notes

1. **Station Selection**: The main station must have road condition sensors. Station SN84770 (Narvik, Norway) is recommended for winter testing scenarios.

2. **Time Zones**: All timestamps are in UTC. The processing pipeline preserves UTC.

3. **Missing Values**: The pipeline uses forward-fill for missing values, then drops rows with missing cloudiness (if critical).

4. **Data Quality**: Real-world data may have gaps. The final dataset size depends on station uptime and measurement availability.

5. **Privacy**: Frost API data is publicly available for research purposes.

## License

This module is provided as-is for research purposes. API credentials and usage comply with respective service terms.