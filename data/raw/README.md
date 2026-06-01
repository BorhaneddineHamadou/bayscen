# Raw Data Directory

This directory contains raw data fetched from Frost API before any processing.

## Files

After running data collection, you will find:

### frost_main_station.csv
Raw data from the main Frost API station with:
- `timestamp`: Observation time (UTC)
- `wind_speed`: Wind speed in m/s
- `road_friction`: Road friction coefficient (0-1)
- `road_surface_condition`: Surface condition code (256=Dry, 512=Moist, 1024=Wet, 2048=Slush, 4096=Snow, 8192=Ice)
- `visibility_in_air_mor`: Visibility in meters
- `road_water_film_thickness`: Water layer thickness in mm
- `sum(precipitation_amount PT10M)`: 10-minute accumulated precipitation in mm

**Resolution**: 10-minute intervals

### frost_cloud_station.csv
Raw cloudiness data from secondary Frost API station:
- `timestamp`: Observation time (UTC)
- `cloud_area_fraction`: Cloud coverage on 0-8 scale (oktas)

**Resolution**: 10-minute intervals

**Note**: This file may be empty if the secondary station doesn't have cloudiness measurements.

## Data Quality Notes

1. **Completeness**: Raw data may have gaps due to sensor downtime or communication issues
2. **Time Zone**: All timestamps are in UTC
3. **Resolution**: Frost API provides 10-minute data
4. **Coordinates**: Frost stations are at fixed locations; coordinates reflect station positions

## Processing

Raw data should NOT be edited manually. Use the processing pipeline:

```bash
python cli.py process --input raw/ --output processed/
```

This will:
- Merge data sources
- Handle missing values
- Resample to hourly
- Normalize to BayScen format