"""
BayScen Data Collection CLI

Command-line interface for collecting and processing weather data
for the BayScen scenario generation framework.

Usage:
    python cli.py collect --config config.yaml
    python cli.py process --input raw/ --output processed/
    python cli.py full --config config.yaml
"""

import argparse
import sys
from pathlib import Path
import yaml
import pandas as pd

from collect import collect_full_dataset
from process import WeatherDataProcessor


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def cmd_collect(args):
    """Execute data collection"""
    print("\n" + "="*70)
    print("BAYSCEN DATA COLLECTION")
    print("="*70)
    
    # Load configuration
    config = load_config(args.config)
    
    # Extract configuration
    frost_config = config['frost_api']
    location_config = config['location']
    time_config = config['time_period']
    output_config = config.get('output', {})
    
    # Create output directory
    output_dir = Path(args.output) if args.output else Path(output_config.get('raw_dir', 'raw'))
    
    # Collect data
    output_files = collect_full_dataset(
        frost_client_id=frost_config['client_id'],
        frost_client_secret=frost_config['client_secret'],
        station_id_main=location_config['frost_station_main'],
        station_id_cloud=location_config['frost_station_cloud'],
        start_time=time_config['start'],
        end_time=time_config['end'],
        output_dir=output_dir
    )
    
    print("\n✓ Collection complete!")
    print(f"\nRaw data saved in: {output_dir}")
    
    return 0


def cmd_process(args):
    """Execute data processing"""
    print("\n" + "="*70)
    print("BAYSCEN DATA PROCESSING")
    print("="*70)
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load raw data
    print(f"\nLoading raw data from: {input_dir}")
    
    main_file = input_dir / "frost_main_station.csv"
    cloud_file = input_dir / "frost_cloud_station.csv"
    
    if not main_file.exists():
        print(f"Error: Main station data not found: {main_file}")
        return 1
    
    df_main = pd.read_csv(main_file)
    df_main['timestamp'] = pd.to_datetime(df_main['timestamp'], utc=True)
    print(f"✓ Loaded main station data: {len(df_main)} records")
    
    df_cloud = None
    if cloud_file.exists():
        df_cloud = pd.read_csv(cloud_file)
        df_cloud['timestamp'] = pd.to_datetime(df_cloud['timestamp'], utc=True)
        print(f"✓ Loaded cloud station data: {len(df_cloud)} records")
    
    # Process data
    processor = WeatherDataProcessor()
    df_processed = processor.process_full_pipeline(df_main, df_cloud)
    
    # Save processed data
    output_file = output_dir / "bayscen_final_data.csv"
    df_processed.to_csv(output_file, index=False)
    
    print(f"\n✓ Processing complete!")
    print(f"\nProcessed data saved: {output_file}")
    print(f"Final dataset: {df_processed.shape[0]} rows × {df_processed.shape[1]} columns")
    
    return 0


def cmd_full(args):
    """Execute full pipeline: collect + process"""
    print("\n" + "="*70)
    print("BAYSCEN FULL DATA PIPELINE")
    print("="*70)
    
    # Step 1: Collect
    print("\n[1/2] COLLECTING RAW DATA...")
    collect_args = argparse.Namespace(
        config=args.config,
        output=args.raw_dir if hasattr(args, 'raw_dir') else None
    )
    result = cmd_collect(collect_args)
    if result != 0:
        return result
    
    # Step 2: Process
    print("\n[2/2] PROCESSING DATA...")
    
    # Load config to get directories
    config = load_config(args.config)
    output_config = config.get('output', {})
    
    raw_dir = Path(args.raw_dir) if hasattr(args, 'raw_dir') and args.raw_dir else Path(output_config.get('raw_dir', 'raw'))
    processed_dir = Path(args.processed_dir) if hasattr(args, 'processed_dir') and args.processed_dir else Path(output_config.get('processed_dir', 'processed'))
    
    process_args = argparse.Namespace(
        input=str(raw_dir),
        output=str(processed_dir)
    )
    result = cmd_process(process_args)
    
    if result == 0:
        print("\n" + "="*70)
        print("✓ FULL PIPELINE COMPLETE")
        print("="*70)
        print(f"\nRaw data: {raw_dir}/")
        print(f"Processed data: {processed_dir}/bayscen_final_data.csv")
    
    return result


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="BayScen Data Collection and Processing CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect raw data using config file
  python cli.py collect --config config.yaml
  
  # Process raw data
  python cli.py process --input raw/ --output processed/
  
  # Run full pipeline
  python cli.py full --config config.yaml
  
  # Specify custom directories
  python cli.py full --config config.yaml --raw-dir data/raw --processed-dir data/processed
"""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Collect command
    collect_parser = subparsers.add_parser('collect', help='Collect raw weather data')
    collect_parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='Path to configuration YAML file'
    )
    collect_parser.add_argument(
        '--output',
        type=Path,
        help='Output directory for raw data (default: from config or "raw/")'
    )
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process raw data')
    process_parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Input directory with raw data'
    )
    process_parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output directory for processed data'
    )
    
    # Full pipeline command
    full_parser = subparsers.add_parser('full', help='Run full pipeline (collect + process)')
    full_parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='Path to configuration YAML file'
    )
    full_parser.add_argument(
        '--raw-dir',
        type=Path,
        help='Directory for raw data (default: from config or "raw/")'
    )
    full_parser.add_argument(
        '--processed-dir',
        type=Path,
        help='Directory for processed data (default: from config or "processed/")'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    if args.command == 'collect':
        return cmd_collect(args)
    elif args.command == 'process':
        return cmd_process(args)
    elif args.command == 'full':
        return cmd_full(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())