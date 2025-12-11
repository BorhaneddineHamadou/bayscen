# Quick Start Guide - BayScen Data Collection

Get weather data for BayScen in 5 minutes.

## Prerequisites

1. **Python 3.8+** installed
2. **Frost API Credentials** (see below)

## Step 1: Get Frost API Credentials (5 minutes)

1. Go to: https://frost.met.no/auth/requestCredentials.html
2. Fill out the form (research/academic use)
3. You'll receive `client_id` and `client_secret` by email

## Step 2: Install Dependencies (30 seconds)

```bash
pip install -r requirements.txt
```

## Step 3: Configure (2 minutes)

Edit `config.yaml`:

```yaml
frost_api:
  client_id: "YOUR_FROST_CLIENT_ID"      # ← Paste here
  client_secret: "YOUR_FROST_CLIENT_SECRET"  # ← Paste here

# Leave the rest as default for Narvik, Norway (good winter data)
```

## Step 4: Run Collection (5-10 minutes)

```bash
python cli.py full --config config.yaml
```

This will:
1. Fetch raw data from Frost API → `raw/`
2. Process and normalize → `processed/bayscen_final_data.csv`

## Step 5: Verify Output

Check the processed data:

```bash
python -c "import pandas as pd; df = pd.read_csv('processed/bayscen_final_data.csv'); print(f'✓ {len(df)} hourly records'); print(df.head())"
```

Expected output: ~30,000-45,000 rows for 1 year of data

## Done! 🎉

Your data is now ready for BayScen:

```python
import pandas as pd

# Load data
df = pd.read_csv('processed/bayscen_final_data.csv')

# Use in BayScen modeling module
# ... (see BayScen main documentation)
```

## Troubleshooting

### "Invalid credentials"
- Double-check your Frost API credentials
- Verify you copied them correctly (no extra spaces)
- Wait a few minutes after receiving email (sometimes takes time to activate)

### "No cloudiness data"
- This is normal if the secondary station doesn't have cloud measurements
- The processing will use default values (50% cloudiness)
- Try a different secondary station ID with cloud measurements

### "No data returned"
- Check your station IDs in `config.yaml`
- Verify the time period (stations may not have data for all dates)
- Try the default stations: SN84770 (main), SN90450 (cloud)

## Next Steps

- Use `data_collection_tutorial.ipynb` for detailed walkthrough
- See main README.md for complete documentation
- Customize stations and time periods in config.yaml
- Explore the BayScen modeling module to train Bayesian Networks

## Support

For API issues:
- Frost: https://frost.met.no/support

For module issues:
- Check README.md troubleshooting section
- Review tutorial notebook for examples