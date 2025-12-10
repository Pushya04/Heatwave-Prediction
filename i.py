"""
India-region extraction for heatwave ML features from Global_TAVG_Gridded_1deg.nc

Based on paper: Predicting maximum temperatures over India 10-days ahead using ML models

Outputs india.csv with columns:
  - date, year, month, day, day_of_year
  - latitude, longitude
  - temperature_anomaly, climatology, temperature_celsius
  - areal_weight, land_mask
  - temp_anom_prev_day, temp_anom_diff_1d

Assumptions:
  - Dataset variables: temperature(time, lat, lon), climatology(12, lat, lon),
    land_mask(lat, lon), areal_weight(lat, lon), latitude(lat), longitude(lon), time(time)
  - Four-month window defaults to March–June (MAMJ), configurable via MONTHS_TO_KEEP.
  - India bounds: lat [6, 38], lon [68, 98] (degrees East/North).
  
IMPORTANT: This version ensures NO missing values in the output dataset.
Missing values are handled through:
  1. Imputation of temperature anomalies (0.0 for missing)
  2. Validation that climatology exists for all land cells
  3. Filtering out any rows with invalid critical fields
"""

import os
from typing import Tuple
from datetime import datetime

import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date
from tqdm import tqdm


# === CONFIG ===
NETCDF_PATH = os.path.join(os.path.dirname(__file__), "Global_TAVG_Gridded_1deg.nc")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "india.csv")

# India geographic bounds (deg)
INDIA_LAT_MIN, INDIA_LAT_MAX = 6.0, 38.0
INDIA_LON_MIN, INDIA_LON_MAX = 68.0, 98.0

# Four-month season for Indian heatwaves. Default MAMJ (March–June)
# Based on paper methodology - adjust if paper specifies different months
MONTHS_TO_KEEP: Tuple[int, int, int, int] = (3, 4, 5, 6)

# Stream settings
TIME_CHUNK = 20  


def to_numpy(x):
    """Convert masked array to regular numpy array, filling masked values with NaN."""
    if isinstance(x, np.ma.MaskedArray):
        return x.filled(np.nan)
    return np.asarray(x)


def fill_missing_values(data: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """Replace NaN values with fill_value."""
    data = np.asarray(data)
    return np.where(np.isnan(data), fill_value, data)


def validate_and_fill_climatology(clim_data: np.ndarray, land_mask: np.ndarray) -> np.ndarray:
    """Ensure climatology has valid values for all land cells. Fill with regional median if needed."""
    clim_data = np.asarray(clim_data).copy()
    land_mask = np.asarray(land_mask)
    
    # For land cells where climatology is missing, use regional median
    land_cells = land_mask >= 0.5
    missing_land = land_cells & np.isnan(clim_data)
    
    if np.any(missing_land):
        # Calculate median climatology from valid land cells
        valid_land_values = clim_data[land_cells & ~np.isnan(clim_data)]
        if len(valid_land_values) > 0:
            fill_value = np.median(valid_land_values)
        else:
            fill_value = 25.0  # Default fallback
        clim_data[missing_land] = fill_value
    
    return clim_data


def main():
    print("=" * 70)
    print("India Region Data Extraction for Heatwave Prediction")
    print("=" * 70)
    
    if not os.path.exists(NETCDF_PATH):
        raise FileNotFoundError(f"NetCDF file not found: {NETCDF_PATH}")

    # Open NetCDF file
    print(f"\nOpening NetCDF file: {NETCDF_PATH}")
    ds = Dataset(NETCDF_PATH, mode="r")
    
    # Extract coordinates
    print("Reading coordinates...")
    lat_full = to_numpy(ds.variables['latitude'][:])
    lon_full = to_numpy(ds.variables['longitude'][:])
    time_var = ds.variables['time']
    time_vals = to_numpy(time_var[:])
    
    # Find India region indices
    lat_mask = (lat_full >= INDIA_LAT_MIN) & (lat_full <= INDIA_LAT_MAX)
    lon_mask = (lon_full >= INDIA_LON_MIN) & (lon_full <= INDIA_LON_MAX)
    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]
    
    if lat_idx.size == 0 or lon_idx.size == 0:
        raise RuntimeError("India bounds produced empty selection. Check lat/lon bounds.")
    
    print(f"India region: {len(lat_idx)} lat points, {len(lon_idx)} lon points")
    print(f"  Latitude range: {lat_full[lat_idx].min():.1f}° to {lat_full[lat_idx].max():.1f}°")
    print(f"  Longitude range: {lon_full[lon_idx].min():.1f}° to {lon_full[lon_idx].max():.1f}°")
    
    # Extract static fields for India region only
    print("\nReading static fields...")
    climatology_full = to_numpy(ds.variables['climatology'][:])  # (12, lat, lon)
    land_mask_full = to_numpy(ds.variables['land_mask'][:])      # (lat, lon)
    areal_weight_full = to_numpy(ds.variables['areal_weight'][:]) # (lat, lon)
    
    # Subset static fields to India region
    clim_india = climatology_full[:, lat_idx, :][:, :, lon_idx]  # (12, latInd, lonInd)
    land_india = land_mask_full[np.ix_(lat_idx, lon_idx)]        # (latInd, lonInd)
    area_india = areal_weight_full[np.ix_(lat_idx, lon_idx)]     # (latInd, lonInd)
    
    # Fill missing values in static fields
    print("Filling missing values in static fields...")
    # Fill missing areal_weight with 1.0 (default)
    area_india = fill_missing_values(area_india, fill_value=1.0)
    # Fill missing land_mask with 0.0 (assume ocean if missing)
    land_india = fill_missing_values(land_india, fill_value=0.0)
    
    # Validate and fill climatology for each month
    print("Validating climatology for all months...")
    for month_idx in range(12):
        clim_india[month_idx] = validate_and_fill_climatology(
            clim_india[month_idx], land_india
        )
    
    # Create meshgrid for India coordinates
    lon_grid, lat_grid = np.meshgrid(lon_full[lon_idx], lat_full[lat_idx])
    lat_flat = lat_grid.flatten()
    lon_flat = lon_grid.flatten()
    land_flat = land_india.flatten()
    area_flat = area_india.flatten()
    
    npoints_india = len(lat_flat)
    print(f"Total India grid cells: {npoints_india:,}")
    print(f"Land cells: {np.sum(land_flat >= 0.5):,}")
    
    # Convert time to datetime objects
    print("\nConverting time to dates...")
    have_datetime = False
    time_datetimes = None
    
    # Try multiple methods to parse time
    try:
        # Method 1: Standard CF convention
        time_datetimes = num2date(time_vals, units=time_var.units, 
                                  calendar=getattr(time_var, 'calendar', 'standard'))
        # Convert to pandas datetime
        if isinstance(time_datetimes, (list, tuple)):
            time_datetimes = pd.to_datetime(time_datetimes)
        else:
            time_datetimes = pd.to_datetime(np.array(time_datetimes))
        have_datetime = True
        print(f"Successfully converted time. Range: {time_datetimes[0]} to {time_datetimes[-1]}")
    except Exception as e1:
        try:
            # Method 2: Try with different calendar
            time_datetimes = num2date(time_vals, units=time_var.units, calendar='gregorian')
            if isinstance(time_datetimes, (list, tuple)):
                time_datetimes = pd.to_datetime(time_datetimes)
            else:
                time_datetimes = pd.to_datetime(np.array(time_datetimes))
            have_datetime = True
            print(f"Successfully converted time (gregorian). Range: {time_datetimes[0]} to {time_datetimes[-1]}")
        except Exception as e2:
            # Method 3: Parse time units manually (e.g., "days since 1800-01-01")
            try:
                units_str = time_var.units.lower()
                if 'since' in units_str:
                    # Extract reference date
                    parts = units_str.split('since')
                    ref_str = parts[1].strip().split()[0]  # Get date part
                    ref_date = pd.to_datetime(ref_str)
                    time_datetimes = ref_date + pd.to_timedelta(time_vals, unit='d')
                    have_datetime = True
                    print(f"Successfully converted time (manual parsing). Range: {time_datetimes[0]} to {time_datetimes[-1]}")
                else:
                    raise ValueError("Cannot parse time units")
            except Exception as e3:
                print(f"Warning: Could not convert time to datetime: {e1}")
                print("Using decimal year representation...")
    
    # Prepare output
    print(f"\nPreparing output CSV: {OUTPUT_CSV}")
    cols = [
        'date', 'year', 'month', 'day', 'day_of_year',
        'latitude', 'longitude',
        'temperature_anomaly', 'climatology', 'temperature_celsius',
        'areal_weight', 'land_mask',
        'temp_anom_prev_day', 'temp_anom_diff_1d',
    ]
    
    # Try to remove existing file, but continue if it's locked
    file_cleared = False
    if os.path.exists(OUTPUT_CSV):
        try:
            os.remove(OUTPUT_CSV)
            file_cleared = True
        except (PermissionError, OSError) as e:
            print(f"Warning: Could not remove existing file (may be open in another program): {e}")
            print("Please close india.csv in other programs and try again.")
            return
    
    # Write header
    with open(OUTPUT_CSV, 'w', encoding='utf-8') as f:
        f.write(','.join(cols) + '\n')
    
    # Variable handles
    temp_var = ds.variables['temperature']  # (time, lat, lon)
    ntime = len(time_vals)
    
    # Track previous anomaly per grid cell for day-to-day differences
    last_anom_by_cell = np.full(npoints_india, np.nan, dtype=np.float32)
    
    # Statistics
    total_rows = 0
    skipped_months = 0
    
    print("\nExtracting data (streaming in chunks)...")
    print(f"Processing {ntime} time steps in chunks of {TIME_CHUNK}")
    
    # Stream over time in chunks
    for t0 in tqdm(range(0, ntime, TIME_CHUNK), desc="Time chunks"):
        t1 = min(ntime, t0 + TIME_CHUNK)
        
        # Read full time chunk for entire globe
        temp_chunk_global = to_numpy(temp_var[t0:t1, :, :])  # (chunk_size, lat_full, lon_full)
        
        # Subset to India region using numpy advanced indexing
        temp_chunk = temp_chunk_global[:, lat_idx, :][:, :, lon_idx]  # (chunk_size, latInd, lonInd)
        
        # Process each time step in chunk
        for rel_idx in range(temp_chunk.shape[0]):
            global_t_idx = t0 + rel_idx
            
            # Get date information
            if have_datetime:
                ts = pd.Timestamp(time_datetimes[global_t_idx])
                year = int(ts.year)
                month = int(ts.month)
                day = int(ts.day)
                day_of_year = int(ts.dayofyear)
                date_str = ts.strftime('%Y-%m-%d')
            else:
                # Fallback: treat time as decimal year
                year_decimal = float(time_vals[global_t_idx])
                year = int(year_decimal)
                month = int((year_decimal - year) * 12) + 1
                if month < 1 or month > 12:
                    month = 1
                day = 1
                day_of_year = int((month - 1) * 30.4) + 1  # Approximate
                date_str = f"{year:04d}-{month:02d}-{day:02d}"
            
            # Check if month is in our target months
            if month not in MONTHS_TO_KEEP:
                skipped_months += 1
                continue
            
            month_idx = month - 1  # 0-indexed for climatology
            
            # Extract temperature anomaly for this time step (2D: latInd x lonInd)
            temp_anom_2d = temp_chunk[rel_idx, :, :]
            temp_anom_flat = temp_anom_2d.flatten()
            
            # Fill missing temperature anomalies with 0.0 (no anomaly)
            temp_anom_flat = fill_missing_values(temp_anom_flat, fill_value=0.0)
            
            # Get climatology for this month (2D: latInd x lonInd)
            clim_2d = clim_india[month_idx, :, :]
            clim_flat = clim_2d.flatten()
            
            # Climatology should already be filled, but ensure no NaN
            clim_flat = fill_missing_values(clim_flat, fill_value=25.0)  # Default fallback
            
            # Calculate absolute temperature: anomaly + climatology
            temp_celsius_flat = temp_anom_flat + clim_flat
            
            # Ensure no NaN in temperature (shouldn't happen after filling, but use climatology as fallback)
            # If temp_celsius is NaN, it means both components were somehow NaN, so use climatology
            nan_mask = np.isnan(temp_celsius_flat)
            if np.any(nan_mask):
                temp_celsius_flat[nan_mask] = clim_flat[nan_mask]
            
            # Day-to-day differences
            # Initialize previous anomaly with 0.0 if NaN (for first occurrence)
            prev_anom_flat = np.where(
                np.isnan(last_anom_by_cell),
                0.0,
                last_anom_by_cell
            )
            diff_flat = temp_anom_flat - prev_anom_flat
            # Ensure no NaN in diff
            diff_flat = fill_missing_values(diff_flat, fill_value=0.0)
            
            # Build DataFrame for this time step
            df_step = pd.DataFrame({
                'date': [date_str] * npoints_india,
                'year': [year] * npoints_india,
                'month': [month] * npoints_india,
                'day': [day] * npoints_india,
                'day_of_year': [day_of_year] * npoints_india,
                'latitude': lat_flat,
                'longitude': lon_flat,
                'temperature_anomaly': temp_anom_flat.astype(np.float32),
                'climatology': clim_flat.astype(np.float32),
                'temperature_celsius': temp_celsius_flat.astype(np.float32),
                'areal_weight': area_flat.astype(np.float32),
                'land_mask': land_flat.astype(np.float32),
                'temp_anom_prev_day': prev_anom_flat.astype(np.float32),
                'temp_anom_diff_1d': diff_flat.astype(np.float32),
            })
            
            # Keep only land points (land_mask >= 0.5)
            df_step = df_step[df_step['land_mask'] >= 0.5].copy()
            
            # Additional validation: remove any rows with NaN (shouldn't happen, but safety check)
            initial_count = len(df_step)
            df_step = df_step.dropna()
            if len(df_step) < initial_count:
                print(f"Warning: Removed {initial_count - len(df_step)} rows with NaN at date {date_str}")
            
            # Validate critical fields have valid values
            critical_fields = ['temperature_celsius', 'climatology', 'latitude', 'longitude', 'month']
            for field in critical_fields:
                if field in df_step.columns:
                    df_step = df_step[df_step[field].notna()].copy()
            
            if len(df_step) > 0:
                # Final check: ensure no NaN values remain
                if df_step.isna().any().any():
                    print(f"Warning: NaN values detected at {date_str}, filtering...")
                    df_step = df_step.dropna()
                
                # Append to CSV only if we have valid rows
                if len(df_step) > 0:
                    df_step.to_csv(OUTPUT_CSV, mode='a', index=False, header=False)
                    total_rows += len(df_step)
            
            # Update last anomaly for next iteration (use current filled values)
            last_anom_by_cell = temp_anom_flat.copy()
    
    # Close dataset
    ds.close()
    
    # Print summary
    print("\n" + "=" * 70)
    print("Extraction Complete!")
    print("=" * 70)
    print(f"Output file: {OUTPUT_CSV}")
    print(f"Total rows written: {total_rows:,}")
    print(f"Time steps skipped (outside target months): {skipped_months:,}")
    print(f"Target months: {MONTHS_TO_KEEP}")
    
    if os.path.exists(OUTPUT_CSV):
        file_size_mb = os.path.getsize(OUTPUT_CSV) / (1024 * 1024)
        print(f"File size: {file_size_mb:.2f} MB")
        
        # Comprehensive data quality check
        print("\n" + "=" * 70)
        print("Data Quality Validation")
        print("=" * 70)
        
        # Read full dataset for validation
        print("Reading full dataset for validation...")
        df_full = pd.read_csv(OUTPUT_CSV)
        
        print(f"\nTotal rows in output: {len(df_full):,}")
        
        # Check for missing values
        print("\nMissing values check:")
        missing_counts = df_full.isna().sum()
        has_missing = missing_counts[missing_counts > 0]
        if len(has_missing) == 0:
            print("  ✓ No missing values found in any column!")
        else:
            print("  ⚠ Warning: Missing values detected:")
            for col, count in has_missing.items():
                print(f"    {col}: {count:,} ({100*count/len(df_full):.2f}%)")
        
        # Validate India region bounds
        print("\nGeographic bounds validation:")
        lat_min, lat_max = df_full['latitude'].min(), df_full['latitude'].max()
        lon_min, lon_max = df_full['longitude'].min(), df_full['longitude'].max()
        print(f"  Latitude range: {lat_min:.1f}° to {lat_max:.1f}°")
        print(f"  Longitude range: {lon_min:.1f}° to {lon_max:.1f}°")
        if (lat_min >= INDIA_LAT_MIN and lat_max <= INDIA_LAT_MAX and 
            lon_min >= INDIA_LON_MIN and lon_max <= INDIA_LON_MAX):
            print("  ✓ All coordinates within India bounds")
        else:
            print("  ⚠ Warning: Some coordinates outside India bounds!")
        
        # Validate months
        print("\nMonth validation:")
        unique_months = sorted(df_full['month'].unique())
        print(f"  Months in dataset: {unique_months}")
        if set(unique_months).issubset(set(MONTHS_TO_KEEP)):
            print(f"  ✓ All months are within target months {MONTHS_TO_KEEP}")
        else:
            print(f"  ⚠ Warning: Found months outside target {MONTHS_TO_KEEP}")
        
        # Temperature statistics
        print("\nTemperature statistics:")
        print(f"  Temperature (Celsius) range: {df_full['temperature_celsius'].min():.2f}°C to {df_full['temperature_celsius'].max():.2f}°C")
        print(f"  Mean temperature: {df_full['temperature_celsius'].mean():.2f}°C")
        print(f"  Temperature anomaly range: {df_full['temperature_anomaly'].min():.2f} to {df_full['temperature_anomaly'].max():.2f}")
        print(f"  Valid temperature values: {df_full['temperature_celsius'].notna().sum():,}/{len(df_full):,} ({100*df_full['temperature_celsius'].notna().sum()/len(df_full):.1f}%)")
        
        # Date range
        print("\nDate range:")
        df_full['date'] = pd.to_datetime(df_full['date'], errors='coerce')
        date_min = df_full['date'].min()
        date_max = df_full['date'].max()
        print(f"  {date_min} to {date_max}")
        
        # Sample data preview
        print("\nSample data (first 5 rows):")
        print(df_full.head().to_string())
        
        print("\n" + "=" * 70)
        print("Validation Complete!")
        print("=" * 70)


if __name__ == "__main__":
    main()
