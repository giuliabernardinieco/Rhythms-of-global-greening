import pandas as pd
import xarray as xr
import numpy as np
import time
import argparse

from wavelets_wrapper.quick_wavelet import run_full_wavelet_analysis
from utils import max_power, period_of_second_peak_power, proportion_of_power_under_curve, period_of_max_power


# Set up the argument parser
parser = argparse.ArgumentParser(description='Process input CSV and write to output file.')
parser.add_argument('-i', '--input', type=str, required=True, help='Path to the input CSV file')
parser.add_argument('-o', '--output1', type=str, required=True, help='Path to the output CSV file')
parser.add_argument('-w', '--output2', type=str, required=True, help='Path to the second output CSV file')

# Parse arguments
args = parser.parse_args()

# Read the input CSV file
points = pd.read_csv(args.input)
print(points.head())
print(points.shape)

# Open dataset of EVI once with chunking (chunks={"time": -1})
evi = xr.open_dataset("EVI_time_series_scales_interpolated_nearest_v2.nc", chunks={})
print(evi)

# # Define bounding box for testing a smaller area
# # somewhere in tuscany, near Livorno, 4 points
# lat_min, lat_max = 43.3426132, 43.4277591
# lon_min, lon_max = 10.5824939, 10.7169211

# # Filter points to retain inside the bounding box
# df_box = points[(points["latitude"] >= lat_min) & (points["latitude"] <= lat_max) &
#             (points["longitude"] >= lon_min) & (points["longitude"] <= lon_max)]

# print(df_box)
# print(df_box.shape)
df_box = points

# Create empty lists to store results
skipped_coords = []
results_window1 = []
results_window2 = []

for index, row in df_box.iterrows():
    point_lat = row["latitude"]
    point_lon = row["longitude"]
    print(f"Processing point: {point_lat}, {point_lon}")
    
    loop_start_time = time.time()
    
    
    # select the EVI variable
    nc_load_start = time.time()
    
    ts = evi["EVI"].sel(lat=point_lat, lon=point_lon, method="nearest").compute()
    
    nc_load_end = time.time()
    nc_load_duration = nc_load_end - nc_load_start
    print(f" --> Time to load netCDF data: {nc_load_duration:.4f} seconds")



    # extract the EVI values
    extract_start_time = time.time()
    
    sig1 = ts.values  # This is the EVI series
    
    extract_end_time = time.time()
    extract_duration = extract_end_time - extract_start_time
    print(f" --> Time to extract EVI values: {extract_duration:.4f} seconds")
    
    # skip if nans or negatives
    if np.isnan(sig1).any() or np.any(sig1 < -0.2):
        print(" --> Skipped due to NAs or negative values.")
        skipped_coords.append({"latitude": point_lat, "longitude": point_lon})
        continue

    
    # Convert datetime64 index to days since first observation
    dates_conversion_start = time.time()
    
    dates = ts['time'].values
    time_vals = (dates - dates[0]) / np.timedelta64(1, 'D')  # Time in days as float
    
    dates_conversion_end = time.time()
    dates_conversion_duration = dates_conversion_end - dates_conversion_start
    print(f" --> Time to convert dates: {dates_conversion_duration:.4f} seconds")
    
    # before running the wavelet analysis, we need to remove the mean from the signal to make it zero mean.
    # this is important because the wavelet transform is sensitive to the mean of the signal.
    sig1_0_mean = sig1 - np.mean(sig1)

    
    # Start the timer for the wavelet analysis
    analysis_start_time = time.time()
    try:
        ####################################
        ####### RUN WAVELET ANALYSIS #######
        ####################################
        cwtX, scales, xpad, xlen, period, xmirror_df, signal_df, fft_df, fft_inv_df, scales_df, scales_orig_df, sumpower_df, fft_pycwt_df, icwt_df, icwt_band1_df, icwt_part2_df, wavelet_power_df_mirrored = run_full_wavelet_analysis(sig1_0_mean, dt=16, mirror=True, cut1=84, cut2=725, wf='morlet', dj=0.1, om0=6, normmean=False, mirrormethod=3)
        coi = cwtX[3]

        original_length = len(sig1)
        print(f" --> Original signal length: {original_length}")
        start_index = 4 * original_length  # 5th repetition starts at (4 * original_length)
        end_index = 5 * original_length    # 5th repetition ends at (5 * original_length)

        #####################################
        ### DEALING WITH MIRRORRED RESULT ###
        #####################################

        # Extract the 5th repetition from the inverse CWT
        icwt_mean_fixed = icwt_df + xmirror_df.iloc[0, 0] # icwt is first shifted by the first value of the mirror
        icwt_5th_repetition = icwt_mean_fixed.iloc[start_index:end_index] # then we extract the 5th repetition
        icwt_5th_repetition = icwt_5th_repetition+np.mean(sig1) # then we add the vale of the mean of the original signal so that we can compare it with the original signal

        # select only the window-filtered recontructed signal from 5th repetition
        icwt_band1_df = icwt_band1_df.iloc[start_index:end_index]
        icwt_band1_df = icwt_band1_df+np.mean(sig1)+xmirror_df.iloc[0, 0] # add the mean of the original signal

        #####################################
        ###### REMOVING UNWANTED POWER ######
        #####################################
        # Because I mirrored the signal, there ia a lot of power that needs to be removed before plotting everything:

        # - I need to remove the power generated at periods greater than the length of the signal
        # - I also want to isolate power in the central domain (time-wise). 

        extended_time = wavelet_power_df_mirrored["time"].unique()
        # Define windows to remove excess power
        time_min = extended_time[start_index]
        time_max = extended_time[end_index-1]    
        period_min = 32    
        period_max = time_vals[-1]

        wavelet_power_df = wavelet_power_df_mirrored.copy() # Create a copy of the DataFrame so that we can store it separatelly

        # Apply the filter
        wavelet_power_df = wavelet_power_df[
            (wavelet_power_df["time"] >= time_min) & 
            (wavelet_power_df["time"] <= time_max) &
            (wavelet_power_df["period"] >= period_min) &
            (wavelet_power_df["period"] <= period_max)
        ]

        # change time value to restore the original time
        wavelet_power_df["time"] = wavelet_power_df["time"] - time_min
        wavelet_power_df = wavelet_power_df.reset_index()
        # ##########################
        # now that I have the correct time axis, I will divide the dataframe in smaller chunks and calculate multiple sumpow
        # Divide the power plot into 2 parts and compare the power in each part
        # First select windows of days between 0 and 4583, then from 4584 to 9168
        filtered_windows = [
            (0, 4583),
            (4584, 9168)
        ]
        # Create a dictionary to store the time-average power for each window
        time_average_power = {}
        # Loop through the filtered windows and calculate the time-average power
        for i, (start, end) in enumerate(filtered_windows):
            # Filter the data for the current window
            filtered_data = wavelet_power_df[
                (wavelet_power_df["time"] >= start) & (wavelet_power_df["time"] <= end)
            ]
            # save the time-average power for the current window
            time_average_power[f"Window {i+1}"] = filtered_data.groupby("period")[["rectified_power", "power"]].mean().reset_index()
            time_average_power[f"Window {i+1}"]["frequency"] = 1 / time_average_power[f"Window {i+1}"]["period"]

        ############################

        # create two sumpower_df DataFrames for the i windows
        sumpower_df_window1 = time_average_power["Window 1"]
        sumpower_df_window2 = time_average_power["Window 2"]

        # end the timings
        analysis_end_time = time.time()
        analysis_duration = analysis_end_time - analysis_start_time
        print(f" --> Wavelet analysis duration: {analysis_duration:.4f} seconds")  
        
        # Save execution time
        loop_end_time = time.time()
        # execution_times.append({
        #     "latitude": point_lat,
        #     "longitude": point_lon,
        #     "execution_time": loop_end_time - loop_start_time
        # })
        print(f" --> Execution time: {loop_end_time - loop_start_time:.2f} seconds")
        
      # Pivot sumpower_df so each period becomes a key with its rectified_power
        power_by_period_window1 = {
            round(row['period'], 1): row['rectified_power']
            for _, row in sumpower_df_window1.iterrows()
        }
        power_by_period_window2 = {
            round(row['period'], 1): row['rectified_power']
            for _, row in sumpower_df_window2.iterrows()
        }

        # Prepare the result row with lat/lon and rectified_power for each period
        result_row_window1 = {
            'latitude': row['latitude'],
            'longitude': row['longitude'],
            'execution_time': loop_end_time - loop_start_time,
            **power_by_period_window1  # merge period-power pairs into the dict
        }
        result_row_window2 = {
            'latitude': row['latitude'],
            'longitude': row['longitude'],
            'execution_time': loop_end_time - loop_start_time,
            **power_by_period_window2  # merge period-power pairs into the dict
        }

        # Append to the results list
        results_window1.append(result_row_window1)
        results_window2.append(result_row_window2)

    except Exception as e:
        print(f" --> Error during analysis: {e}")
        continue


# Save results
pd.DataFrame(results_window1).to_csv(args.output1, index=False)
pd.DataFrame(results_window2).to_csv(args.output2, index=False)
