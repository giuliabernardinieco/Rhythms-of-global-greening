# REQUIREMENTS:
import pandas as pd
from scipy.signal import find_peaks
import numpy as np

def max_power(sumpower_df, period_cut_a, period_cut_b):
        """
        Returns the maximum rectified_power in sumpower_df between period a and b.

        Parameters:
        - sumpower_df (pd.DataFrame): DataFrame containing 'period' and 'rectified_power' columns.
        - period_cut_a (float): Minimum period.
        - period_cut_b (float): Maximum period.

        Returns:
        - float: Maximum rectified_power in the given period range.
        """
        filtered_df = sumpower_df[(sumpower_df["period"] >= period_cut_a) & (sumpower_df["period"] <= period_cut_b)]
        
        if filtered_df.empty:
            print(f"No data found in period range {period_cut_a}-{period_cut_b}.")
            return None
        
        max_val = filtered_df["rectified_power"].max()
        return max_val
    
def period_of_max_power(sumpower_df):
        """
        Returns the period of the maximum rectified_power in sumpower_df.

        Parameters:
        - sumpower_df (pd.DataFrame): DataFrame containing 'period' and 'rectified_power' columns.

        Returns:
        - float: Period of the maximum rectified_power.
        """
        max_row = sumpower_df.loc[sumpower_df["rectified_power"].idxmax()]
        return max_row["period"]
    
def proportion_of_power_under_curve(sumpower_df, period_cut_a, period_cut_b):
        """
        Returns the proportion of rectified_power under the curve between period a and b.

        Parameters:
        - sumpower_df (pd.DataFrame): DataFrame containing 'period' and 'rectified_power' columns.
        - period_cut_a (float): Minimum period.
        - period_cut_b (float): Maximum period.

        Returns:
        - float: Proportion of rectified_power under the curve in the given period range.
        """
        filtered_df = sumpower_df[(sumpower_df["period"] >= period_cut_a) & (sumpower_df["period"] <= period_cut_b)]
        
        if filtered_df.empty:
            print(f"No data found in period range {period_cut_a}-{period_cut_b}.")
            return None
        
        total_power_cuts_a_b = filtered_df["rectified_power"].sum()
        total_power = sumpower_df["rectified_power"].sum()
        
        proportion = total_power_cuts_a_b / total_power
        return proportion


def period_of_second_peak_power(sumpower_df):
    """
    Returns the period of the second peak in rectified_power in sumpower_df.

    Parameters:
    - sumpower_df (pd.DataFrame): DataFrame with 'period' and 'rectified_power' columns.

    Returns:
    - float: Period of the second peak in rectified_power, or None if fewer than 2 peaks exist.
    """
    # sorted by period for correct order
    sorted_df = sumpower_df.sort_values(by="period")

    power = sorted_df["rectified_power"].values
    peaks, _ = find_peaks(power)

    if len(peaks) < 2:
        print("Not enough peaks to identify the second one.")
        return None

    # Get indices of peaks sorted by height
    peak_heights = power[peaks]
    sorted_peak_indices = peaks[np.argsort(peak_heights)[::-1]]  # descending sort

    # Get the second highest peak
    second_peak_idx = sorted_peak_indices[1]
    period_second_peak = sorted_df.iloc[second_peak_idx]["period"]
    return period_second_peak
