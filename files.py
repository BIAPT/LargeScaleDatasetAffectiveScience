"""
files_qc.py

Cleaned and documented utilities for loading, QC, cleaning, and preprocessing
physiological sensor files (EDA, BVP, TMP).

Author: Jackie Girgis

Purpose: Accompaniment for "A Large-Scale Dataset of Emotion-Annotated Physiological Signals Collected from a Public Exhibit"
"""

# Standard library
import os
import csv
import time
import logging
import traceback
import warnings
from typing import Tuple, Dict, Any, List, Optional

# 3rd party
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
import heartpy as hp
import neurokit2 as nk

# Configure logging once (users can override in their scripts)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Files:
    """
    Handles loading, QC, cleaning, and preprocessing of physiological sensor data.

    Main responsibilities:
      - Read folder path configuration from a small CSV file (two columns: key,path)
      - Run parallel quality checks for participants described in a 'Participant_videos.csv'
      - Provide per-signal cleaning helpers (clean_eda, clean_bvp, clean_tmp)
      - Save QC summary CSVs

    Example use:
        files = Files("paths_config.csv")
        sensor_df, ppg_df, eda_df, tmp_df = files.quality_assessments_parallel(num_jobs=8)

    Expected configuration CSV format:
        key,path
        Divided,/path/to/divided
        Demographics,/path/to/demographics
        Divided_QC,/path/to/divided_qc
    """

    def __init__(self, filepath: str):
        """
        Initialize file paths based on a 2-column CSV configuration file.

        Args:
            filepath: Path to CSV containing (key,path) rows.
                      Required keys used by this class: 'Divided', 'Demographics', 'Divided_QC'.

        Raises:
            FileNotFoundError or KeyError when config file is missing or required keys are absent.
        """
        self._paths: Dict[str, str] = {}

        # Defensive open/parse
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Config file not found: {filepath}")

        with open(filepath, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    key = row[0].strip()
                    path = row[1].strip()
                    self._paths[key] = path

        # Required keys
        try:
            self.divided = self._paths["Divided"]
            self.demographics = self._paths["Demographics"]
            self.divided_qc = self._paths["Divided_QC"]
        except KeyError as e:
            missing = e.args[0]
            raise KeyError(f"Missing required path in config file: {missing}")

    # -------------------------
    # Parallel QC orchestration
    # -------------------------
    def quality_assessments_parallel(self, num_jobs: int = 4) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Run QC for all participants discovered in Participant_videos.csv using joblib parallel.

        Args:
            num_jobs: number of parallel workers (set to None or -1 to use all cores).

        Returns:
            Tuple of DataFrames: (sensor_df, ppg_df, eda_df, tmp_df) summarizing QC results.

        Notes:
            - Participant_videos.csv must live in self.demographics and contain a 'Keys' column.
            - This function writes CSV summaries to the demographics folder:
                sensor_dict.csv, ppg_dict.csv, eda_dict.csv, tmp_dict.csv
        """
        warnings.filterwarnings("ignore")
        start_time = time.time()

        status_path = os.path.join(self.demographics, "Participant_videos.csv")
        if not os.path.isfile(status_path):
            raise FileNotFoundError(f"Participant_videos.csv not found at {status_path}")

        status_df = pd.read_csv(status_path)
        if "Keys" not in status_df.columns:
            raise ValueError("Participant_videos.csv must contain a 'Keys' column")

        file_list = status_df["Keys"].unique().tolist()
        total_files = len(file_list)
        logging.info(f"Starting parallel QC for {total_files} participants with {num_jobs} jobs...")

        # Run QC in parallel; quality_assessments returns four dicts per participant
        results = Parallel(n_jobs=num_jobs)(
            delayed(self.quality_assessments)(participant_key) for participant_key in file_list
        )

        # Collect lists of dicts
        sensor_dfs, ppg_dfs, eda_dfs, tmp_dfs = [], [], [], []
        for sensor_dict, ppg_dict, eda_dict, tmp_dict in results:
            sensor_dfs.append(sensor_dict)
            ppg_dfs.append(ppg_dict)
            eda_dfs.append(eda_dict)
            tmp_dfs.append(tmp_dict)

        def unpack_data(data_list: List[Dict[str, Any]]) -> pd.DataFrame:
            """
            Convert a list of mapping-dicts into a flat DataFrame with a 'Keys' column.

            Each element of data_list is expected to be a dictionary mapping file_key -> metrics_dict.
            """
            rows: List[Dict[str, Any]] = []
            for d in data_list:
                for key, value in (d or {}).items():
                    row = {"Keys": key}
                    if isinstance(value, dict):
                        row.update(value)
                    else:
                        # If value isn't dict, store it as a single value under 'value'
                        row["value"] = value
                    rows.append(row)
            return pd.DataFrame(rows)

        sensor_df = unpack_data(sensor_dfs)
        ppg_df = unpack_data(ppg_dfs)
        eda_df = unpack_data(eda_dfs)
        tmp_df = unpack_data(tmp_dfs)

        # Save CSVs (safe saving)
        self.save_to_csv(sensor_df, os.path.join(self.demographics, "sensor_dict.csv"))
        self.save_to_csv(ppg_df, os.path.join(self.demographics, "ppg_dict.csv"))
        self.save_to_csv(eda_df, os.path.join(self.demographics, "eda_dict.csv"))
        self.save_to_csv(tmp_df, os.path.join(self.demographics, "tmp_dict.csv"))

        elapsed = time.time() - start_time
        logging.info(f"QC completed in {elapsed:.2f} seconds.")
        return sensor_df, ppg_df, eda_df, tmp_df

    # -------------------------
    # Utilities
    # -------------------------
    def save_to_csv(self, df: pd.DataFrame, path: str):
        """
        Safely save a DataFrame to CSV. If a CSV already exists and contains a 'Keys' index,
        this function will append new rows without duplicating existing keys.

        Args:
            df: DataFrame that must contain a column named 'Keys'.
            path: output CSV path.
        """
        if "Keys" not in df.columns:
            raise ValueError("DataFrame must contain a 'Keys' column to save with save_to_csv()")

        # If the file exists, try to merge without duplicating Keys
        if os.path.isfile(path):
            try:
                existing_df = pd.read_csv(path)
                if "Keys" in existing_df.columns:
                    # keep existing rows, add new unique rows
                    new_rows_df = df[~df["Keys"].isin(existing_df["Keys"])]
                    combined_df = pd.concat([existing_df, new_rows_df], ignore_index=True)
                    combined_df.to_csv(path, index=False)
                    return
            except Exception:
                logging.warning(f"Could not merge with existing CSV at {path}; overwriting.")
        # Default: write new
        df.to_csv(path, index=False)

    # -------------------------
    # Per-signal QC helpers
    # -------------------------
    def check_temp_quality(self, participant: str, video_id: str, tmp_dict: Dict[str, dict]) -> Tuple[bool, Optional[pd.DataFrame], Dict[str, dict]]:
        """
        Quick quality checks for skin temperature (TMP).

        Args:
            participant: participant id string (without video prefix)
            video_id: 'baseline' or a video id like '3.01'
            tmp_dict: dictionary to be updated with QC outcomes

        Returns:
            eliminated: bool (True if unusable)
            df: loaded DataFrame (or None on error)
            tmp_dict: updated dictionary

        Notes:
            - Assumes CSV file named '{video_id}_{participant}.csv' with 'TMP' column.
            - Uses simple heuristics: percent values <20°C, unrealistic min/max, short duration.
        """
        key = f"{video_id}_{participant}"
        tmp_dict.setdefault(key, {})
        participant_filename = f"{key}.csv"
        participant_filepath = os.path.join(self.divided, participant_filename)
        eliminated = False
        df = None

        try:
            df = pd.read_csv(participant_filepath)
            if "TMP" not in df.columns or "EDA" not in df.columns:
                raise ValueError("Expected 'TMP' and 'EDA' columns in TMP file.")

            # Trim first 5 seconds of baseline if needed (75 samples @15Hz)
            if video_id == "baseline" and len(df) > 75:
                df = df.iloc[75:].reset_index(drop=True)

            tmp = df["TMP"].values
            duration = len(df) / 15.0  # seconds

            # Heuristics
            temp_below_20_pct = (tmp < 20).mean() * 100
            if temp_below_20_pct > 9:
                eliminated = True
                tmp_dict[key]["Reason"] = f"{temp_below_20_pct:.1f}% values < 20°C"
            elif round(float(tmp.min())) == 10 and round(float(tmp.max())) == 40:
                eliminated = True
                tmp_dict[key]["Reason"] = "Min/Max reflect device error (10/40)"
            elif duration < 10:
                eliminated = True
                tmp_dict[key]["Reason"] = "Too short (<10s)"
            else:
                tmp_dict[key]["Passed"] = True

        except Exception as e:
            eliminated = True
            tmp_dict[key]["Eliminated"] = True
            tmp_dict[key]["Error"] = repr(e)
            logging.debug(f"check_temp_quality error for {key}: {e}")
            df = None

        return eliminated, df, tmp_dict

    def check_bvp_quality(self, participant: str, video_id: str, ppg_dict: Dict[str, dict]) -> Tuple[bool, Dict[str, dict], Optional[pd.DataFrame], Optional[np.ndarray], Optional[np.ndarray], Optional[int], Optional[int]]:
        """
        Evaluate the quality of a participant's BVP (PPG) signal and return a usable trimmed segment.

        Args:
            participant: participant id string (without video prefix)
            video_id: 'baseline' or video id like '3.01'
            ppg_dict: dictionary to store metrics

        Returns:
            eliminated: bool
            ppg_dict: updated dictionary
            useable_df: trimmed DataFrame (or None)
            hp_filtered_ppg: filtered signal (or None)
            peaks_without_over_79: peak indices after removing >79 artifacts (or None)
            start_index, end_index: indices delimiting the usable segment (or None)

        Warning:
            - This method uses multiple heuristics that are tailored to your dataset (e.g., >79 amplitude removals).
            - neurokit2.ppg_process returns a signals DataFrame and info dict; behavior depends on NK version.
        """
        participant_key = f"{video_id}_{participant}"
        ppg_dict.setdefault(participant_key, {})
        filename = f"{participant_key}.csv"
        filepath = os.path.join(self.divided, filename)

        # Initialize default returns
        eliminated = True
        useable_df = None
        hp_filtered_ppg = None
        peaks_without_over_79 = None
        start_index = None
        end_index = None

        try:
            df = pd.read_csv(filepath)
            if "BVP" not in df.columns:
                raise ValueError("Expected 'BVP' column in BVP file.")

            # Trim baseline warmup if present
            if video_id == "baseline" and len(df) > 75:
                df = df.iloc[75:].reset_index(drop=True)

            bvp = df["BVP"].values.astype(float)

            # HeartPy filter (bandpass). Note: heartpy.filter_signal returns numpy array.
            hp_filtered_ppg = hp.filter_signal(
                bvp,
                cutoff=[0.8, 2.5],
                filtertype="bandpass",
                sample_rate=15,
                order=4,
                return_top=False
            )
            # Restore DC to filtered signal (approx)
            hp_filtered_ppg = hp_filtered_ppg + np.mean(bvp)

            # Process and detect peaks with NeuroKit2 (signals dataframe, info)
            signals, info = nk.ppg_process(hp_filtered_ppg, sampling_rate=15, method_peaks="bishop", method_cleaning="none", correct_artifacts=True)

            # neurokit2 returns signals with 'PPG_Rate' and 'PPG_Peaks' columns in common versions
            if "PPG_Peaks" not in signals.columns:
                raise ValueError("ppg_process did not produce 'PPG_Peaks' column; check neurokit2 version.")

            # Keep PPG_Clean and PPG_Rate
            signals["PPG_Clean"] = hp_filtered_ppg
            signals["PPG_Rate"] = signals.get("PPG_Rate", np.nan).round(0)

            orig_duration = len(signals) / 15.0

            # Compute instantaneous HR from peak-to-peak intervals
            peak_indices = np.where(signals["PPG_Peaks"].values == 1)[0]
            if len(peak_indices) < 3:
                # Not enough peaks to derive stable HR
                raise ValueError("Too few peaks detected in PPG signal.")

            peak_times = peak_indices / 15.0
            time_diffs = np.diff(peak_times)
            hr = 60.0 / time_diffs
            # Fill HR values in signals (shifted to align with intervals)
            signals["HR"] = np.nan
            if len(hr) > 0:
                # Assign HR from second peak onward
                signals.loc[peak_indices[1:], "HR"] = np.concatenate(([np.nan], hr))[: len(peak_indices)-1]
            signals["HR"] = signals["HR"].bfill().ffill()

            under_40_pct = round((signals["HR"] < 40).mean() * 100)
            over_150_pct = round((signals["HR"] > 150).mean() * 100)

            # Identify peaks with high amplitude artifacts
            peaks2 = peak_indices
            bvp_vals_at_peaks = df["BVP"].values[peaks2]
            peaks_over_79 = peaks2[bvp_vals_at_peaks > 79]
            peaks_under_79 = peaks2[bvp_vals_at_peaks <= 79]
            peaks_without_over_79 = np.setdiff1d(peaks2, peaks_over_79)

            # Distances between clean peaks (in samples)
            distances_between_peaks = np.diff(peaks_without_over_79)
            if len(distances_between_peaks) == 0:
                raise ValueError("No clean peak intervals found after artifact removal.")

            # find longest low-variance segment of consecutive intervals (sliding-window O(n^2))
            threshold = 3.0
            longest_segment_start = 0
            longest_segment_end = 0
            longest_segment_length = 0
            for i in range(len(distances_between_peaks)):
                for j in range(i, len(distances_between_peaks)):
                    segment = distances_between_peaks[i: j + 1]
                    if np.std(segment) < threshold and (j - i + 1) > longest_segment_length:
                        longest_segment_start = i
                        longest_segment_end = j + 1
                        longest_segment_length = j - i + 1

            # Map segment to original peak indices
            if longest_segment_length == 0:
                raise ValueError("Could not find a stable low-variance segment of peak intervals.")

            # index mapping: segment positions correspond to peaks_without_over_79 indices
            start_index = int(peaks_under_79[longest_segment_start])
            end_index = int(peaks_under_79[longest_segment_end])  # inclusive end

            # Fine-tune start/end using slope/derivative minima
            slope_ppg = np.gradient(np.gradient(signals["PPG_Clean"].values)) - 2.0
            slope_peaks, _ = find_peaks(slope_ppg)
            # defensively pick nearest minima if needed
            if len(slope_peaks) > 0:
                left_candidates = slope_peaks[slope_peaks < start_index]
                if left_candidates.size > 0:
                    local_min = int(left_candidates[-1])
                else:
                    local_min = int(slope_peaks[np.argmin(np.abs(slope_peaks - start_index))])
                right_candidates = slope_peaks[slope_peaks > end_index]
                if right_candidates.size > 0:
                    final_min = int(right_candidates[0])
                else:
                    final_min = int(slope_peaks[np.argmin(np.abs(slope_peaks - end_index))])
                # override start/end with refined minima
                start_index, end_index = local_min, final_min

            new_duration = (end_index - start_index) / 15.0
            proportion_remaining = (new_duration / orig_duration) * 100.0

            # Extract usable segment (slice on original df rows)
            useable_df = df.iloc[start_index: end_index + 1].reset_index(drop=True)
            eliminated = False

            # Apply elimination heuristics
            if under_40_pct > 35 or over_150_pct > 7 or new_duration < 10:
                eliminated = True
                ppg_dict[participant_key]["Eliminated"] = True
                ppg_dict[participant_key]["Reason"] = f"HR issues or too short (under40={under_40_pct}, over150={over_150_pct}, dur={new_duration:.1f}s)"
            else:
                ppg_dict[participant_key].update({
                    "Proportion_Remaining": proportion_remaining,
                    "under_40_pct": under_40_pct,
                    "over_150_pct": over_150_pct,
                    "start_index": int(start_index),
                    "end_index": int(end_index),
                    "Passed": True
                })

        except Exception as e:
            eliminated = True
            ppg_dict[participant_key]["Eliminated"] = True
            ppg_dict[participant_key]["Error"] = repr(e)
            logging.debug(f"check_bvp_quality error for {participant_key}: {e}")
            traceback.print_exc()

        return eliminated, ppg_dict, useable_df, hp_filtered_ppg, peaks_without_over_79, start_index, end_index

    # -------------------------
    # Cleaning helpers
    # -------------------------
    def clean_eda(self, signal_values: np.ndarray, visualization: bool = False) -> pd.DataFrame:
        """
        Clean EDA signal with NeuroKit2.

        Args:
            signal_values: raw EDA numpy array
            visualization: if True, show plots (useful for debugging)

        Returns:
            eda_signals DataFrame with columns such as ['EDA_Raw','EDA_Clean','EDA_Phasic']

        Example:
            eda_df = files.clean_eda(eda_values, visualization=False)
        """
        signals, info = nk.eda_process(signal_values, sampling_rate=15, method_cleaning="biosppy", method_peaks="nabian2018")
        wanted_columns = [col for col in ["EDA_Raw", "EDA_Clean", "EDA_Phasic"] if col in signals.columns]
        eda_signals = signals[wanted_columns].copy()
        if visualization:
            nk.eda_plot(signals, info)
            plt.show()
            plt.close()
        return eda_signals

    def clean_bvp(self, signal_values: np.ndarray, visualization: bool = False) -> pd.DataFrame:
        """
        Clean BVP (PPG) signal using HeartPy + NeuroKit2.

        Args:
            signal_values: raw BVP array
            visualization: show diagnostic plots

        Returns:
            DataFrame containing columns like 'BVP_Raw', 'BVP_Clean'
        """
        hp_filtered_ppg = hp.filter_signal(signal_values, cutoff=[0.8, 2.5], filtertype="bandpass", sample_rate=15, order=4, return_top=False)
        hp_filtered_ppg = hp_filtered_ppg + np.mean(signal_values)

        signals, info = nk.ppg_process(hp_filtered_ppg, sampling_rate=15, method_peaks="bishop", method_cleaning="none", correct_artifacts=True)
        if visualization:
            nk.ppg_plot(signals, info)
            plt.show()
            plt.close()

        wanted_columns = []
        if "PPG_Raw" in signals.columns:
            wanted_columns.append("PPG_Raw")
        if "PPG_Clean" in signals.columns:
            wanted_columns.append("PPG_Clean")

        bvp_signals = signals[wanted_columns].copy().rename(columns={"PPG_Raw": "BVP_Raw", "PPG_Clean": "BVP_Clean"})
        # Attach cleaned array if missing
        if "BVP_Clean" not in bvp_signals.columns:
            bvp_signals["BVP_Clean"] = hp_filtered_ppg
        return bvp_signals

    def clean_tmp(self, signal_values: np.ndarray, visualization: bool = False) -> pd.DataFrame:
        """
        Smooth TMP using a Savitzky-Golay filter.

        Args:
            signal_values: raw TMP array
            visualization: show the raw vs smoothed plot

        Returns:
            DataFrame with column 'TMP_Clean'
        """
        smoothing_duration = 2.0  # seconds
        window_length = int(smoothing_duration * 15)
        if window_length % 2 == 0:
            window_length += 1
        polyorder = 3
        # If signal is shorter than window_length, fall back to minimal smoothing (no op)
        if len(signal_values) < window_length:
            smoothed_tmp = signal_values.copy()
        else:
            smoothed_tmp = savgol_filter(signal_values, window_length, polyorder)
        df_smoothed = pd.DataFrame({"TMP_Clean": smoothed_tmp})
        if visualization:
            plt.plot(signal_values, label="raw")
            plt.plot(smoothed_tmp, label="smoothed")
            plt.legend()
            plt.show()
            plt.close()
        return df_smoothed

    # -------------------------
    # Finger contact / EDA checks
    # -------------------------
    def check_finger(self, participant: str, video_id: str, eda_dict: Dict[str, dict]) -> Tuple[bool, Optional[pd.DataFrame], Dict[str, dict]]:
        """
        Check EDA sensor contact and flatlines.

        Args:
            participant: participant id (without video prefix)
            video_id: 'baseline' or video id like '3.01'
            eda_dict: dict for storing QC info

        Returns:
            eliminated: bool
            df: loaded DataFrame or None
            eda_dict: updated dict

        Heuristics:
            - A threshold is used to detect flat or very low EDA values (value < 0.05).
            - Several conditions (counts, durations) lead to elimination.
        """
        key = f"{video_id}_{participant}"
        eda_dict.setdefault(key, {})
        filename = f"{key}.csv"
        filepath = os.path.join(self.divided, filename)
        eliminated = False
        df = None

        try:
            df = pd.read_csv(filepath)
            if "EDA" not in df.columns:
                raise ValueError("Expected 'EDA' column in EDA file.")

            threshold = 0.05
            # Baseline warmup handling
            if video_id == "baseline" and len(df) > 75:
                df = df.iloc[75:].reset_index(drop=True)
                # find first index above threshold and truncate
                above_idx = df.index[df["EDA"] > threshold].tolist()
                if above_idx:
                    df = df.loc[above_idx[0]:].reset_index(drop=True)

            eda_raw = df["EDA"].values
            below_threshold = eda_raw < threshold

            # Convert boolean runs to start/end indices
            int_array = below_threshold.astype(int)
            diff = np.diff(int_array)
            starts = np.where(diff == 1)[0] + 1
            ends = np.where(diff == -1)[0] + 1
            # handle edge cases (start/end in true state)
            if below_threshold.size and below_threshold[0]:
                starts = np.insert(starts, 0, 0)
            if below_threshold.size and below_threshold[-1]:
                ends = np.append(ends, len(below_threshold))

            # Align starts/ends
            if len(ends) > len(starts):
                ends = ends[1:]
            elif len(starts) > len(ends):
                starts = starts[:-1]

            lengths = ends - starts if len(starts) == len(ends) else np.array([])
            count_below_threshold = int(below_threshold.sum())
            proportion_below_threshold = (count_below_threshold / max(len(df), 1)) * 100
            occurrences = len(lengths)
            duration = len(df) / 15.0

            # Detect flat signals (very large amplitude occurrences are suspicious)
            count_above_30 = int((eda_raw > 30).sum())

            # Elimination rules (tweakable)
            if count_below_threshold >= 11 or occurrences >= 3 or duration < 10 or count_above_30 > 0:
                eliminated = True
                eda_dict[key]["Eliminated"] = "Failed_Check_Finger"
                eda_dict[key]["Proportion_Below_Threshold"] = count_below_threshold
                eda_dict[key]["Occurrences_Below_Threshold"] = occurrences
                eda_dict[key]["Over_30_raw_eda"] = count_above_30
            else:
                eda_dict[key]["Passed"] = True
                eda_dict[key]["Proportion_Below_Threshold"] = count_below_threshold
                eda_dict[key]["Occurrences_Below_Threshold"] = occurrences
                eda_dict[key]["Over_30_raw_eda"] = count_above_30

        except Exception as e:
            eliminated = True
            eda_dict[key]["Eliminated"] = "Check_Finger_Exception"
            eda_dict[key]["Error"] = repr(e)
            logging.debug(f"check_finger error for {key}: {e}")
            traceback.print_exc()
            df = None

        return eliminated, df, eda_dict

    # -------------------------
    # Mark as QC passed and save filtered files
    # -------------------------
    def qc_passed(self, df: pd.DataFrame, signal_type: str, video_type: bool, eda_dict: Dict, ppg_dict: Dict, tmp_dict: Dict, participant_key: str, sensor_dictionary: Dict[str, dict], save: bool = True) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        Standardized actions when a signal passes QC:
          - Process dataframe (normalize, clean)
          - Save processed CSV to QC folder
          - Update sensor_dictionary flags

        Args:
            df: DataFrame with at least columns ['counter','time_values','MRK', signal_type]
            signal_type: 'EDA', 'BVP', or 'TMP'
            video_type: bool True if baseline (special handling), False for video
            eda_dict, ppg_dict, tmp_dict: dicts to update (passed by caller)
            participant_key: e.g. 'baseline_240101_135256' or '3.01_240101_135256'
            sensor_dictionary: mapping from file key to sensor flags
            save: if True, writes a file to self.divided_qc

        Returns:
            eda_dict, ppg_dict, tmp_dict, sensor_dictionary (all possibly updated)
        """
        wanted_columns = ["counter", "time_values", "MRK", signal_type]
        # Defensive: if columns missing, still continue by selecting available columns
        filtered_columns = df[[c for c in wanted_columns if c in df.columns]].copy()

        processed_filtered_columns = self.process_dataframe(filtered_columns, video_type, signal_type, vid_length=593, baseline_length=2008)
        if save:
            out_name = f"{participant_key}_{signal_type.lower()}_qc.csv"
            out_path = os.path.join(self.divided_qc, out_name)
            processed_filtered_columns.to_csv(out_path, index=False)

        # Update sensor dictionary flag
        sensor_dictionary.setdefault(participant_key, {})
        sensor_dictionary[participant_key][signal_type] = True

        return eda_dict, ppg_dict, tmp_dict, sensor_dictionary

    # -------------------------
    # Mark eliminations consistently
    # -------------------------
    def eliminated(self, baseline: bool, elim_type: str, elim_signals: List[str], eda_dict: Dict, ppg_dict: Dict, tmp_dict: Dict, participant_key: str, participant_video_filenames: List[str], sensor_dictionary: Dict[str, dict]) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        Mark the provided signals as eliminated for the given participant (or baseline + all videos).

        Args:
            baseline: True if this elimination pertains to the baseline recording (propagate to videos)
            elim_type: string describing elimination reason
            elim_signals: list of signals to mark ['EDA','BVP','TMP']
            eda_dict, ppg_dict, tmp_dict: dicts to update
            participant_key: base key (e.g. 'baseline_240101_135256' or '3.01_240101_135256')
            participant_video_filenames: list of filenames for this participant (with .csv)
            sensor_dictionary: mapping to update
        """
        # Helper to set false and propagate if baseline
        def set_false(dic, key, participant_video_filenames, sig):
            dic.setdefault(key, {})
            dic[key]["Eliminated"] = elim_type
            sensor_dictionary.setdefault(key, {})
            sensor_dictionary[key][sig] = False
            if baseline:
                for filename in participant_video_filenames:
                    key_noext = filename[:-4]
                    dic.setdefault(key_noext, {})
                    dic[key_noext]["Eliminated"] = elim_type
                    sensor_dictionary.setdefault(key_noext, {})
                    sensor_dictionary[key_noext][sig] = False

        if "EDA" in elim_signals:
            set_false(eda_dict, participant_key, participant_video_filenames, "EDA")
        if "BVP" in elim_signals:
            set_false(ppg_dict, participant_key, participant_video_filenames, "BVP")
        if "TMP" in elim_signals:
            set_false(tmp_dict, participant_key, participant_video_filenames, "TMP")

        return eda_dict, ppg_dict, tmp_dict, sensor_dictionary

    # -------------------------
    # Main QC for one participant
    # -------------------------
    def quality_assessments(self, participant_key: str) -> Tuple[Dict[str, dict], Dict[str, dict], Dict[str, dict], Dict[str, dict]]:
        """
        Perform QC for a single participant across baseline and video files.

        Args:
            participant_key: matches the 'Keys' column in Participant_videos.csv

        Returns:
            sensor_dict, ppg_dict, eda_dict, tmp_dict
            Each is a dict mapping file_key -> metrics / flags
        """
        logging.info(f"Starting QC for participant: {participant_key}")
        warnings.filterwarnings("ignore")

        sensor_dict: Dict[str, dict] = {}
        ppg_dict: Dict[str, dict] = {}
        eda_dict: Dict[str, dict] = {}
        tmp_dict: Dict[str, dict] = {}

        status_path = os.path.join(self.demographics, "Participant_videos.csv")
        if not os.path.isfile(status_path):
            logging.error(f"Participant_videos.csv missing at {status_path}")
            return sensor_dict, ppg_dict, eda_dict, tmp_dict

        status_df = pd.read_csv(status_path, header=0)
        try:
            status_df_index = status_df.index[status_df["Keys"] == participant_key].item()
        except Exception as e:
            logging.error(f"Participant {participant_key} not found in status_df: {e}")
            return sensor_dict, ppg_dict, eda_dict, tmp_dict

        # Collect the 4 video IDs for this participant (V1..V4)
        v_list = ["V1", "V2", "V3", "V4"]
        participant_video_filenames: List[str] = []
        for v in v_list:
            video_id = status_df.at[status_df_index, v]
            # skip NaN entries (no assigned video in that slot)
            if not pd.isna(video_id):
                filename = f"{video_id}_{participant_key}.csv"
                key_name = filename[:-4]
                sensor_dict.setdefault(key_name, {})
                ppg_dict.setdefault(key_name, {})
                eda_dict.setdefault(key_name, {})
                tmp_dict.setdefault(key_name, {})
                participant_video_filenames.append(filename)

        # baseline
        baseline_key = f"baseline_{participant_key}"
        sensor_dict.setdefault(baseline_key, {})
        ppg_dict.setdefault(baseline_key, {})
        eda_dict.setdefault(baseline_key, {})
        tmp_dict.setdefault(baseline_key, {})

        signals = ["EDA", "BVP", "TMP"]

        for signal_type in signals:
            wanted_columns = ["counter", "time_values", "MRK", signal_type]

            # ---- EDA QC: baseline then each video ----
            if signal_type == "EDA":
                try:
                    eliminated, df, eda_dict = self.check_finger(participant_key, "baseline", eda_dict)
                    if eliminated:
                        eda_dict, ppg_dict, tmp_dict, sensor_dict = self.eliminated(True, "Failed_EDA_Check_Finger_BL", ["EDA", "BVP", "TMP"], eda_dict, ppg_dict, tmp_dict, baseline_key, participant_video_filenames, sensor_dict)
                        # If baseline failed, skip the rest of EDA/BVP/TMP for this participant
                        continue
                    else:
                        eda_dict, ppg_dict, tmp_dict, sensor_dict = self.qc_passed(df, "EDA", True, eda_dict, ppg_dict, tmp_dict, baseline_key, sensor_dict)

                    # videos after baseline
                    for video_file in participant_video_filenames:
                        vid_id = video_file.split("_")[0]
                        participant_eda_key = f"{vid_id}_{participant_key}"
                        eliminated, df, eda_dict = self.check_finger(participant_key, vid_id, eda_dict)
                        if eliminated:
                            eda_dict, ppg_dict, tmp_dict, sensor_dict = self.eliminated(False, "Failed_EDA_Check_Finger", ["EDA", "BVP", "TMP"], eda_dict, ppg_dict, tmp_dict, participant_eda_key, participant_video_filenames, sensor_dict)
                            continue
                        else:
                            eda_dict, ppg_dict, tmp_dict, sensor_dict = self.qc_passed(df, "EDA", False, eda_dict, ppg_dict, tmp_dict, participant_eda_key, sensor_dict)

                except Exception as e:
                    logging.error(f"Participant: {participant_key}, Error in EDA QC: {e}")
                    traceback.print_exc()

            # ---- BVP QC ----
            if signal_type == "BVP" and sensor_dict.get(baseline_key, {}).get("BVP") != False:
                try:
                    eliminated, ppg_dict, useable_df, hp_filtered_ppg, peaks_without_over_79, start_index, end_index = self.check_bvp_quality(participant_key, "baseline", ppg_dict)
                    if eliminated:
                        sensor_dict[baseline_key]["BVP"] = False
                        for baseline_problem in participant_video_filenames:
                            ppg_dict.setdefault(baseline_problem[:-4], {})["Eliminated"] = "Baseline_BVP_eliminated"
                            sensor_dict.setdefault(baseline_problem[:-4], {})["BVP"] = False
                    else:
                        sensor_dict[baseline_key]["BVP"] = True
                        if useable_df is not None:
                            filtered_columns = useable_df[[c for c in wanted_columns if c in useable_df.columns]]
                            processed_filtered_columns = self.process_dataframe(filtered_columns, True, "BVP", vid_length=593, baseline_length=2008)
                            processed_filtered_columns.to_csv(os.path.join(self.divided_qc, f"{baseline_key}_bvp_qc.csv"), index=False)

                    # Per-video
                    for video_file in participant_video_filenames:
                        vid_id = video_file.split("_")[0]
                        eliminated, ppg_dict, useable_df, hp_filtered_ppg, peaks_without_over_79, start_index, end_index = self.check_bvp_quality(participant_key, vid_id, ppg_dict)
                        if eliminated:
                            sensor_dict.setdefault(video_file[:-4], {})["BVP"] = False
                            ppg_dict.setdefault(video_file[:-4], {})["Eliminated"] = "Check_BVP_Quality_eliminated"
                            continue
                        else:
                            sensor_dict.setdefault(video_file[:-4], {})["BVP"] = True
                            if useable_df is not None:
                                filtered_columns = useable_df[[c for c in wanted_columns if c in useable_df.columns]]
                                processed_filtered_columns = self.process_dataframe(filtered_columns, False, "BVP", vid_length=593, baseline_length=2008)
                                processed_filtered_columns.to_csv(os.path.join(self.divided_qc, f"{video_file[:-4]}_bvp_qc.csv"), index=False)

                except Exception as e:
                    logging.error(f"Participant: {participant_key}, Error in BVP QC: {e}")
                    traceback.print_exc()

            # ---- TMP QC ----
            if signal_type == "TMP" and sensor_dict.get(baseline_key, {}).get("TMP") != False:
                try:
                    eliminated, useable_df, tmp_dict = self.check_temp_quality(participant_key, "baseline", tmp_dict)
                    if eliminated:
                        sensor_dict[baseline_key]["TMP"] = False
                        for baseline_problem in participant_video_filenames:
                            tmp_dict.setdefault(baseline_problem[:-4], {})["Eliminated"] = "Baseline_TMP_eliminated"
                            sensor_dict.setdefault(baseline_problem[:-4], {})["TMP"] = False
                    else:
                        sensor_dict[baseline_key]["TMP"] = True
                        if useable_df is not None:
                            filtered_columns = useable_df[[c for c in wanted_columns if c in useable_df.columns]]
                            processed_filtered_columns = self.process_dataframe(filtered_columns, True, "TMP", vid_length=593, baseline_length=2008)
                            processed_filtered_columns.to_csv(os.path.join(self.divided_qc, f"{baseline_key}_tmp_qc.csv"), index=False)

                    for video_file in participant_video_filenames:
                        vid_id = video_file.split("_")[0]
                        eliminated, useable_df, tmp_dict = self.check_temp_quality(participant_key, vid_id, tmp_dict)
                        if eliminated:
                            sensor_dict.setdefault(video_file[:-4], {})["TMP"] = False
                            tmp_dict.setdefault(video_file[:-4], {})["Eliminated"] = "Baseline_TMP_eliminated"
                            continue
                        else:
                            sensor_dict.setdefault(video_file[:-4], {})["TMP"] = True
                            if useable_df is not None:
                                filtered_columns = useable_df[[c for c in wanted_columns if c in useable_df.columns]]
                                processed_filtered_columns = self.process_dataframe(filtered_columns, False, "TMP", vid_length=593, baseline_length=2008)
                                processed_filtered_columns.to_csv(os.path.join(self.divided_qc, f"{video_file[:-4]}_tmp_qc.csv"), index=False)

                except Exception as e:
                    logging.error(f"Participant: {participant_key}, Error in TMP QC: {e}")
                    traceback.print_exc()

        logging.info(f"Finished QC for participant: {participant_key}")
        return sensor_dict, ppg_dict, eda_dict, tmp_dict

    # -------------------------
    # Processing pipeline for a single signal DataFrame
    # -------------------------
    def process_dataframe(self, df: pd.DataFrame, video_type: bool, signal: str, vid_length: int, baseline_length: int) -> pd.DataFrame:
        """
        Normalize, clean, and return a processed DataFrame for a single signal type.

        Args:
            df: DataFrame containing at least the column named `signal` (case-sensitive)
            video_type: True for baseline handling, False for video segments
            signal: 'EDA', 'BVP', or 'TMP'
            vid_length, baseline_length: ints (currently accepted but not used to crop)

        Returns:
            DataFrame with added columns:
              - 'normalized' (z-scored original)
              - 'cleaned' (cleaned signal)
              - 'cleaned_normalized' (z-scored cleaned)
              - for EDA: 'phasic' and 'phasic_normalized' if available

        Notes:
            - This function dynamically calls methods named clean_{signal_lower}.
              e.g., signal='EDA' -> method clean_eda is used.
            - If std == 0, fallback subtracts the mean to avoid division-by-zero.
        """
        df = df.reset_index(drop=True)
        original_length = len(df)

        signal_lower = signal.lower()
        cleaning_method_name = f"clean_{signal_lower}"
        cleaning_method = getattr(self, cleaning_method_name, None)
        if cleaning_method is None:
            raise AttributeError(f"No cleaning method found for signal '{signal}'. Expected method name: {cleaning_method_name}")

        # Defensive: if signal column missing, return df unchanged with warning
        if signal not in df.columns:
            logging.warning(f"Signal column '{signal}' not found in df. Returning original DataFrame.")
            return df

        # Normalize original signal
        mean_signal = df[signal].mean()
        std_signal = df[signal].std()
        if std_signal == 0 or np.isnan(std_signal):
            df["normalized"] = df[signal] - mean_signal
        else:
            df["normalized"] = (df[signal] - mean_signal) / std_signal

        # Clean signal (cleaning method expects numpy array)
        cleaned_df = cleaning_method(df[signal].values)
        cleaned_col_name = f"{signal}_Clean"
        if cleaned_col_name in cleaned_df.columns:
            df["cleaned"] = cleaned_df[cleaned_col_name].values[: len(df)]
        else:
            # Fallback: if cleaning method returned a single-column DataFrame, use that
            df["cleaned"] = cleaned_df.iloc[:, 0].values[: len(df)]

        # Normalize cleaned signal
        mean_clean = df["cleaned"].mean()
        std_clean = df["cleaned"].std()
        if std_clean == 0 or np.isnan(std_clean):
            df["cleaned_normalized"] = df["cleaned"] - mean_clean
        else:
            df["cleaned_normalized"] = (df["cleaned"] - mean_clean) / std_clean

        if signal == "EDA":
            phasic_col = f"{signal}_Phasic"
            if phasic_col in cleaned_df.columns:
                df["phasic"] = cleaned_df[phasic_col].values[: len(df)]
                mean_ph = df["phasic"].mean()
                std_ph = df["phasic"].std()
                if std_ph == 0 or np.isnan(std_ph):
                    df["phasic_normalized"] = df["phasic"] - mean_ph
                else:
                    df["phasic_normalized"] = (df["phasic"] - mean_ph) / std_ph

        # Reset and return
        df = df.reset_index(drop=True)
        return df
