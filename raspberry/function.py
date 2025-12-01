import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
from datetime import datetime
import zipfile
import io
import paho.mqtt.client as mqtt
import base64


class KalmanFilterAccelerometer:
    """
    Kalman filter implementation for processing accelerometer data.
    """
    
    def __init__(self, process_noise=0.01, measurement_noise=0.1, dimension=3):
        self.dimension = dimension
        self.state_dim = 2 * dimension
        
        # State transition matrix (F)
        self.F = np.eye(self.state_dim)
        for i in range(dimension):
            self.F[i, i + dimension] = 1  # Position is updated by velocity
            
        # Measurement matrix (H)
        self.H = np.zeros((dimension, self.state_dim))
        for i in range(dimension):
            self.H[i, i] = 1
            
        # Process noise covariance (Q)
        self.Q = np.eye(self.state_dim) * process_noise
        
        # Measurement noise covariance (R)
        self.R = np.eye(dimension) * measurement_noise
        
        # Initial state
        self.x = np.zeros(self.state_dim)
        
        # Initial covariance estimate (P)
        self.P = np.eye(self.state_dim)
        
    def predict(self):
        # Project the state ahead
        self.x = self.F @ self.x
        
        # Project the error covariance ahead
        self.P = self.F @ self.P @ self.F.T + self.Q
        
    def update(self, measurement):
        # Ensure measurement is the right shape
        measurement = np.array(measurement).reshape(self.dimension, 1)
        
        # Compute the Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update the estimate with measurement
        y = measurement - self.H @ self.x.reshape(self.state_dim, 1)
        self.x = self.x + (K @ y).flatten()
        
        # Update the error covariance
        I = np.eye(self.state_dim)
        self.P = (I - K @ self.H) @ self.P
        
        return self.x[:self.dimension]  # Return position estimate
        
    def filter_data(self, accelerometer_data):
        filtered_data = np.zeros_like(accelerometer_data)
        
        for i, measurement in enumerate(accelerometer_data):
            self.predict()
            filtered_pos = self.update(measurement)
            filtered_data[i] = filtered_pos
            
        return filtered_data


def count_reps(filtered_data, axis=1, height=0.05, distance=20):
    """
    Count repetitions in filtered accelerometer data.
    
    Parameters:
    -----------
    filtered_data : array_like
        Filtered accelerometer data.
    axis : int
        Axis to analyze for repetitions (0=x, 1=y, 2=z).
    height : float
        Minimum height of peaks to consider as reps.
    distance : int
        Minimum number of samples between peaks.
        
    Returns:
    --------
    int
        Number of repetitions detected.
    array_like
        Indices of the repetition peaks.
    """
    # Extract data for the specific axis
    axis_data = filtered_data[:, axis]
    
    # Find peaks in the data
    peaks, _ = find_peaks(axis_data, height=height, distance=distance)
    
    return len(peaks), peaks


def calculate_rep_durations(filtered_data, peaks, timestamps):
    """
    Calculate the duration of each repetition.
    
    Parameters:
    -----------
    filtered_data : array_like
        Filtered accelerometer data.
    peaks : array_like
        Indices of repetition peaks.
    timestamps : array_like
        Timestamps corresponding to each data point.
        
    Returns:
    --------
    array_like
        Duration of each repetition in seconds.
    """
    if len(peaks) < 2:
        return np.array([])
    
    # Calculate time between peaks (rep duration)
    durations = []
    for i in range(len(peaks) - 1):
        start_idx = peaks[i]
        end_idx = peaks[i+1]
        
        # Calculate duration in seconds
        duration = timestamps[end_idx] - timestamps[start_idx]
        
        # Apply scaling if necessary (if durations are unrealistically short)
        if duration < 0.1:  # Less than 0.1 seconds seems too fast for a rep
            scaling_factor = 3.0  # Adjust to get realistic times
            duration *= scaling_factor
            
        durations.append(duration)
    
    # For the last rep, use the same duration as the previous rep
    if len(durations) > 0:
        durations.append(durations[-1])
    
    return np.array(durations)


def detect_rest_periods(filtered_data, axis=1, threshold=0.2, min_rest_samples=50):
    """
    Detect rest periods between sets.
    
    Parameters:
    -----------
    filtered_data : array_like
        Filtered accelerometer data.
    axis : int
        Axis to analyze for rest periods (0=x, 1=y, 2=z).
    threshold : float
        Maximum acceleration magnitude to consider as rest.
    min_rest_samples : int
        Minimum number of consecutive samples below threshold to count as rest.
        
    Returns:
    --------
    list
        List of tuples (start_idx, end_idx) for each rest period.
    """
    # Calculate magnitude of acceleration for the given axis
    activity = np.abs(filtered_data[:, axis])
    
    # Find segments below threshold
    is_resting = activity < threshold
    
    rest_periods = []
    rest_start = None
    
    # Detect continuous rest periods
    for i, resting in enumerate(is_resting):
        if resting and rest_start is None:
            rest_start = i
        elif not resting and rest_start is not None:
            if i - rest_start >= min_rest_samples:
                rest_periods.append((rest_start, i))
            rest_start = None
    
    # Handle case where rest continues until the end
    if rest_start is not None and len(is_resting) - rest_start >= min_rest_samples:
        rest_periods.append((rest_start, len(is_resting)))
    
    return rest_periods


def calculate_range_of_motion(filtered_data, rep_peaks, primary_axis, window_size=5):
    """
    Calculate the range of motion for each repetition.
    
    Parameters:
    -----------
    filtered_data : array_like
        Filtered accelerometer data.
    rep_peaks : array_like
        Indices of repetition peaks.
    primary_axis : int
        Primary axis of movement (0=X, 1=Y, 2=Z).
    window_size : int
        Number of data points before and after the peak to consider for ROM calculation.
        
    Returns:
    --------
    dict
        Dictionary containing ROM metrics for each repetition.
    """
    if len(rep_peaks) == 0:
        return {
            'rom_values': np.array([]),
            'rom_percentages': np.array([]),
            'max_rom_idx': 0,
            'max_rom_value': 0
        }
    
    # Calculate ROM for each repetition
    rom_values = []
    
    for i, peak_idx in enumerate(rep_peaks):
        # Define window around peak
        start_idx = max(0, peak_idx - window_size)
        end_idx = min(len(filtered_data) - 1, peak_idx + window_size)
        
        # Extract data within window
        window_data = filtered_data[start_idx:end_idx+1, primary_axis]
        
        # Calculate min and max within the window
        if len(window_data) > 0:
            min_val = np.min(window_data)
            max_val = np.max(window_data)
            
            # Range of motion is the difference between max and min
            rom = max_val - min_val
            rom_values.append(rom)
        else:
            rom_values.append(0)
    
    rom_values = np.array(rom_values)
    
    # Find the maximum ROM and its index
    if len(rom_values) > 0:
        max_rom_value = np.max(rom_values)
        max_rom_idx = np.argmax(rom_values)
        
        # Calculate ROM percentages relative to the maximum
        if max_rom_value > 0:
            rom_percentages = (rom_values / max_rom_value) * 100
        else:
            rom_percentages = np.zeros_like(rom_values)
    else:
        max_rom_value = 0
        max_rom_idx = 0
        rom_percentages = np.array([])
    
    return {
        'rom_values': rom_values,
        'rom_percentages': rom_percentages,
        'max_rom_idx': max_rom_idx,
        'max_rom_value': max_rom_value
    }

def save_workout_csv(timestamps_ms, raw_history, filename=None):
    """
    Sauvegarde un CSV au format :
      timestamp_ms ; accX_g ; accY_g ; accZ_g
      
    Paramètres :
    -----------
    timestamps_ms : list
        Liste des timestamps en millisecondes.
    raw_history : list
        Liste des données brutes : [ [ax, ay, az], ... ]
    filename : str
        Nom du fichier CSV (optionnel). Si None → généré automatiquement.

    Retour :
    --------
    str : Nom du fichier créé.
    """

    if filename is None:
        # Nom automatique : workout_2025-02-03_14h32.csv
        now = datetime.now().strftime("%Y-%m-%d_%Hh%M")
        filename = f"workout_{now}.csv"

    df = pd.DataFrame({
        "timestamp_ms": timestamps_ms,
        "accX_g": [row[0] for row in raw_history],
        "accY_g": [row[1] for row in raw_history],
        "accZ_g": [row[2] for row in raw_history],
    })

    df.to_csv(filename, sep=';', index=False)
    print(f"CSV créé : {filename}")

    return filename

def analyze_workout_from_csv(csv_filepath, process_noise=0.003, measurement_noise=0.1):
    """
    Analyze workout data from CSV file using only accelerometer data and timestamps.
    All other columns are dropped.
    
    Parameters:
    -----------
    csv_filepath : str
        Path to the CSV file containing workout data.
    process_noise : float
        Process noise parameter for Kalman filter.
    measurement_noise : float
        Measurement noise parameter for Kalman filter.
        
    Returns:
    --------
    dict
        Dictionary containing workout metrics and the loaded DataFrame.
    """
    # Load data from CSV
    df = pd.read_csv(csv_filepath, delimiter=';')
    
    # IMPORTANT: Keep only timestamp and accelerometer data, drop everything else
    required_columns = ['timestamp_ms', 'accX_g', 'accY_g', 'accZ_g']
    
    # Make sure all required columns exist
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in CSV file")
    
    # Create simplified dataframe with only accelerometer data and timestamps
    simplified_df = df[required_columns].copy()
    
    # Extract accelerometer data
    accel_data = simplified_df[['accX_g', 'accY_g', 'accZ_g']].values
    
    # Extract timestamp data and convert to seconds
    #timestamps = simplified_df['timestamp_ms'].values / 1000  # Convert milliseconds to seconds (i dont the unit was actually ms)
    timestamps = simplified_df['timestamp_ms'].values / 100  # I think this conversion makes more sense in a workout sense
    # Calculate sampling rate from timestamps
    if len(timestamps) > 1:
        time_diffs = np.diff(timestamps)
        median_diff = np.median(time_diffs[time_diffs > 0])  # Ignore zero differences
        sampling_rate = 1 / median_diff if median_diff > 0 else 50  # Default to 50Hz if can't determine
    else:
        sampling_rate = 50  # Default sampling rate
    
    # Apply Kalman filter to the accelerometer data
    kf = KalmanFilterAccelerometer(process_noise=process_noise, measurement_noise=measurement_noise)
    filtered_data = kf.filter_data(accel_data)
    
    # Determine the most active axis based on variance
    axis_variance = np.var(filtered_data, axis=0)
    primary_axis = np.argmax(axis_variance)
    
    # Count repetitions - use adaptive parameters
    axis_data = filtered_data[:, primary_axis]
    signal_mean = np.mean(axis_data)
    signal_std = np.std(axis_data)
    
    # Adjust height based on signal characteristics
    height = signal_mean + 0.35 * signal_std
    
    # Reasonable default for min_distance
    min_distance = int(len(simplified_df) / 15)  # Estimate assuming about 15 reps total
    min_distance = max(1, min_distance)  # Ensure it's at least 1
    
    # Count repetitions
    rep_count, rep_peaks = count_reps(filtered_data, axis=primary_axis, 
                                     height=height, distance=min_distance)
    
    # Calculate range of motion for each repetition
    rom_results = calculate_range_of_motion(filtered_data, rep_peaks, primary_axis)
    
    # Use rest period detection to identify sets
    # Adjust threshold based on signal characteristics
    threshold = signal_mean * 0.4  # Use 40% of mean as threshold for rest
    min_rest_samples = int(sampling_rate * 1)  # At least 1 seconds of rest
    
    rest_periods = detect_rest_periods(filtered_data, axis=primary_axis, 
                                      threshold=threshold, 
                                      min_rest_samples=min_rest_samples)
    
    # Use rest periods to define sets
    if len(rest_periods) > 0:
        set_starts = [0] + [end for _, end in rest_periods]
        set_ends = [start for start, _ in rest_periods] + [len(filtered_data)]
        sets = list(zip(set_starts, set_ends))
    else:
        # No rest periods detected, assume one set
        sets = [(0, len(filtered_data))]
    
    # Calculate rest durations
    rest_durations = []
    for start, end in rest_periods:
        duration = (timestamps[end] - timestamps[start])
        rest_durations.append(duration)
    
    # Calculate rep durations
    rep_durations = calculate_rep_durations(filtered_data, rep_peaks, timestamps)
    
    # Prepare results
    results = {
        'df': simplified_df,
        'filtered_data': filtered_data,
        'primary_axis': primary_axis,
        'detected_rep_count': rep_count,
        'rep_peaks': rep_peaks,
        'rep_durations': rep_durations,  # Now storing durations instead of speeds
        'rest_periods': rest_periods,
        'rest_durations': np.array(rest_durations),
        'timestamps': timestamps,
        'sets': sets,
        'sampling_rate': sampling_rate,
        'unique_reps_per_set': rep_count // max(1, len(sets)),
        'rom_values': rom_results['rom_values'],
        'rom_percentages': rom_results['rom_percentages'],
        'max_rom_idx': rom_results['max_rom_idx'],
        'max_rom_value': rom_results['max_rom_value']
    }
    
    return results

def dict_to_zip(results,zip_path="results.zip"):
    """
    Create a zip of a results dict
    
    Parameters:
    -----------
    results : dict
        Results from analyze_workout_from_csv function containing workout metrics.
        Expected keys include:
        - df: DataFrame containing raw data
        - filtered_data: Kalman-filtered accelerometer data
        - primary_axis: Index of primary movement axis
        - rep_peaks: Indices of repetition peaks
        - rest_periods: List of (start, end) tuples for rest periods
        - timestamps: Array of measurement timestamps
        - sets: List of (start, end) tuples for each set
        - rom_percentages: Array of ROM percentages for each rep
        - max_rom_idx: Index of rep with maximum ROM
        - rep_durations: Array of durations for each rep
        - rest_durations: Array of durations for each rest period
    """
    # Unpack the results
    df = results['df']
    filtered_data = results['filtered_data']
    primary_axis = results['primary_axis']
    rep_peaks = results['rep_peaks']
    rest_periods = results['rest_periods']
    timestamps = results['timestamps']
    sets = results['sets']
    rom_percentages = results['rom_percentages']
    max_rom_idx = results['max_rom_idx']
    rep_durations = results['rep_durations']

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:

        # 1. Raw dataframe
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        z.writestr("raw_data.csv", csv_buffer.getvalue())

        # 2. Filtered data (CSV)
        filtered_df = pd.DataFrame(filtered_data)
        csv_buffer = io.StringIO()
        filtered_df.to_csv(csv_buffer, index=False)
        z.writestr("filtered_data.csv", csv_buffer.getvalue())

        # 3. Simple TXT values
        z.writestr("primary_axis.txt", str(primary_axis))
        z.writestr("max_rom_idx.txt", str(max_rom_idx))

        # 4. List/array exports as CSV
        def write_csv(name, data):
            csv_buffer = io.StringIO()
            pd.DataFrame(data).to_csv(csv_buffer, index=False, header=False)
            z.writestr(name, csv_buffer.getvalue())

        write_csv("rep_peaks.csv", rep_peaks)
        write_csv("rest_periods.csv", rest_periods)
        write_csv("timestamps.csv", timestamps)
        write_csv("sets.csv", sets)
        write_csv("rom_percentages.csv", rom_percentages)
        write_csv("rep_durations.csv", rep_durations)
        #write_csv("rest_durations.csv", rest_durations)

    return zip_path
    
def send_mqtt(zip_path, broker = "20.251.170.166", port = 1883,TOPIC = "sensors/zip"):
    # Lire le ZIP
    with open(zip_path, "rb") as f:
        zip_bytes = f.read()

    # Encoder en base64
    zip_b64 = base64.b64encode(zip_bytes).decode()  # Converti en str UTF-8

    # Créer le client MQTT
    client = mqtt.Client()
    client.connect(broker, port, 60)
    client.loop_start()

    # Publier le message
    client.publish(topic, zip_b64)
    print(f"[MQTT] ZIP envoyé sur le topic '{topic}'")

    client.loop_stop()
    client.disconnect()
        


def visualize_workout_analysis_from_csv(results):
    """
    Visualize the workout analysis results using only accelerometer data.
    
    Parameters:
    -----------
    results : dict
        Results from analyze_workout_from_csv function containing workout metrics.
        Expected keys include:
        - df: DataFrame containing raw data
        - filtered_data: Kalman-filtered accelerometer data
        - primary_axis: Index of primary movement axis
        - rep_peaks: Indices of repetition peaks
        - rest_periods: List of (start, end) tuples for rest periods
        - timestamps: Array of measurement timestamps
        - sets: List of (start, end) tuples for each set
        - rom_percentages: Array of ROM percentages for each rep
        - max_rom_idx: Index of rep with maximum ROM
        - rep_durations: Array of durations for each rep
        - rest_durations: Array of durations for each rest period
    """
    # Unpack the results
    df = results['df']
    filtered_data = results['filtered_data']
    primary_axis = results['primary_axis']
    rep_peaks = results['rep_peaks']
    rest_periods = results['rest_periods']
    timestamps = results['timestamps']
    sets = results['sets']
    rom_percentages = results['rom_percentages']
    max_rom_idx = results['max_rom_idx']
    rep_durations = results['rep_durations']
    
    # Determine how many plots to show based on available data
    num_plots = 3  # Default: Accelerometer data, ROM, and Rep Duration
    if len(results['rest_durations']) > 0:
        num_plots = 4  # Add Rest Duration plot if rest periods were detected
    
    # Create figure
    plt.figure(figsize=(15, num_plots * 3))
    
    # Plot raw and filtered accelerometer data
    axis_labels = ['X', 'Y', 'Z']
    plt.subplot(num_plots, 1, 1)
    
    # Raw accelerometer data for all axes
    raw_data = df[['accX_g', 'accY_g', 'accZ_g']].values
    plt.plot(timestamps, raw_data[:, 0], 'r-', alpha=0.3, label=f'Raw X-axis')
    plt.plot(timestamps, raw_data[:, 1], 'g-', alpha=0.3, label=f'Raw Y-axis')
    plt.plot(timestamps, raw_data[:, 2], 'b-', alpha=0.3, label=f'Raw Z-axis')
    
    # Filtered data for primary axis
    plt.plot(timestamps, filtered_data[:, primary_axis], 'k-', 
             label=f'Filtered {axis_labels[primary_axis]}-axis', linewidth=2)
    
    # Mark repetitions
    plt.plot(timestamps[rep_peaks], filtered_data[rep_peaks, primary_axis], 'ro', 
             label=f'Detected Reps ({len(rep_peaks)})', markersize=8)
    
    # Mark the rep with maximum range of motion with a star
    if len(rep_peaks) > 0 and max_rom_idx < len(rep_peaks):
        max_rom_peak = rep_peaks[max_rom_idx]
        plt.plot(timestamps[max_rom_peak], filtered_data[max_rom_peak, primary_axis], 'y*', 
                markersize=15, label='Max ROM Rep')
    
    # Highlight sets
    for i, (start, end) in enumerate(sets):
        color = f'C{i}'
        # Make sure indices are within bounds
        end_idx = min(end, len(timestamps) - 1)  # Ensure we don't go out of bounds
        plt.axvspan(timestamps[start], timestamps[end_idx], color=color, alpha=0.2, 
                   label=f'Set {i+1}')
    
    # Highlight rest periods
    for start, end in rest_periods:
        # Make sure indices are within bounds
        end_idx = min(end, len(timestamps) - 1)  # Ensure we don't go out of bounds
        plt.axvspan(timestamps[start], timestamps[end_idx], color='gray', alpha=0.3, 
                   label='Rest' if start == rest_periods[0][0] else None)
    
    plt.title('Workout Analysis from Raw Accelerometer Data')
    plt.ylabel('Acceleration (g)')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    # Map repetitions to sets for consistent coloring
    rep_to_set_mapping = []
    
    if len(rep_peaks) > 0:
        for rep_idx, peak_idx in enumerate(rep_peaks):
            # Find which set this rep belongs to
            for set_idx, (start, end) in enumerate(sets):
                if start <= peak_idx < end:
                    rep_to_set_mapping.append(set_idx)
                    break
            else:
                # If no set contains this rep (shouldn't happen), assign to last set
                rep_to_set_mapping.append(len(sets) - 1)
    
    # Plot range of motion percentages
    plt.subplot(num_plots, 1, 2)
    
    if len(rom_percentages) > 0:
        rep_indices = np.arange(1, len(rom_percentages)+1)
        
        # Use the rep_to_set_mapping to determine bar colors
        bar_colors = [f'C{set_idx}' for set_idx in rep_to_set_mapping]
        
        # Create bars for ROM percentages
        rom_bars = plt.bar(rep_indices, rom_percentages, color=bar_colors)
        
        # Highlight the bar with maximum ROM
        if max_rom_idx < len(rom_bars):
            rom_bars[max_rom_idx].set_color('yellow')
            rom_bars[max_rom_idx].set_edgecolor('black')
            rom_bars[max_rom_idx].set_linewidth(2)
        
        # Add percentage values on top of each bar
        for i, percentage in enumerate(rom_percentages):
            plt.text(i+1, percentage + 2, f'{percentage:.1f}%', 
                    horizontalalignment='center', fontsize=9)
        
        plt.title('Range of Motion (% of Maximum)')
        plt.xlabel('Repetition Number')
        plt.ylabel('ROM Percentage (%)')
        plt.ylim(0, 110)  # Set y-axis to go slightly above 100%
        plt.grid(True)
    
    # Plot rep durations
    plt.subplot(num_plots, 1, 3)
    
    if len(rep_durations) > 0:
        rep_indices = np.arange(1, len(rep_durations)+1)
        
        # Use the rep_to_set_mapping to determine bar colors
        bar_colors = [f'C{set_idx}' for set_idx in rep_to_set_mapping]
        
        plt.bar(rep_indices, rep_durations, color=bar_colors)
        
        # Add duration values on top of each bar
        for i, duration in enumerate(rep_durations):
            plt.text(i+1, duration + 0.01, f'{duration:.2f}s', 
                    horizontalalignment='center', fontsize=9)
        
        plt.title('Repetition Duration')
        plt.xlabel('Repetition Number')
        plt.ylabel('Duration (seconds)')
        plt.grid(True)
    
    # Plot rest durations if available
    if len(results['rest_durations']) > 0:
        plt.subplot(num_plots, 1, 4)
        
        rest_indices = np.arange(1, len(results['rest_durations'])+1)
        plt.bar(rest_indices, results['rest_durations'])
        
        # Add rest duration values on top of each bar
        for i, duration in enumerate(results['rest_durations']):
            plt.text(i+1, duration + 0.1, f'{duration:.1f}s', 
                    horizontalalignment='center')
        
        plt.title('Rest Durations Between Sets')
        plt.xlabel('Rest Period Number')
        plt.ylabel('Duration (seconds)')
        plt.grid(True)
    else:
        # If there are no rest durations, add a note about it
        if num_plots == 4:  # We originally planned for 4 plots
            plt.subplot(num_plots, 1, 4)
            plt.text(0.5, 0.5, 'No rest periods detected - continuous workout', 
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=14, transform=plt.gca().transAxes)
            plt.axis('off')  # Hide the axes
    
    plt.tight_layout()
    plt.savefig("workout_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    # Print a summary of the workout
    print(f"Workout Summary:")
    print(f"------------------------------------------")
    print(f"Number of sets: {len(sets)}")
    print(f"Detected repetitions: {results['detected_rep_count']}")
    
    # Print reps per set
    if len(sets) > 0 and len(rep_to_set_mapping) > 0:
        reps_per_set = {}
        for set_idx in rep_to_set_mapping:
            if set_idx not in reps_per_set:
                reps_per_set[set_idx] = 0
            reps_per_set[set_idx] += 1
        
        print("Repetitions per set:")
        for set_idx, rep_count in reps_per_set.items():
            print(f"  - Set {set_idx + 1}: {rep_count} reps")
    
    if len(rep_durations) > 0:
        print(f"Average rep duration: {np.mean(rep_durations):.2f} seconds")
        print(f"Total workout time: {np.sum(rep_durations):.2f} seconds")
    
    if len(results['rest_durations']) > 0:
        print(f"Average rest duration: {np.mean(results['rest_durations']):.2f} seconds")
        print(f"Total rest time: {np.sum(results['rest_durations']):.2f} seconds")
    else:
        print(f"No rest periods detected - continuous workout")
    
    if len(rom_percentages) > 0:
        print(f"Range of Motion Analysis:")
        print(f"  - Maximum ROM: Rep #{max_rom_idx + 1}")
        print(f"  - Average ROM: {np.mean(rom_percentages):.1f}% of maximum")


if __name__ == "__main__":
    csv_filepath = 'test.csv'
    
    # Get workout analysis results
    results = analyze_workout_from_csv(csv_filepath, process_noise=0.003, measurement_noise=0.1)
    #print(results['sets'])
    path = dict_to_zip(results)
    send_mqtt(path,broker = None, port = None,TOPIC = None)

    # Visualize results
    #visualize_workout_analysis_from_csv(results)
