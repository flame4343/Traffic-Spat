import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# Font settings
# ---------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# ---------------------------
# Parameter settings
# ---------------------------
TARGET_LINK_ID = '455135'
PERIOD = (130 if TARGET_LINK_ID in ['46333684', '46333701', '455137', '455135']
          else 120 if TARGET_LINK_ID in ['508612029', '19615665', '508385934', '604385887']
          else 1200)

LOCAL_TIMEZONE = 'Asia/Shanghai'
SKIP_DAYS = [4, 5, 14, 15, 21, 22, 28]
ANALYSIS_START = "07:00:00"
ANALYSIS_END = "09:00:00"
BIN_SIZE_SEC_START = 1
BIN_SIZE_SEC_STOP = 1
DISTANCE_THRESHOLD = 5
SMOOTHING_WINDOW = 5

# Time window in seconds
time_window = pd.to_datetime(ANALYSIS_END) - pd.to_datetime(ANALYSIS_START)
TIME_WINDOW_SECONDS = int(time_window.total_seconds())

# ---------------------------
# Data loading and preprocessing
# ---------------------------
column_names = [
    'trace_id', 'cross_id', 'stop_line_length',
    'in_link_id', 'out_link_id',
    'in_link_entry_time', 'in_link_out_time',
    'over_stop_line_time', 'stop_info'
]
df = pd.read_csv(
    r'C:\Users\Mingcheng Liao\OneDrive\实习\Navinfo\TrafficPredict\SPaT '
    r'Estimation with FCD\Dataset\stop_line_trace_new2.csv',
    names=column_names, header=0
)
print(f"File loaded successfully: {len(df)} rows")

df['in_link_entry_time']   /= 1000
df['in_link_out_time']     /= 1000
df['over_stop_line_time']  /= 1000

df['Link_ID'] = df['in_link_id'].astype(str)
filtered_df = df[df['Link_ID'] == TARGET_LINK_ID].copy()
print(f"Rows after filtering for target link: {len(filtered_df)}")

def extract_start_ts_and_distance(stopinfo):
    if pd.isna(stopinfo) or not isinstance(stopinfo, str):
        return []
    fields = stopinfo.split(',')
    extracted = []
    for i in range(0, len(fields), 3):
        chunk = fields[i:i+3]
        if len(chunk) < 3:
            continue
        try:
            wait = float(chunk[0]) / 1000
            d    = float(chunk[1])
            if d < 0:
                return []
            ts   = int(float(chunk[2]) / 1000)
            extracted.append((d, wait, ts))
        except ValueError:
            continue
    return extracted

filtered_df['start_d_ts'] = filtered_df['stop_info'].apply(extract_start_ts_and_distance)
start_exploded = filtered_df.explode('start_d_ts').dropna(subset=['start_d_ts'])
start_exploded[['d','wait','start']] = pd.DataFrame(
    start_exploded['start_d_ts'].tolist(),
    index=start_exploded.index
)
start_exploded = start_exploded[start_exploded['d'] <= 100]

start_exploded['ts_correction'] = -0.15 * start_exploded['d']
start_exploded['start_ts'] = (
    start_exploded['start']
    - start_exploded['ts_correction']
    + start_exploded['wait']
)

filtered_df['outtime'] = pd.to_numeric(filtered_df['over_stop_line_time'], errors='coerce')
filtered_df.dropna(subset=['outtime'], inplace=True)
filtered_df['real_start_ts'] = filtered_df['outtime']

start_exploded['local_time'] = (
    pd.to_datetime(start_exploded['start_ts'], unit='s', utc=True)
      .dt.tz_convert(LOCAL_TIMEZONE)
)
filtered_df['local_time'] = (
    pd.to_datetime(filtered_df['real_start_ts'], unit='s', utc=True)
      .dt.tz_convert(LOCAL_TIMEZONE)
)

start_exploded['date'] = start_exploded['local_time'].dt.date
start_exploded['day']  = start_exploded['local_time'].dt.day
filtered_df['date'] = filtered_df['local_time'].dt.date
filtered_df['day']  = filtered_df['local_time'].dt.day

start_exploded = start_exploded[~start_exploded['day'].isin(SKIP_DAYS)]
filtered_df    = filtered_df   [~filtered_df['day'].isin(SKIP_DAYS)]
print(f"After skipping specified dates: parking events={len(start_exploded)}, passing events={len(filtered_df)}")

start_exploded = start_exploded[
    (start_exploded['local_time'].dt.time >= pd.to_datetime(ANALYSIS_START).time()) &
    (start_exploded['local_time'].dt.time <  pd.to_datetime(ANALYSIS_END).time())
]
filtered_df = filtered_df[
    (filtered_df['local_time'].dt.time >= pd.to_datetime(ANALYSIS_START).time()) &
    (filtered_df['local_time'].dt.time <  pd.to_datetime(ANALYSIS_END).time())
]
print(f"In analysis interval: parking events={len(start_exploded)}, passing events={len(filtered_df)}")

starts_within_threshold = start_exploded[start_exploded['d'] <= DISTANCE_THRESHOLD]
first_starts = (
    starts_within_threshold
      .groupby('date')['local_time']
      .min()
      .rename('day_first_start')
      .reset_index()
)
first_starts['day_base'] = first_starts['date'].apply(
    lambda d: pd.Timestamp(f"{d} {ANALYSIS_START}", tz=LOCAL_TIMEZONE)
)
first_starts['shift_timedelta'] = first_starts['day_base'] - first_starts['day_first_start']

start_exploded = start_exploded.merge(
    first_starts[['date','day_base','shift_timedelta']],
    on='date', how='left'
)
filtered_df = filtered_df.merge(
    first_starts[['date','day_base','shift_timedelta']],
    on='date', how='left'
)

start_exploded.dropna(subset=['shift_timedelta'], inplace=True)
filtered_df.dropna(subset=['shift_timedelta'], inplace=True)

start_exploded['adjusted_time'] = start_exploded['local_time'] + start_exploded['shift_timedelta']
filtered_df['adjusted_time']    = filtered_df['local_time']    + filtered_df['shift_timedelta']

start_exploded['relative_seconds'] = (
    start_exploded['adjusted_time'] - start_exploded['day_base']
).dt.total_seconds()
filtered_df['relative_seconds'] = (
    filtered_df['adjusted_time'] - filtered_df['day_base']
).dt.total_seconds()

start_exploded['shifted_bin'] = (
    start_exploded['relative_seconds'] // BIN_SIZE_SEC_START
).fillna(0).astype(int)
filtered_df['shifted_bin']    = (
    filtered_df['relative_seconds']    // BIN_SIZE_SEC_STOP
).fillna(0).astype(int)

all_bins_start = np.arange(0, TIME_WINDOW_SECONDS + BIN_SIZE_SEC_START, BIN_SIZE_SEC_START)
all_bins_stop  = np.arange(0, TIME_WINDOW_SECONDS + BIN_SIZE_SEC_STOP,  BIN_SIZE_SEC_STOP)

start_counts = start_exploded['shifted_bin'].value_counts().reindex(all_bins_start, fill_value=0)
stop_counts  = filtered_df   ['shifted_bin'].value_counts().reindex(all_bins_stop,  fill_value=0)

occupied_seconds = []
for _, row in start_exploded.iterrows():
    start_sec = row['relative_seconds'] - row['wait']
    end_sec   = row['relative_seconds']
    start_sec = max(0, start_sec)
    end_sec   = min(TIME_WINDOW_SECONDS, end_sec)
    occupied_seconds.extend(
        range(int(np.floor(start_sec)), int(np.floor(end_sec)))
    )

occupied_mod = np.array(occupied_seconds) % PERIOD
occupancy_hist, occupancy_bins = np.histogram(
    occupied_mod, bins=PERIOD, range=(0, PERIOD)
)

filtered_df['mod_vals'] = filtered_df['shifted_bin'] % PERIOD
passing_hist, passing_bins = np.histogram(
    filtered_df['mod_vals'], bins=PERIOD, range=(0, PERIOD)
)

kernel = np.ones(SMOOTHING_WINDOW) / SMOOTHING_WINDOW
smoothed_occupancy_hist = np.convolve(occupancy_hist, kernel, mode='same')
smoothed_passing_hist   = np.convolve(passing_hist,   kernel, mode='same')

smoothed_occupancy_hist /= smoothed_occupancy_hist.sum()
smoothed_passing_hist   /= smoothed_passing_hist.sum()

# Remove low-frequency bins
smoothed_occupancy_hist[smoothed_occupancy_hist < 0.0005] = 0
smoothed_passing_hist  [smoothed_passing_hist   < 0.002 ] = 0

# 🚦 Green light estimation (low-frequency start + continuous rise + weighted fusion)
low_threshold     = np.percentile(smoothed_occupancy_hist[smoothed_occupancy_hist > 0], 10)
min_rising_len    = 3
min_rise_position = int(PERIOD * 0.1)

# 1. Find a continuous rising segment (start after 10% of cycle)
low_freq_indices = np.where(smoothed_occupancy_hist < low_threshold)[0]
estimated_index   = None
for i in low_freq_indices:
    if i < min_rise_position or i + min_rising_len >= len(smoothed_occupancy_hist):
        continue
    segment = smoothed_occupancy_hist[i:i + min_rising_len + 1]
    if np.all(np.diff(segment) > 0):
        estimated_index = i
        break

# 2. Fallback: use maximum slope point if no rising segment
if estimated_index is None:
    occupancy_diff = np.diff(smoothed_occupancy_hist)
    search_range   = occupancy_diff[min_rise_position:]
    estimated_index = min_rise_position + int(np.argmax(search_range))
    print("⚠️ No continuous rising segment found; using max slope point as fallback")

max_rise_index = estimated_index

# 3. Find the first zero in passing histogram
zero_indices     = np.where(smoothed_passing_hist == 0)[0]
first_zero_index = zero_indices[0] if len(zero_indices) > 0 else None

# Logging intermediate results
print(f"🚗 max_rise_index (parking surge position): {max_rise_index}")
print(f"🚦 first_zero_index (first zero passing) index: {first_zero_index}")

# 4. Weighted fusion estimate (if passing data is valid)
if first_zero_index is not None:
    estimated_green_light_bin = int(
        (len(filtered_df) * max_rise_index + len(start_exploded) * first_zero_index)
        / (len(start_exploded) + len(filtered_df))
    )
    print(f"📊 Weighted calculation details: {len(filtered_df)}*{max_rise_index} + {len(start_exploded)}*{first_zero_index}")
    print(f"Red light duration: {PERIOD - estimated_green_light_bin} seconds")
else:
    estimated_green_light_bin = max_rise_index
    print(f"⚠️ No zero passing position detected; using surge point {estimated_green_light_bin}")

# 📏 Comparison with manual markings
mark_points = {
    '455135': [64, 68],
    '455137': [64, 68],
    '46333701': [37, 41],
    '46333684': [37, 41],
    '508612029': [37, 41],
    '19615665': [42, 46],
    '508385934': [43, 47],
    '604385887': [37, 41]
}

if TARGET_LINK_ID in mark_points:
    true_green_start = mark_points[TARGET_LINK_ID][1]
    diff = abs(estimated_green_light_bin - true_green_start)
    print(f"Distance to manual mark: {diff} seconds")
else:
    print("⚠️ No manual marking available for comparison")

plt.figure(figsize=(10, 6))

# Plot "Parking" area: red fill without border lines
plt.fill_between(
    occupancy_bins[:-1],
    smoothed_occupancy_hist,
    color='red',
    alpha=0.7,
    label='Parking'
)

# Plot "Passing" area: green fill without border lines
plt.fill_between(
    passing_bins[:-1],
    smoothed_passing_hist,
    color='green',
    alpha=0.7,
    label='Passing'
)

# Compute the difference and find crossing point for signal change
diff_curve = smoothed_occupancy_hist - smoothed_passing_hist
cross_indices = np.where(np.diff(np.sign(diff_curve)))[0]
red_minimum_point = estimated_green_light_bin
valid_crossing_indices = [idx for idx in cross_indices if occupancy_bins[idx] > red_minimum_point]

if valid_crossing_indices:
    idx = valid_crossing_indices[0]
    x0, x1 = occupancy_bins[idx], occupancy_bins[idx + 1]
    d0, d1 = diff_curve[idx], diff_curve[idx + 1]
    crossing_point = x0 - d0 * (x1 - x0) / (d1 - d0) if d1 != d0 else x0

    ax = plt.gca()
    plt.draw()
    _, ymax = ax.get_ylim()

    # Highlight region between red_minimum_point and crossing_point
    plt.fill_between(
        [red_minimum_point, crossing_point],
        [0, 0],
        [ymax, ymax],
        color='yellow',
        alpha=0.5,
        label='Possible signal\n changing time'
    )
else:
    crossing_point = None

plt.xlabel(f'Mapped Time Range [0, {PERIOD}) Seconds', fontsize=24)
plt.ylabel('Normalized Frequency', fontsize=24)
plt.title(f'Parking vs Passing Time Comparison\n(Period = {PERIOD} seconds)', fontsize=25)
plt.legend(loc='upper right', fontsize=24)
plt.tick_params(axis='both', labelsize=17)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()

plt.savefig('normalized_smoothed_occupancy_passing_combined_hist.png')
plt.show()
print("Figure saved as: 'normalized_smoothed_occupancy_passing_combined_hist.png'")
