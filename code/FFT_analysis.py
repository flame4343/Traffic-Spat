import pandas as pd
import matplotlib.pyplot as plt
import pytz
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize


# Configure matplotlib to use fonts that support Chinese
plt.rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei font
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False  # Fix issue where minus sign '-' appears as a square

# Define constants
INPUT_FILE = r'C:\Users\Mingcheng Liao\OneDrive\实习\Navinfo\TrafficPredict\SPaT Estimation with FCD\Dataset\rt_car_light_travel_202501101539.csv'  # Replace with your original CSV file path
OUTPUT_FILE = 'filtered_with_start_timestamps.csv'  # File path to save pre-processed CSV
OUTPUT_IMAGE_HIST = 'selected_window_hist.png'  # Output path for the final histogram
OUTPUT_IMAGE_SCATTER = 'selected_window_scatter.png'  # Output path for the scatter plot
OUTPUT_CSV = 'final_counts.csv'  # File path to save the final counts CSV

TARGET_LINK_ID = '455137'  # Target Link ID
LOCAL_TIMEZONE = 'Asia/Shanghai'  # Local timezone, e.g., Beijing time
SKIP_DAYS = [4, 5, 14, 15, 21, 22, 28]  # List of dates to skip

# Time interval definition (7:00 => 25200 seconds, 9:00 => 32400 seconds)
INTERVAL_START_SEC = 7 * 3600  # 25200 seconds, i.e., 7:00 AM
INTERVAL_END_SEC = 9 * 3600    # 32400 seconds, i.e., 9:00 AM
BIN_SIZE_SEC = 3               # Size of each bin (3 seconds)

# Optional plotting time window (seconds relative to 7:00 AM)
# For example, to select 7:30 to 8:00 AM, set to 1800 to 3600 seconds
# If not specified, the entire interval is plotted by default
PLOT_WINDOW_START_SEC = 0     # 0 seconds, i.e., 7:00 AM
PLOT_WINDOW_END_SEC = 7200    # 7200 seconds, i.e., 9:00 AM


# ==============================
# 1. Define parsing function
# ==============================
def extract_start_ts_and_distance(stopinfo):
    """
    Extract all start timestamps and corresponding distances d from the stopinfo field.
    Each start event consists of 5 fields in order: d, ?, ts, ?, ?.
    Returns a list of (d, ts) tuples.
    """
    if pd.isna(stopinfo) or not isinstance(stopinfo, str):
        return []
    fields = stopinfo.split(',')
    extracted = []
    # Each 5 fields represent one start event
    for i in range(0, len(fields), 5):
        chunk = fields[i:i + 5]
        if len(chunk) < 5:
            continue  # Ignore incomplete events
        try:
            wait = float(chunk[0])
            d = float(chunk[1])  # First field is distance d
            ts = float(chunk[2]) + wait  # Third field is start timestamp ts
            extracted.append((d, ts))
        except ValueError:
            continue  # Ignore fields that cannot be converted
    return extracted


# ==============================
# 2. Read the CSV and extract start timestamps and distances
# ==============================
print(f"Reading file: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE, encoding='utf-8')

# Ensure 'entryid' is of string type
df['entryid'] = df['entryid'].astype(str)

# Extract 'Link_ID' (assuming last 8 characters of 'entryid')
df['Link_ID'] = df['entryid'].str[7:]  # Correct extraction method: use the last 8 characters

# Filter data for the target Link ID
filtered_df = df[df['Link_ID'] == TARGET_LINK_ID].copy()
print(f"Rows after filtering for Link_ID {TARGET_LINK_ID}: {len(filtered_df)}")

# Extract start timestamps and distances d from 'stopinfo'
filtered_df['start_d_ts'] = filtered_df['stopinfo'].apply(extract_start_ts_and_distance)

# Save pre-processed data
filtered_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
print(f"Preprocessed data saved to '{OUTPUT_FILE}'")


# ==============================
# 3. Consolidate all timestamps, convert to local time, and filter
# ==============================
# Explode the list and create DataFrame
exploded = filtered_df.explode('start_d_ts').dropna(subset=['start_d_ts'])

# Split (d, ts) tuples into two separate columns
exploded[['d', 'original_ts']] = pd.DataFrame(exploded['start_d_ts'].tolist(), index=exploded.index)

# ==============================
# 3.1. Filter data where d <= 100
# ==============================
exploded = exploded[exploded['d'] <= 100]
print(f"Rows where d ≤ 100: {len(exploded)}")

# Calculate correction values and adjust start timestamps
exploded['correction'] = exploded['d'] * 0.14300298367472764 + 0.4760282403296463  # Correction formula
exploded['real_start_ts'] = exploded['original_ts'] - exploded['correction']  # Real start timestamp

# Check and filter negative real start timestamps
negative_ts = exploded[exploded['real_start_ts'] < 0]
if not negative_ts.empty:
    print(f"Warning: {len(negative_ts)} negative real start timestamps found; filtering these out.")
    exploded = exploded[exploded['real_start_ts'] >= 0]

# Convert to local time
# Convert 'real_start_ts' to UTC time
real_start_times_utc = pd.to_datetime(exploded['real_start_ts'], unit='s', utc=True)

# Convert to local timezone
real_start_times_local = real_start_times_utc.dt.tz_convert(LOCAL_TIMEZONE)
exploded['local_time'] = real_start_times_local

# Add date and day columns
exploded['date'] = exploded['local_time'].dt.date
exploded['day'] = exploded['local_time'].dt.day

# Skip specified dates
exploded = exploded[~exploded['day'].isin(SKIP_DAYS)]
print(f"Rows remaining after skipping dates {SKIP_DAYS}: {len(exploded)}")

# ==============================
# 4. Filter data between 7:00 and 9:00 AM
# ==============================
exploded = exploded[
    (exploded['local_time'].dt.time >= pd.to_datetime('07:00:00').time()) &
    (exploded['local_time'].dt.time < pd.to_datetime('09:00:00').time())
]
print(f"Rows between 07:00 and 09:00: {len(exploded)}")

# ==============================
# 5. Calculate relative seconds and map
# ==============================
# Calculate relative seconds separately for each day
first_times = exploded.groupby('date')['local_time'].min()

# Calculate relative seconds
exploded = exploded.merge(first_times.rename('day_start_time'), on='date')
exploded['relative_seconds'] = (exploded['local_time'] - exploded['day_start_time']).dt.total_seconds()

# Map 7:00 to 0 seconds and 9:00 to 7200 seconds
exploded['shifted_seconds'] = exploded['relative_seconds']  # Already filtered from 7:00, so this is relative seconds

# Only keep data in the range 0 to 7200 seconds (07:00 to 09:00)
exploded = exploded[
    (exploded['shifted_seconds'] >= 0) &
    (exploded['shifted_seconds'] <= (INTERVAL_END_SEC - INTERVAL_START_SEC))
]
print(f"Rows with relative seconds in 0–{INTERVAL_END_SEC - INTERVAL_START_SEC} s: {len(exploded)}")

# ==============================
# 6. Bin data every BIN_SIZE_SEC seconds and fill zeros
# ==============================
exploded['shifted_bin'] = (exploded['shifted_seconds'] // BIN_SIZE_SEC) * BIN_SIZE_SEC
exploded['shifted_bin'] = exploded['shifted_bin'].astype(int)

# Create all possible bins (from 0 to 7200, step BIN_SIZE_SEC seconds)
all_bins = np.arange(0, (INTERVAL_END_SEC - INTERVAL_START_SEC) + BIN_SIZE_SEC, BIN_SIZE_SEC)

# Count frequency of each bin
counts = exploded['shifted_bin'].value_counts().sort_index()
counts = counts.reindex(all_bins, fill_value=0)

# ==============================
# 7. Export final statistics to CSV
# ==============================
counts_df = counts.reset_index()
counts_df.columns = ['timestamp', 'data']
counts_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
print(f"Final statistics saved to '{OUTPUT_CSV}'")

# ==============================
# 12. Plot FFT amplitude spectrum for the full interval
# ==============================
# Compute FFT of the detrended counts data
Y = np.fft.fft(counts.values - np.mean(counts.values))
freqs = np.fft.fftfreq(len(counts.values), d=BIN_SIZE_SEC)
N_half = len(counts.values) // 2
freqs_pos = freqs[:N_half]
amplitude = np.abs(Y[:N_half])

# Find the top five peaks
top5_idx = np.argsort(amplitude)[-5:][::-1]
top5_freqs = freqs_pos[top5_idx]
top5_amps = amplitude[top5_idx]

print("Top five peaks in full-window FFT:")
for i, (freq, amp) in enumerate(zip(top5_freqs, top5_amps), start=1):
    if freq > 0:
        period = 1.0 / freq
        print(f"Peak {i}: {period:.2f} s (freq = {freq:.6f} Hz, amp = {amp:.2f})")

# Plot only the full-window spectrum
plt.figure(figsize=(10, 6))
plt.plot(freqs_pos, amplitude)
plt.title('FFT Amplitude Spectrum (Full Window)', fontsize=23)
plt.xlabel('Frequency (Hz)', fontsize=22)
plt.ylabel('Amplitude', fontsize=22)
plt.grid(True)
plt.xlim(0, 0.05)

for freq, amp in zip(top5_freqs, top5_amps):
    if 0 < freq < 0.05:
        period = 1.0 / freq
        plt.scatter(freq, amp, s=80, zorder=5, label=f'Peak: {period:.2f} s')

plt.legend(fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig('fft_amplitude_spectrum_full_window.png')
plt.show()
