import pandas as pd
import matplotlib.pyplot as plt
import pytz
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from tqdm import tqdm
from datetime import datetime, timedelta

# Configure matplotlib fonts
plt.rcParams['font.sans-serif'] = ['SimHei']  # Still using SimHei for Chinese support if needed
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign rendering

# Constants
INPUT_FILE = r'C:\Users\Mingcheng Liao\OneDrive\实习\Navinfo\TrafficPredict\SPaT Estimation with FCD\Dataset\rt_car_light_travel_202501101539.csv'
OUTPUT_FILE = 'filtered_with_start_timestamps.csv'
LOCAL_TIMEZONE = 'Asia/Shanghai'
SKIP_DAYS = [4, 5, 14, 15, 21, 22, 28]


def generate_time_windows(pivot_time_str):
    """
    Given a pivot time 'HH:MM:SS', return two adjacent 30-minute windows:
      - window1 is [pivot−30min, pivot)
      - window2 is [pivot, pivot+30min)
    Each window dict contains:
      'name', 'start_time', 'end_time', and 'output_prefix'.
    """
    pivot = pd.to_datetime(pivot_time_str)
    window1_start = (pivot - pd.Timedelta(minutes=30)).time()
    window1_end   = pivot.time()
    window2_start = pivot.time()
    window2_end   = (pivot + pd.Timedelta(minutes=30)).time()

    return [
        {
            'name': f"{window1_start.strftime('%H:%M')}-{window1_end.strftime('%H:%M')}",
            'start_time': window1_start,
            'end_time': window1_end,
            'output_prefix': 'window1'
        },
        {
            'name': f"{window2_start.strftime('%H:%M')}-{window2_end.strftime('%H:%M')}",
            'start_time': window2_start,
            'end_time': window2_end,
            'output_prefix': 'window2'
        }
    ]


def extract_start_ts_and_distance(stopinfo):
    """
    Parse stopinfo string into a list of (distance d, timestamp ts) tuples.
    Each event is encoded as 5 comma-separated fields: d, ?, ts, ?, ?.
    """
    if pd.isna(stopinfo) or not isinstance(stopinfo, str):
        return []
    fields = stopinfo.split(',')
    result = []
    for i in range(0, len(fields), 5):
        chunk = fields[i:i+5]
        if len(chunk) < 5:
            continue
        try:
            wait = float(chunk[0])
            d    = float(chunk[1])
            ts   = float(chunk[2]) + wait
            result.append((d, ts))
        except ValueError:
            continue
    return result


def compute_degree_of_dispersion(times, C):
    """
    Compute dispersion for cycle C on list of green-start times (in seconds).
    Returns normalized RMS distance to the centroid.
    """
    if len(times) == 0:
        return np.inf
    arr = np.array(times)
    # Pairwise circular distances
    diff = np.abs(arr[:, None] - arr[None, :]) % C
    dist = np.minimum(diff, C - diff)
    sums = dist.sum(axis=1)
    centroid = arr[np.argmin(sums)]
    d_centroid = np.abs(centroid - arr) % C
    d_centroid = np.minimum(d_centroid, C - d_centroid)
    rms = np.sqrt((d_centroid**2).sum() / len(arr))
    return rms / C


def compute_penalty(times, C, m_0_i, w=0.5):
    """
    Compute a penalty term for cycle C (example formula).
    """
    return 0.1 * (1 - C/600)**2


def estimate_optimal_cycle(green_times, C_candidates, m_0_i, w=0.5):
    """
    For each C in C_candidates, compute:
      - dispersion
      - penalty
      - modified_dispersion = dispersion + penalty
    Return optimal C, its modified_dispersion, and all series.
    """
    dispersions = []
    penalties   = []
    modified    = []

    for C in tqdm(C_candidates, desc="Estimating optimal cycle"):
        d = compute_degree_of_dispersion(green_times, C)
        p = compute_penalty(green_times, C, m_0_i, w)
        dispersions.append(d)
        penalties.append(p)
        modified.append(d + p)

    idx = np.argmin(modified)
    return (C_candidates[idx],
            modified[idx],
            C_candidates,
            dispersions,
            penalties,
            modified)


def process_traffic_data(TARGET_LINK_ID, pivot_time_str):
    """
    1. Read raw CSV and filter by TARGET_LINK_ID.
    2. Extract start timestamps and distances.
    3. Convert to local time, skip bad days.
    4. For each of two 30-minute windows around pivot_time_str,
       estimate optimal cycle via modified dispersion.
    Returns list of optimal cycles and the maximum of their min_disp.
    """
    print(f"Reading input file: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, encoding='utf-8')
    df['entryid'] = df['entryid'].astype(str)
    df['Link_ID'] = df['entryid'].str[7:]
    df = df[df['Link_ID'] == TARGET_LINK_ID].copy()
    print(f"Rows after filtering: {len(df)}")

    df['start_d_ts'] = df['stopinfo'].apply(extract_start_ts_and_distance)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Preprocessed data saved to: {OUTPUT_FILE}")

    # Explode and clean
    exploded = df.explode('start_d_ts').dropna(subset=['start_d_ts'])
    exploded[['d', 'orig_ts']] = pd.DataFrame(
        exploded['start_d_ts'].tolist(), index=exploded.index
    )
    exploded = exploded[exploded['d'] <= 100]
    exploded['correction'] = exploded['d'] * 0.14300298367472764 + 0.4760282403296463
    exploded['real_ts'] = exploded['orig_ts'] - exploded['correction']
    exploded = exploded[exploded['real_ts'] >= 0]

    # Local time conversion
    exploded['local_time'] = (
        pd.to_datetime(exploded['real_ts'], unit='s', utc=True)
          .dt.tz_convert(LOCAL_TIMEZONE)
    )
    exploded['date'] = exploded['local_time'].dt.date
    exploded['day']  = exploded['local_time'].dt.day
    exploded = exploded[~exploded['day'].isin(SKIP_DAYS)]
    print(f"Rows after skipping days {SKIP_DAYS}: {len(exploded)}")

    # Generate two 30-min windows around pivot
    windows = generate_time_windows(pivot_time_str)

    optimal_cycles = []
    min_disps      = []
    for win in windows:
        print(f"\nProcessing window: {win['name']}  ({win['start_time']}–{win['end_time']})")
        wdata = exploded[
            (exploded['local_time'].dt.time >= win['start_time']) &
            (exploded['local_time'].dt.time <  win['end_time'])
        ].copy()
        print(f"Window data rows: {len(wdata)}")

        # Seconds from window start
        base = win['start_time'].hour*3600 + win['start_time'].minute*60
        wdata['shifted_seconds'] = (
            wdata['local_time'].dt.hour*3600 +
            wdata['local_time'].dt.minute*60 +
            wdata['local_time'].dt.second
            - base
        )
        wdata = wdata[(wdata['shifted_seconds'] >= 0) &
                      (wdata['shifted_seconds'] <= 30*60)]

        green_times = wdata['shifted_seconds'].tolist()
        C_cand = np.arange(10, 181, 1)

        opt_C, min_disp, _, _, _, _ = estimate_optimal_cycle(
            green_times,
            C_cand,
            m_0_i=2,
            w=0.5
        )
        print(f"  Optimal cycle: {opt_C} s, Min modified dispersion: {min_disp:.4f}")

        optimal_cycles.append(opt_C)
        min_disps.append(min_disp)

    return optimal_cycles, max(min_disps)
