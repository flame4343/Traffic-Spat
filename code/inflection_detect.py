import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Configure matplotlib fonts
plt.rcParams['font.sans-serif'] = ['SimHei']  # support for Chinese characters if needed
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False  # fix negative sign rendering

# Define constants
INPUT_FILE = r'C:\Users\Mingcheng Liao\OneDrive\实习\Navinfo\TrafficPredict\SPaT Estimation with FCD\Dataset\rt_car_light_travel_202501101539.csv'
TARGET_LINK_ID = '608160357'
LOCAL_TIMEZONE = 'Asia/Shanghai'
SKIP_DAYS = [4, 5, 14, 15, 21, 22, 28]

# Time interval for analysis (7:00 to 9:00 => 25200–32400 seconds)
INTERVAL_START_SEC = 7 * 3600
INTERVAL_END_SEC   = 9 * 3600


def extract_wait_times(stopinfo):
    """
    Parse the 'stopinfo' field and return a list of wait durations.
    Each event is encoded as 5 comma-separated fields: wait, d, ts, ?, ?.
    """
    if pd.isna(stopinfo) or not isinstance(stopinfo, str):
        return []
    fields = stopinfo.split(',')
    waits = []
    for i in range(0, len(fields), 5):
        chunk = fields[i:i + 5]
        if len(chunk) < 5:
            continue
        try:
            wait = float(chunk[0])
            waits.append(wait)
        except ValueError:
            continue
    return waits


def extract_ts(stopinfo):
    """
    Parse the 'stopinfo' field and return a list of start timestamps.
    Each event is encoded as 5 comma-separated fields: wait, d, ts, ?, ?.
    """
    if pd.isna(stopinfo) or not isinstance(stopinfo, str):
        return []
    fields = stopinfo.split(',')
    ts_list = []
    for i in range(0, len(fields), 5):
        chunk = fields[i:i + 5]
        if len(chunk) < 5:
            continue
        try:
            ts = float(chunk[2])
            ts_list.append(ts)
        except ValueError:
            continue
    return ts_list


# 2. Read CSV and extract wait times and timestamps
print(f"Reading file: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE, encoding='utf-8')

df['entryid'] = df['entryid'].astype(str)
df['Link_ID'] = df['entryid'].str[7:]

filtered_df = df[(df['Link_ID'] == TARGET_LINK_ID) & (df.get('laneflag', 0) >= 1)].copy()
print(f"Rows after filtering for Link_ID {TARGET_LINK_ID}: {len(filtered_df)}")

filtered_df['wait_times'] = filtered_df['stopinfo'].apply(extract_wait_times)
filtered_df['ts_times']   = filtered_df['stopinfo'].apply(extract_ts)

# 3. Explode wait and timestamp lists into aligned columns
exploded_wait = filtered_df.explode('wait_times').dropna(subset=['wait_times']).reset_index(drop=True)
exploded_ts   = filtered_df.explode('ts_times'  ).dropna(subset=['ts_times'  ]).reset_index(drop=True)

exploded = pd.concat(
    [exploded_wait['wait_times'], exploded_ts['ts_times']], axis=1
).rename(columns={'wait_times': 'wait', 'ts_times': 'ts'})

# 4. Convert timestamps and filter
exploded['real_start_ts'] = exploded['ts']

negative_ts = exploded[exploded['real_start_ts'] < 0]
if not negative_ts.empty:
    print(f"Warning: {len(negative_ts)} negative timestamps found; filtering out.")
    exploded = exploded[exploded['real_start_ts'] >= 0]

exploded['local_time'] = (
    pd.to_datetime(exploded['real_start_ts'], unit='s', utc=True)
      .dt.tz_convert(LOCAL_TIMEZONE)
)

exploded['date'] = exploded['local_time'].dt.date
exploded['day']  = exploded['local_time'].dt.day

exploded = exploded[~exploded['day'].isin(SKIP_DAYS)]
print(f"Rows after skipping days {SKIP_DAYS}: {len(exploded)}")

# 5. Keep only 07:00–09:00 data
exploded = exploded[
    (exploded['local_time'].dt.time >= pd.to_datetime('07:00:00').time()) &
    (exploded['local_time'].dt.time <  pd.to_datetime('09:00:00').time())
]
print(f"Rows between 07:00 and 09:00: {len(exploded)}")

# 6. Extract list of wait durations
wait_list = exploded['wait'].tolist()
print(f"Extracted {len(wait_list)} waiting times")

# 7. Sort and prepare for fitting
wait_list_sorted = sorted(wait_list)
indices_sorted   = list(range(1, len(wait_list_sorted) + 1))


def linear(x, a, b):
    """Linear function a*x + b for curve fitting."""
    x_arr = np.array(x)
    return a * x_arr + b


def find_turning_points(indices, data, window_size=3, threshold_factor=10):
    """
    Calculate gradient of 'data' and identify points where the change
    exceeds 'threshold_factor' times the mean gradient stddev.
    """
    gradients = np.gradient(data)
    gradient_std = np.array([
        np.std(gradients[max(0, i - window_size):i + window_size])
        for i in range(len(gradients))
    ])
    threshold = threshold_factor * np.mean(gradient_std)

    turning_points = [
        i for i in range(1, len(gradients))
        if abs(gradients[i] - gradients[i - 1]) > threshold
    ]
    return turning_points, gradient_std


def exclude_turning_areas(indices, data, turning_points, window_size=20):
    """
    Exclude regions around each turning point from the index list.
    """
    exclude = set()
    for t in turning_points:
        start = max(0, t - window_size)
        end   = min(len(data), t + window_size)
        exclude.update(range(start, end))
    return [i for i in indices if i - 1 not in exclude]


def find_best_split(indices, data):
    """
    Identify a split point at the first turning point (or midpoint),
    then fit two linear models on low/high regions.
    """
    turning_pts, grad_std = find_turning_points(indices, data)
    if turning_pts:
        split = turning_pts[0]
    else:
        split = len(indices) // 2

    low_idx  = indices[:split]
    high_idx = indices[split:]

    p_low,  _ = curve_fit(linear, low_idx,  data[:split])
    p_high, _ = curve_fit(linear, high_idx, data[split:])

    return split, p_low, p_high, turning_pts, grad_std


def calculate_intersection(p_low, p_high):
    """
    Compute intersection of y = a1 x + b1 and y = a2 x + b2.
    """
    a1, b1 = p_low
    a2, b2 = p_high
    x_int = (b2 - b1) / (a1 - a2)
    y_int = linear([x_int], *p_low)[0]
    return x_int, y_int


def plot_results(indices, data, p_low, p_high, x_int, y_int, turning_pts, grad_std, split):
    """Plot data, fits, turning points, gradient std, and intersection."""
    plt.figure(figsize=(10, 6))
    plt.plot(indices, data, color='gray', alpha=0.5,
             linestyle='-', marker='o', markersize=4, label='Raw data')

    low_idx  = indices[:split]
    high_idx = indices[split:]

    plt.plot(low_idx,  linear(low_idx,  *p_low),  color='blue',  lw=2, label='Low region fit')
    plt.plot(high_idx, linear(high_idx, *p_high), color='red',   lw=2, label='High region fit')

    plt.scatter(x_int, y_int, color='purple', s=80,
                label=f'Intersection: ({x_int:.2f}, {y_int:.2f})')

    tp_x = [indices[i] for i in turning_pts]
    tp_y = [data[i]        for i in turning_pts]
    plt.scatter(tp_x, tp_y, color='green', marker='x', s=80, label='Turning points')

    plt.plot(indices, grad_std, linestyle='--', alpha=0.5,
             color='orange', label='Gradient std')

    plt.title("Fitting Result of Waiting Times", fontsize=25)
    plt.xlabel("Sorted Index", fontsize=24)
    plt.ylabel("Waiting Time (seconds)", fontsize=24)
    plt.legend(fontsize=24)
    plt.grid(True)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.tight_layout()
    plt.show()


# Execute fitting procedure
best_split, params_low, params_high, turning_pts, grad_std = find_best_split(indices_sorted, wait_list_sorted)
x_int, y_int = calculate_intersection(params_low, params_high)
plot_results(indices_sorted, wait_list_sorted,
             params_low, params_high,
             x_int, y_int,
             turning_pts, grad_std,
             best_split)
