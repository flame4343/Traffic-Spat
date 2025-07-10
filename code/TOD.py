from Dispersion import process_traffic_data
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Set target link ID
TARGET_LINK_ID = '455135'

# Set start and end times
start_time = datetime.strptime('09:20:00', '%H:%M:%S')
end_time   = datetime.strptime('09:50:00', '%H:%M:%S')

# Create a list of time points from start to end at 5-minute intervals
time_points = []
current_time = start_time
while current_time <= end_time:
    time_points.append(current_time.strftime('%H:%M:%S'))
    current_time += timedelta(minutes=5)

# Loop over each time point and call process_traffic_data
optimal_cycles_all = []
min_disps_all      = []
for pivot_time_str in time_points:
    print(f"Processing time point: {pivot_time_str}")
    optimal_cycles, total_min_disp = process_traffic_data(TARGET_LINK_ID, pivot_time_str)
    optimal_cycles_all.append(optimal_cycles)
    min_disps_all.append(total_min_disp)

# Print results for all time points
print("Optimal cycles for all time points:", optimal_cycles_all)
print("Sum of minimum corrected dispersion for all time points:", min_disps_all)

# Plot the dispersion values over time
plt.figure(figsize=(10, 6))
plt.plot(time_points,
         min_disps_all,
         marker='o',
         linestyle='-',
         color='black',
         label='Minimum Corrected Dispersion')

# Find and annotate the minimum point
min_disp_index = min_disps_all.index(min(min_disps_all))
min_disp_time  = time_points[min_disp_index]
min_disp_value = min(min_disps_all)

plt.annotate(
    f'Minimum: {min_disp_time}\n{min_disp_value:.4f}',
    xy=(min_disp_time, min_disp_value),
    xytext=(min_disp_time, min_disp_value + 0.03),
    arrowprops=dict(facecolor='red', arrowstyle="->"),
    fontsize=24,
    color='red'
)

# Set title and axis labels
plt.title('Sum of Minimum Corrected Dispersion for All Time Points', fontsize=24)
plt.xlabel('Time Point', fontsize=24)
plt.ylabel('Minimum Corrected Dispersion', fontsize=24)

# Adjust tick label font sizes and orientation
plt.xticks(rotation=0, fontsize=19)
plt.yticks(fontsize=19)

# Set y-axis range
plt.ylim(0.15, 0.3)

plt.tight_layout()
plt.legend(fontsize=24)
plt.show()
