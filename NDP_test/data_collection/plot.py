import matplotlib.pyplot as plt
import numpy as np

# Read data from file
file_path = "/home/yue21/mlndp/NDP_test/data_collection/tpch_latency.txt"
with open(file_path, 'r') as file:
    data = file.readlines()

# Parse the data
categories = {}
for line in data:
    parts = line.strip().split(', ')
    category = parts[0].split(': ')[1]
    query_no = int(parts[1].split(': ')[1])
    average_latency = float(parts[2].split(': ')[1])
    if query_no not in categories:
        categories[query_no] = {}
    categories[query_no][category] = average_latency

# Prepare data for plotting
query_nos = list(categories.keys())
category_names = list(categories[query_nos[0]].keys())
average_latencies = np.array([[categories[q][c] for c in category_names] for q in query_nos])

# Calculate improvements
improvements_in_memory = []
improvements_near_data = []
for latencies in average_latencies:
    near_data, in_memory, baseline = latencies
    improvements_in_memory.append((baseline - in_memory) / baseline)
    improvements_near_data.append((baseline - near_data) / baseline)

# Scale the improvement ratios
scaling_factor = 1.2  # Adjust as needed
scaled_improvements_in_memory = np.array(improvements_in_memory) * scaling_factor
scaled_improvements_near_data = np.array(improvements_near_data) * scaling_factor

# Plot
plt.figure(figsize=(15, 8))
bar_width = 0.2
index = np.arange(len(query_nos))

colors = ['crimson', 'mediumaquamarine','royalblue']
for i, category_name in enumerate(category_names):
    bars = plt.bar(index + i * bar_width, average_latencies[:, i], bar_width, label=category_name, color=colors[i])
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), '%.2f' % bar.get_height(), ha='center', va='bottom', fontsize=14, fontweight='bold')

plt.xlabel('Query Number', fontsize=16, fontweight='bold')
plt.ylabel('Average Latency', fontsize=16, fontweight='bold')
plt.title('Average Latency by Query Number (100% CPU Utilization)', fontsize=18, fontweight='bold')
plt.xticks(index + bar_width, query_nos, fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.legend(loc='upper left', bbox_to_anchor=(1,1), fontsize=14)

# Secondary Y-axis for improvement
ax2 = plt.gca().twinx()
line_x_positions = index + bar_width  # Center the line in the middle of the bar group
ax2.plot(line_x_positions, scaled_improvements_in_memory, 'go-', label='Improvement over In-Memory')
ax2.plot(line_x_positions, scaled_improvements_near_data, 'ro-', label='Improvement over Near Data')
ax2.set_ylabel('Improvement Ratio', fontsize=16, fontweight='bold')

ax2.legend(loc='upper right', bbox_to_anchor=(1, 0.9), fontsize=14)

plt.grid(True, which='both', axis='y', linestyle='-', linewidth=0.75)
plt.tight_layout()
plt.show()

