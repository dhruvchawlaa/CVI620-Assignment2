import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csv_path = 'Driving Log Data/driving_log.csv'

df = pd.read_csv(csv_path)

steering_angles = df['Steering'].values

min_steering = np.min(steering_angles)
max_steering = np.max(steering_angles)

print(f"Minimum Steering Angle: {min_steering}")
print(f"Maximum Steering Angle: {max_steering}")

counts, bins, _ = plt.hist(steering_angles, bins=50, range=(min_steering, max_steering), rwidth=0.8)
plt.xlabel('Steering Angle')
plt.ylabel('Frequency')
plt.title('Distribution of Steering Angles')

max_frequency = np.max(counts)
plt.ylim(0, max_frequency * 1.1) 
plt.savefig('steering_histogram.png')
plt.show()