import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_steering_histogram(csv_path='Driving Log Data/driving_log.csv'):
    # Read the CSV file
    columns = [
        'Center Image',
        'Left Image',
        'Right Image',
        'Steering',
        'Throttle',
        'Brake',
        'Speed'
    ]
    data_df = pd.read_csv(csv_path, names=columns, skiprows=1)
    
    # Get steering angles
    steering_angles = data_df['Steering'].values
    
    # Calculate min and max
    min_steering = np.min(steering_angles)
    max_steering = np.max(steering_angles)
    
    print(f"Minimum Steering Angle: {min_steering:.4f}")
    print(f"Maximum Steering Angle: {max_steering:.4f}")
    
    # Create figure with better styling
    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8')
    
    # Plot histogram
    counts, bins, _ = plt.hist(
        steering_angles, 
        bins=50, 
        range=(min_steering, max_steering), 
        color='royalblue',
        edgecolor='black',
        alpha=0.7,
        rwidth=0.85
    )
    
    # Add labels and title
    plt.xlabel('Steering Angle', fontsize=12, labelpad=10)
    plt.ylabel('Frequency', fontsize=12, labelpad=10)
    plt.title('Steering Angle Distribution', fontsize=14, pad=20)
    
    # Adjust y-axis limit
    max_frequency = np.max(counts)
    plt.ylim(0, max_frequency * 1.1)
    
    # Add grid
    plt.grid(axis='y', alpha=0.4)
    
    # Save the figure (high quality)
    plt.savefig('steering_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    
    print("Histogram saved as 'steering_histogram.png'")

if __name__ == '__main__':
    plot_steering_histogram()