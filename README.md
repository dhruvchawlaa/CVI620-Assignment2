# Self-Driving Car Simulation Project Using CNN

## CVI620 Final Project, Winter 2025

### Overview
This project develops a neural network model to control a self-driving car by predicting steering angles from front-camera images. The model is trained using data collected from the Udacity self-driving car simulator and tested in the same environment. The project includes data collection, preprocessing, augmentation, model training, and testing, following the guidelines provided in the assignment.

### Team Members
| Name            | Email                   |
|-----------------|-------------------------|
| Dhruv Chawla    | dchawla3@myseneca.ca    |
| Aashna Kundra   | akundra5@myseneca.ca    |
| Saeed Bafana    | sbafana@myseneca.ca     |

### Project Structure
- **Driving Log Data/**: Contains the dataset (`driving_log.csv` and `IMG/` folder) collected from the simulator.
- **plot_steering_histogram.py**: Script to plot the histogram of steering angles for dataset review.
- **steering_histogram.png**: Generated histogram of steering angles.
- **self_driving_car.py**: Main script for preprocessing, augmentation, training, and plotting training loss.
- **TestSimulation.py**: Script to test the trained model in the simulator.
- **demo.mp4**: Screen recording of the trained model running in the simulator.
- **README.md**: Project documentation.

### Setup Instructions
1. **Download Dataset**:
   - Dataset available at: [https://mega.nz/file/7dIjyDBA#9V-wgafKWrL3vH-i3gmuRSIBxGdhR8m78qDKPIY8aVQ](https://mega.nz/file/7dIjyDBA#9V-wgafKWrL3vH-i3gmuRSIBxGdhR8m78qDKPIY8aVQ)
   
2. **Create and Activate the Environment**:
   ```bash
   conda create --name as2_venv --file package_list.txt
   conda activate as2_venv