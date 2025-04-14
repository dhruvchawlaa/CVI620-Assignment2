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
- **data_augmentation.py**: Augments images with transformations like flipping and brightness adjustment.
- **data_preprocessing.py**: Loads, processes, and augments data.
- **plot_steering_histogram.py**: Script to plot the histogram of steering angles for dataset review.
- **steering_histogram.png**: Generated histogram of steering angles.
- **TestSimulation.py**: Script to test the trained model in the simulator.
- **train_model.py**: Trains the model using augmented data.
- **demo.mp4**: Screen recording of the trained model running in the simulator.
- **README.md**: Project documentation.

### Setup Instructions
1. **Download Dataset**:
   - Dataset available at: [https://mega.nz/file/ScRwxIab#w6fcn1w8eQ2YUs8_cjoU5obIaVD-0iK5Mme2VZMWUjE)
   
2. **Create and Activate the Environment**:
   ```bash
   conda create --name as2_venv --file package_list.txt
   conda activate as2_venv
