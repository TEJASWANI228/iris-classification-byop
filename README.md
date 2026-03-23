# 🌸 Iris Flower Species Classifier

An interactive machine learning web application that predicts 
iris flower species using 3 different ML models.

## What this project does
- Compares 3 ML models: Decision Tree, Random Forest, KNN
- Beautiful pastel web interface to enter flower measurements
- Shows prediction confidence with probability breakdown
- Validates input measurements for realistic values
- Achieves 100% accuracy on the Iris dataset

## How to run this project

### Step 1: Install Python
Download from python.org (version 3.9 or above)

### Step 2: Install required libraries
pip install scikit-learn pandas matplotlib flask numpy

### Step 3: Run the ML script (optional)
python iris_project.py

### Step 4: Run the web application
python app.py

### Step 5: Open browser
Go to http://127.0.0.1:5000

## Project Files
- iris_project.py — Core ML code, data exploration, model training
- app.py — Flask web application with interactive UI
- iris_scatter_plot.png — Data visualization

## Models Used
| Model | Accuracy |
|---|---|
| Decision Tree | 100% |
| Random Forest | 100% |
| KNN | 100% |

Note: All models achieve 100% on Iris because it is a 
small, clean, and well-separated dataset. On complex 
real-world data, Random Forest would outperform the others.

## What I learned
- How to load and explore a real ML dataset
- How Decision Tree, Random Forest and KNN work differently
- How to evaluate and compare multiple ML models
- How to build a web interface using Flask
- How to validate user inputs for realistic predictions
- How to visualize data using matplotlib

## Technologies used
- Python 3.9
- scikit-learn
- pandas
- matplotlib
- Flask
- numpy
