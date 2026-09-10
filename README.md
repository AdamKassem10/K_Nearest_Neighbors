# K-Nearest Neighbors (K-NN)

Predicting whether a customer will purchase a product based on age and estimated salary.

## Overview

Say we've identified two categories in our data and add a new data point, how do we know which category it falls into? K-NN deals with this by looking at the points closest to the new one and letting them "vote" on its category.

How it works:

1. Choose the number K of neighbours (a common default is K = 5)
2. Take the K nearest neighbours of the new data point, according to the Euclidean distance (Manhattan or another distance metric can also be used)
3. Among these K neighbours, count the number of data points in each category
4. Assign the new data point to the category with the most neighbours

Euclidean distance:

d = sqrt((x2 - x1)^2 + (y2 - y1)^2)


## Dataset

`Social_Network_Ads.csv` contains the following columns:
- `Age`
- `EstimatedSalary`
- `Purchased` (0 = did not purchase, 1 = purchased)

## What the script does

1. Imports the dataset and splits it into a training set and a test set (75/25 split)
2. Applies feature scaling to Age and Estimated Salary
3. Trains a K-NN classifier with K = 5 on the training set
4. Makes a single prediction for a 30 year old with a salary of 87,000
5. Predicts results on the test set and compares predicted vs actual outcomes
6. Builds a confusion matrix and calculates accuracy
7. Visualizes the decision boundary on both the training set and the test set

## Results

Confusion Matrix:

[[64 4]
[ 3 29]]


**Accuracy: 93%**

This is a noticeable improvement over the Logistic Regression model on the same dataset, which reached 89% accuracy, showing how a different algorithm can better capture the non-linear boundary between the two classes.

## Tech used

- Python
- NumPy
- Pandas
- Matplotlib
- scikit-learn

## How to run

pip install numpy pandas matplotlib scikit-learn
python k_nearest_neighbors.py
