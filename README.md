
# Calories Burned Prediction

This project aims to predict the number of calories burned during a workout using biological and exercise data such as age, gender, height, weight, duration of workout, heart rate, and body temperature. By utilizing various machine learning models, including Linear Regression, Random Forest, XGBoost, and Ridge and Lasso Regression, the project explores and compares predictive models to estimate calorie expenditure. The project also incorporates extensive data visualization to understand the relationships between the variables.


## Project Motivation

As a pretty big gym enjoyer, accurately predicting calories burned during a workout is important for me to manage my weight, fitness levels, and overall health. This project aims to provide a machine learning-based approach to estimate calories burned based on common workout and biological data. The insights from this project can be used in personal fitness tracking applications or integrated with wearable fitness devices.
## Data

The project uses two CSV files:

**exercise.csv:** Contains user data on biological measures and exercise metrics (age, gender, height, weight, heart rate, duration, etc.). 

**calories.csv:** Contains the number of calories burned by users for specific workouts

## Key variables:

**Age:** User’s age in years. \
**Gender:** User’s gender (male or female). \
**Duration:** Length of the workout session (in minutes). \
**Heart_Rate:** User’s heart rate during the workout (beats per minute). \
**Calories:** Number of calories burned during the workout.

## Dependencies

The project requires the following libraries:

- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- xgboost

To import these libraries run the following:

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
```


## Exploratory Data Analysis


**Distributions by Gender:** Males generally burn more calories, have higher heart rates, and weigh more than females. This difference is visualized in the distribution plots.

**Scatterplots:** Relationships between key features (e.g., duration, heart rate, and calories) were visualized, color-coded by gender.

**Correlation Heatmap:** Showed strong positive correlations between calories burned, duration, and heart rate.
## Model Evaluation

The following machine learning models were used to predict calories burned:

**Linear Regression:** Baseline model with moderate performance. 

**Random Forest:** Performed well, comparable to XGBoost in capturing non linear relationships. 

**XGBoost:** Best-performing model with the lowest MSE and highest R² score.

**Ridge and Lasso Regression:** Regularization techniques that performed reasonably well but were outperformed by ensemble models.


| Model | MSE     | R^2 |
| :-------- | :------- | :------------------------- |
| Linear Regression|131.995746|0.967294|
| Random Forest|7.109941|0.998238|
| XGBoost|4.568956|0.998868|
| Ridge Regression|132.001967|0.967292|
| Lasso Regression|133.082123|0.967025|


## Future Work
Future improvements could include feature engineering by creating interaction terms (e.g., duration * heart rate) or including additional workout-related features (e.g., workout intensity). Hyperparameter tuning of models such as Random Forest and XGBoost would further optimize their performance. Additionally, expanding the dataset with more diverse demographics and integrating the model into a real-time fitness application would enhance the practical use of the project.

**Thanks for reading!**
