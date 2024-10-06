# Employee_turnover_prediction
Use a dataset of employee information and build a model that predicts which employees are most likely to leave the company<br>


This project aims to predict employee attrition using machine learning models. The dataset contains employee-related data, and the goal is to predict whether an employee will leave the company ("Left"). The project uses Decision Tree and Random Forest classifiers to achieve this, along with data preprocessing techniques.

# Project Structure

**task 2/WA_Fn-UseC_-HR-Employee-Attrition.csv:** The dataset used for training and testing the models.<br />
**attrition_analysis.py:** Main Python script that handles data preprocessing, model training, evaluation, and visualization.<br />
**README.md:** This file providing an overview of the project.<br />

# Getting Started 
**Prerequisites**

Ensure you have the following installed:<br>

Python 3.x<br />
Pandas<br />
NumPy<br />
Matplotlib<br />
Seaborn<br />
Scikit-learn<br />

Install the required Python libraries using pip:<br>

pip install pandas numpy matplotlib seaborn scikit-learn<br>

# Dataset
The dataset used in this project contains features related to employee demographics, performance, and company structure. The target variable is "Left", which indicates whether an employee has left the company.

# Running the Project

Clone this repository.<br>
Ensure the dataset is in the task 2/ directory.<br>
Run the attrition_analysis.py script.<br>
bash
Copy code
python attrition_analysis.py
## Key Steps in the Script
**Data Loading:** The dataset is loaded using Pandas.<br>
**Exploratory Data Analysis (EDA):**<br>
Display the first few rows of the dataset (head()).<br>
Show basic statistics of the dataset (describe()).<br>
A heatmap is generated to visualize feature correlations.<br>
## Data Preprocessing:
One-hot encoding is applied to categorical variables (get_dummies()).<br>
Data is split into features (X) and target (y).<br>
The dataset is split into training and testing sets using train_test_split().<br>
Feature scaling is performed using StandardScaler().<br>
## Model Training:
A Decision Tree Classifier is trained using the fit() method.
## Model Evaluation:
The accuracy of the Decision Tree model is printed.<br>
A confusion matrix is plotted to visualize the performance.<br>
A classification report (precision, recall, f1-score) is generated.<br>
## Random Forest Classifier:
A Random Forest Classifier is also trained and evaluated similarly, with its accuracy compared to the Decision Tree.


