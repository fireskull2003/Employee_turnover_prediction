# Employee_turnover_prediction
Use a dataset of employee information and build a model that predicts which employees are most likely to leave the company


This project aims to predict employee attrition using machine learning models. The dataset contains employee-related data, and the goal is to predict whether an employee will leave the company ("Left"). The project uses Decision Tree and Random Forest classifiers to achieve this, along with data preprocessing techniques.

Project Structure
task 2/WA_Fn-UseC_-HR-Employee-Attrition.csv: The dataset used for training and testing the models.
attrition_analysis.py: Main Python script that handles data preprocessing, model training, evaluation, and visualization.
README.md: This file providing an overview of the project.
Getting Started
Prerequisites<br />

Ensure you have the following installed:<br />

Python 3.x
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Install the required Python libraries using pip:

pip install pandas numpy matplotlib seaborn scikit-learn

# Dataset
The dataset used in this project contains features related to employee demographics, performance, and company structure. The target variable is "Left", which indicates whether an employee has left the company.

# Running the Project

Clone this repository.
Ensure the dataset is in the task 2/ directory.
Run the attrition_analysis.py script.
bash
Copy code
python attrition_analysis.py
Key Steps in the Script
Data Loading: The dataset is loaded using Pandas.
Exploratory Data Analysis (EDA):
Display the first few rows of the dataset (head()).
Show basic statistics of the dataset (describe()).
A heatmap is generated to visualize feature correlations.
Data Preprocessing:
One-hot encoding is applied to categorical variables (get_dummies()).
Data is split into features (X) and target (y).
The dataset is split into training and testing sets using train_test_split().
Feature scaling is performed using StandardScaler().
Model Training:
A Decision Tree Classifier is trained using the fit() method.
Model Evaluation:
The accuracy of the Decision Tree model is printed.
A confusion matrix is plotted to visualize the performance.
A classification report (precision, recall, f1-score) is generated.
Random Forest Classifier:
A Random Forest Classifier is also trained and evaluated similarly, with its accuracy compared to the Decision Tree.


