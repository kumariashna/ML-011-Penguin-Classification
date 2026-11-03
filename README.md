# 🐧 ML-011: Penguin Species Classification

## 1. Project Overview

Penguin Species Classification is a Machine Learning project that predicts the species of a penguin using its physical measurements.

The project uses the Palmer Penguins dataset, which contains biological measurements of different penguin species.

It demonstrates how Machine Learning classification algorithms can identify patterns in real-world biological data.

## 2. Project Information

* Project ID: ML - 011
* Project Name: Penguin Species Classification
* Domain: Wildlife / Machine Learning
* Type: Supervised Machine Learning
* Learning Category: Classification
* Language: Python

## 3. Project Objectives

* Understand a real-world biological dataset.
* Clean and preprocess missing data.
* Explore penguin measurements.
* Visualize differences between species.
* Train Machine Learning classification models.
* Compare model performance.
* Predict penguin species using new measurements.

## 4. Dataset Information

Dataset: Palmer Penguins Dataset

Species included:

1. Adelie
2. Chinstrap
3. Gentoo

Important features:

| Feature           | Description                       |
| ----------------- | --------------------------------- |
| bill_length_mm    | Bill length in millimeters        |
| bill_depth_mm     | Bill depth in millimeters         |
| flipper_length_mm | Flipper length in millimeters     |
| body_mass_g       | Body mass in grams                |
| island            | Island where penguin was observed |
| sex               | Recorded biological sex           |
| species           | Target variable                   |

Dataset Source:
https://github.com/allisonhorst/palmerpenguins

## 5. Technologies Used

* Python
* Jupyter Notebook
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib

## 6. Machine Learning Algorithms

### Decision Tree Classifier

A tree-based classification algorithm that makes predictions using feature-based decisions.

### K-Nearest Neighbors (KNN)

A classification algorithm that predicts a class using the nearest data points.

### StandardScaler

Used to scale numerical features before training the scaled KNN model.

## 7. Project Workflow

1. Import Python libraries.
2. Load the Palmer Penguins dataset.
3. Explore dataset structure.
4. Identify missing values.
5. Remove missing and duplicate records.
6. Separate features and target.
7. Encode categorical variables.
8. Split training and testing data.
9. Perform Exploratory Data Analysis.
10. Visualize penguin measurements.
11. Train Decision Tree and KNN models.
12. Evaluate model performance.
13. Generate confusion matrices.
14. Compare model accuracy.
15. Analyze feature importance.
16. Save the best-performing model.
17. Test predictions using new measurements.

## 8. Model Evaluation

The project evaluates models using:

* Accuracy Score
* Precision
* Recall
* F1-Score
* Confusion Matrix

Three model configurations are compared:

* Decision Tree
* KNN
* Scaled KNN

The best model is selected based on test accuracy.

Actual accuracy values are available in the Jupyter Notebook.

## 9. Project Outputs

* Cleaned penguin dataset
* Species distribution chart
* Bill length comparison
* Flipper length comparison
* Body mass comparison
* Correlation heatmap
* Feature relationship visualization
* Model accuracy comparison
* Confusion matrices
* Feature importance chart
* Saved Machine Learning model

## 10. Project Structure

```text
ML-011-Penguin-Classification/
│
├── app.py
├── data/
├── images/
├── index.html
├── models/
│   ├── encoder.pkl
│   ├── penguin_model.pkl
│   └── scaler.pkl
├── ML_011_Penguin_Classification.ipynb
├── train_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 11. Run the Streamlit Web App

Open the project folder in VS Code, then open a PowerShell terminal in that folder.

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Streamlit opens the app at `http://localhost:8501`. In VS Code, you can also run
these commands from **Terminal → New Terminal** after opening this project.

The saved model and encoder are included in `models/`. To rebuild them, run
`python train_model.py` from the project folder. This uses the original
notebook's cleaned data, feature encoding and stratified train/test split, and
verifies that the existing scaler matches before reusing it.

The prediction form asks for bill length, bill depth, flipper length, body
mass, island and sex. The original model also uses observation year, so the app
uses 2008 (the middle year in the dataset) as a fixed value rather than asking
for an additional input.

The original analysis remains in `ML_011_Penguin_Classification.ipynb` and can
be opened in VS Code with the Jupyter extension.

## 12. Learning Outcomes

Through this project, learners understand:

* Dataset exploration
* Data cleaning
* Missing-value handling
* Feature encoding
* Training and testing datasets
* Exploratory Data Analysis
* Classification algorithms
* Feature scaling
* Model evaluation
* Model saving and prediction

## 13. Disclaimer

This project is developed for educational and Machine Learning practice purposes. Predictions are based on the available dataset and should not be treated as professional biological identification.

## 14. Team

AI Engineer | Full Stack Developer

Project ID: ML-011

Domain: Machine Learning

---

**Project Status:** Educational Machine Learning Project
