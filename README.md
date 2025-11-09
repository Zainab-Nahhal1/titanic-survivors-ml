# Titanic Survival Prediction


## Project Overview
This project predicts whether a passenger survived the Titanic disaster using structured passenger data. It demonstrates a complete machine learning workflow, including:

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Model comparison and selection
- Hyperparameter tuning
- Evaluation and final prediction generation

**Dataset:** [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data)



## Objectives
- Build a predictive model to classify survival.
- Explore and compare multiple machine learning algorithms.
- Select and fine-tune the best-performing model.
- Evaluate the model on training data and generate test predictions.


## Dataset Description
The dataset contains the following columns:

| Column       | Description |
|--------------|-------------|
| PassengerId  | Unique ID for each passenger |
| Survived     | Survival status (0 = No, 1 = Yes) |
| Pclass       | Ticket class (1, 2, 3) |
| Name         | Passenger name |
| Sex          | Gender |
| Age          | Age in years |
| SibSp        | Number of siblings/spouses aboard |
| Parch        | Number of parents/children aboard |
| Ticket       | Ticket number |
| Fare         | Passenger fare |
| Cabin        | Cabin number |
| Embarked     | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |

---

## Methodology

### 1. Data Cleaning
- Remove duplicates
- Fill missing `Age` with median
- Fill missing `Embarked` with mode
- Fill missing `Fare` with median
- Drop `Cabin` due to too many missing values

### 2. Feature Engineering
- Encode `Sex` as numeric (0 = male, 1 = female)
- One-hot encode `Embarked` (`Embarked_C`, `Embarked_Q`, `Embarked_S`)

### 3. Model Selection
- Models tested:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Random Forest
  - Decision Tree
  - K-Nearest Neighbors (KNN)
  - Gradient Boosting
- Used 5-fold cross-validation
- **Best model:** Gradient Boosting (~83% CV accuracy)

### 4. Hyperparameter Tuning
- Grid search over:
  - `n_estimators`: [100, 200, 300]
  - `learning_rate`: [0.01, 0.05, 0.1]
  - `max_depth`: [3, 4, 5]
- Best parameters:
- Best parameters:
n_estimators=300, learning_rate=0.05, max_depth=4

### 5. Evaluation Metrics
- Accuracy: 88.9%
- Precision: 92%
- Recall: 78%
- F1-score: 84%
- Confusion Matrix:
[[526 23]
[ 76 266]]

---

## Results
- Final predictions are saved as:  
results/TitanicFinalPrediction.csv
- Gradient Boosting provides the best balance between precision and recall.


---

## How to Run

1. Place `train.csv` and `test.csv` in the `data/` folder.
2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
# Activate environment:
# macOS/Linux
source venv/bin/activate
# Windows
venv\Scripts\activate

pip install -r requirements.txt

##Run the pipeline:
python main.py
# or using Makefile
make train