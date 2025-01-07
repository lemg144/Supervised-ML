# Binary Classification with a Tabular Stroke Prediction Dataset

## About the Dataset

The dataset for this competition (both train and test) was generated from a deep learning model trained on the Stroke Prediction Dataset. Feature distributions are close to, but not exactly the same, as the original. Feel free to use the original dataset as part of this competition, both to explore differences as well as to see whether incorporating the original in training improves model performance.

[Dataset Link](https://www.kaggle.com/competitions/playground-series-s3e2/data)


## Requirements

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn
- xgboost
- catboost
- lightgbm

### Running the Jupyter Notebook

1. Open `Stroke Classification.ipynb` in Jupyter Notebook or JupyterLab.
2. Run the cells sequentially to execute the code.

## Project Workflow

1. **Data Reading and Cleaning**:
    - Load the dataset.
    - Perform data cleaning and preprocessing.

2. **Exploratory Data Analysis (EDA)**:
    - Visualize the data distribution.
    - Analyze the features.

3. **Feature Engineering**:
    - One-hot encode categorical features.
    - Handle imbalanced data using undersampling.

4. **Model Training and Evaluation**:
    - Train models using XGBoost, CatBoost, and LightGBM.
    - Evaluate models using cross-validation and ROC curves.
    - Display confusion matrix and classification report.

## Results

The project evaluates multiple models and selects the best-performing one based on ROC AUC scores. The chosen model was XGBoost, which was fine-tuned using GridSearchCV. The final model predicted with 72% accuracy.
